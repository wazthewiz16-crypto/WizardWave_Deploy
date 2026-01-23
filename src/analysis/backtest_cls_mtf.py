# !/usr/bin/env python3
"""
CLS MTF Strategy Backtester & Validator.
Checks signal generation across Daily/Weekly timeframes using synthetic resampling.
"""
import sys
import os
import pandas as pd
import yfinance as yf
import json
from datetime import datetime
import pandas_ta as ta

# Add Project Root to Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_cls import CLSRangeStrategy

def fetch_data(symbol, interval='1h', period='720d'):
    # Correction for Tickers
    ticker = symbol.replace('/', '-')
    if 'USDT' in ticker:
        ticker = ticker.replace('USDT', 'USD')
    
    print(f"  Fetching {ticker} ({interval})...")
    try:
        df = yf.Ticker(ticker).history(interval=interval, period=period)
        if df.empty:
            print("    Warning: Empty dataframe")
            return pd.DataFrame() # ... (rest of function)
            
        df.reset_index(inplace=True)
        # Standardize columns
        df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        # TZ Handling
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_localize(None)
            df.set_index('datetime', inplace=True)
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"    Error fetching: {e}")
        return pd.DataFrame()

def run_cls_backtest():
    print("=== CLS MTF STRATEGY BACKTEST ===")
    
    # Load Config
    with open('strategy_config_experimental.json', 'r') as f:
        config = json.load(f)
        
    assets = config['assets']
    
    # Results Container
    results = []

    for asset in assets:
        print(f"\nProcessing {asset}...")
        
        # --- TEST 1: DAILY CLS (Source 1H) ---
        print("  [Mode: Daily CLS from 1H]")
        df_1h = fetch_data(asset, interval='1h', period='1y') # 1 Year of 1H data
        
        # Custom Params based on Asset Class
        atr_mult = 1.5
        vol_mult = 1.2
        
        if '=X' in asset: # Forex
            print("    [Forex Detected] Using Specialized Forex Logic...")
            atr_mult = 1.0 # Not used in apply_mtf_forex but kept for consistency
        
        if not df_1h.empty:
            # Resample HTF (Daily)
            df_daily_syn = CLSRangeStrategy.resample_daily_from_1h(df_1h)
            
            if not df_daily_syn.empty:
                # Apply Strategy
                strat_d = CLSRangeStrategy(swing_window=10, atr_multiplier=atr_mult, volume_multiplier=vol_mult)
                
                if '=X' in asset:
                    # Use the new specialized method
                    df_res_d = strat_d.apply_mtf_forex(df_daily_syn, df_1h)
                else:
                    df_res_d = strat_d.apply_mtf(df_daily_syn, df_1h) # HTF=Daily, LTF=1H
                
                # Analyze Signals
                signals_d = df_res_d[df_res_d['signal_type'] != 'NONE']
                print(f"    Daily Signals: {len(signals_d)}")
                if not signals_d.empty:
                    # Filter for Dec 2025 to check user's specific trade
                    dec_2025 = signals_d[(signals_d.index >= '2025-12-01') & (signals_d.index <= '2025-12-31')]
                    print(f"    Dec 2025 Signals: {len(dec_2025)}")
                    if not dec_2025.empty:
                        print(dec_2025[['close', 'signal_type', 'range_high', 'range_low']])
                    else:
                        print(signals_d[['close', 'signal_type', 'range_high', 'range_low']].tail(3))
                    
                    results.append({'Asset': asset, 'Mode': 'Daily', 'Signals': len(signals_d)})
                else:
                    # Debug: Why no signals?
                    # Check if any HTF Ranges were defined
                    ranges_exist = df_res_d['range_high'].notna().sum()
                    print(f"    Debug: Active Ranges defined on {ranges_exist} bars (out of {len(df_res_d)})")
            else:
                print("    Failed to resample Daily")
        
        # --- TEST 2: WEEKLY CLS (Source 4H) ---
        print("  [Mode: Weekly CLS from 4H]")
        df_4h = fetch_data(asset, interval='1h', period='720d') # Using 1h and resampling to 4h is safer with yfinance sometimes, or just fetch 1h and aggregate to Weekly?
        
        if not df_4h.empty:
            # Resample HTF (Weekly)
            df_weekly_syn = CLSRangeStrategy.resample_weekly_from_4h(df_4h) # Works on 1h too
            
            # Use 4H as LTF
            df_trade_4h = df_4h.resample('4h').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()

            if not df_weekly_syn.empty and not df_trade_4h.empty:
                # Weekly moves are massive. Lower thresholds.
                strat_w = CLSRangeStrategy(swing_window=3, atr_multiplier=1.0, volume_multiplier=vol_mult)
                df_res_w = strat_w.apply_mtf(df_weekly_syn, df_trade_4h) # HTF=Weekly, LTF=4H
                
                signals_w = df_res_w[df_res_w['signal_type'] != 'NONE']
                print(f"    Weekly Signals: {len(signals_w)}")
                if not signals_w.empty:
                    print(signals_w[['close', 'signal_type', 'range_high', 'range_low']].tail(3))
                    results.append({'Asset': asset, 'Mode': 'Weekly', 'Signals': len(signals_w)})
                else:
                     ranges_exist = df_res_w['range_high'].notna().sum()
                     print(f"    Debug: Active Ranges defined on {ranges_exist} bars")
                     
        # --- TEST 3: MONTHLY CLS (Source 4H) ---
        print("  [Mode: Monthly CLS from 4H]")
        # We reuse df_4h (2 years of data)
        if not df_4h.empty:
            # Resample HTF (Monthly)
            # 4H -> Daily -> Monthly (safest chain)
            df_trade_4h = df_4h.resample('4h').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            df_daily_for_m = CLSRangeStrategy.resample_daily_from_1h(df_4h) # 1H/4H works same for this
            df_monthly_syn = CLSRangeStrategy.resample_monthly_from_daily(df_daily_for_m)
            
            if len(df_monthly_syn) > 5:
                # Monthly needs very low swing window because 24 months = 24 bars.
                # Window 3 is 3 months.
                strat_m = CLSRangeStrategy(swing_window=3, atr_multiplier=1.0, volume_multiplier=vol_mult)
                df_res_m = strat_m.apply_mtf(df_monthly_syn, df_trade_4h) # HTF=Monthly, LTF=4H
                
                signals_m = df_res_m[df_res_m['signal_type'] != 'NONE']
                print(f"    Monthly Signals: {len(signals_m)}")
                if not signals_m.empty:
                    print(signals_m[['close', 'signal_type', 'range_high', 'range_low']].tail(3))
                    results.append({'Asset': asset, 'Mode': 'Monthly', 'Signals': len(signals_m)})

        # --- TEST 4: HYBRID MONTHLY (Source Daily 5y -> Monthly, Entry 1H) ---
        print("  [Mode: Hybrid Monthly CLS (Daily 5y Src)]")
        # Fetch Deep History Daily
        df_daily_deep = fetch_data(asset, interval='1d', period='5y')
        
        if not df_daily_deep.empty and not df_1h.empty:
            df_monthly_deep = CLSRangeStrategy.resample_monthly_from_daily(df_daily_deep)
            
            if len(df_monthly_deep) > 12:
                # We have enough data for Monthly ATR/Swing
                strat_hm = CLSRangeStrategy(swing_window=3, atr_multiplier=1.2, volume_multiplier=vol_mult)
                df_res_hm = strat_hm.apply_mtf(df_monthly_deep, df_1h) # HTF=Monthly (Deep), LTF=1H
                
                signals_hm = df_res_hm[df_res_hm['signal_type'] != 'NONE']
                print(f"    Hybrid Monthly Signals: {len(signals_hm)}")
                if not signals_hm.empty:
                    print(signals_hm[['close', 'signal_type', 'range_high', 'range_low']].tail(3))
                    results.append({'Asset': asset, 'Mode': 'HybridMonthly', 'Signals': len(signals_hm)})
                else:
                    ranges_exist = df_res_hm['range_high'].notna().sum()
                    print(f"    Debug: Active Ranges defined on {ranges_exist} bars")

    print("\nTotal Results:", results)

if __name__ == "__main__":
    run_cls_backtest()

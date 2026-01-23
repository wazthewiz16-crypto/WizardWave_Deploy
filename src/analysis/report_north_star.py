# !/usr/bin/env python3
"""
North Star Strategy Performance Report.
Tests the "High Quality Only" CLS logic on Crypto, Forex, and Indices.
"""
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

# Add Project Root to Path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.strategies.strategy_cls import CLSRangeStrategy

# Assets to Test
ASSETS = {
    'Crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD'],
    'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
    'Indices': ['^GSPC', '^IXIC', '^DJI'], # SPX, Nasdaq, Dow
    'Metals': ['GC=F', 'SI=F']
}

def fetch_data(ticker, period='720d', interval='1h'):
    print(f"Fetching {ticker}...")
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df = df.rename(columns={'DeepHistory': 'open', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df.columns = [c.lower() for c in df.columns]
        
        # TZ Handling
        if df.index.tz is not None:
             df.index = df.index.tz_localize(None)
             
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def run_report():
    print("# North Star Strategy Performance Report\n")
    print("| Asset Class | Ticker | Signals (2Y) | Win Rate (Est) | Avg Return |")
    print("|---|---|---|---|---|")
    
    total_signals = 0
    
    for category, tickers in ASSETS.items():
        for ticker in tickers:
            df_1h = fetch_data(ticker)
            if df_1h.empty:
                print(f"| {category} | {ticker} | NO DATA | - | - |")
                continue
                
            # Create Daily HTF
            df_daily = CLSRangeStrategy.resample_daily_from_1h(df_1h)
            
            # Run Strategy
            strat = CLSRangeStrategy()
            df_res = strat.apply_north_star(df_daily, df_1h)
            
            # Analyze Results
            signals = df_res[df_res['signal_type'] != 'NONE'].copy()
            count = len(signals)
            total_signals += count
            
            if count > 0:
                # Estimate WR
                wins = 0
                total_return = 0.0
                
                for idx, row in signals.iterrows():
                    # Look ahead 48 bars (2 days)
                    future = df_res.loc[idx:].iloc[1:49] # Exclude entry bar
                    if future.empty: continue
                    
                    entry = row['close']
                    target = row['target_price']
                    stop = row['stop_loss']
                    
                    if row['signal_type'] == 'CLS_LONG':
                        hit_tp = future[future['high'] >= target]
                        hit_sl = future[future['low'] <= stop]
                        
                        tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                        sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                        
                        if tp_idx < sl_idx:
                            wins += 1
                            total_return += (target - entry) / entry
                        elif sl_idx < tp_idx:
                            total_return -= (entry - stop) / entry
                        else:
                            # Timed out
                            end_price = future['close'].iloc[-1]
                            total_return += (end_price - entry) / entry
                            
                    elif row['signal_type'] == 'CLS_SHORT':
                        hit_tp = future[future['low'] <= target]
                        hit_sl = future[future['high'] >= stop]
                        
                        tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                        sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                        
                        if tp_idx < sl_idx:
                            wins += 1
                            total_return += (entry - target) / entry
                        elif sl_idx < tp_idx:
                            total_return -= (stop - entry) / entry
                        else:
                            end_price = future['close'].iloc[-1]
                            total_return += (entry - end_price) / entry

                wr = (wins / count) * 100
                avg_ret = (total_return / count) * 100
                print(f"| {category} | {ticker} | {count} | {wr:.1f}% | {avg_ret:.2f}% |")
            else:
                print(f"| {category} | {ticker} | 0 | - | - |")
                
    print(f"\nTotal Signals Analyzed: {total_signals}")

if __name__ == "__main__":
    run_report()

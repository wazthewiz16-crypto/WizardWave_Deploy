# !/usr/bin/env python3
"""
North Star Strategy: 180-Day Return Simulator.
Calculates Cumulative PnL assuming 1% Risk per Trade.
"""
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

# Add Project Root to Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_cls import CLSRangeStrategy

# Assets to Test (Full Portfolio)
ASSETS = {
    'Crypto': [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 
        'BNB-USD', 'LINK-USD', 'ARB-USD', 'AVAX-USD', 'ADA-USD'
    ],
    'Forex': ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X'],
    'Indices': ['^NDX', '^GSPC', '^AXJO', 'DX-Y.NYB', '^DJI'],
    'Commodities': ['GC=F', 'SI=F', 'CL=F']
}

RISK_PER_TRADE = 0.01 # 1% of Account
START_CAPITAL = 10000

def fetch_data(ticker, period='200d', interval='1h'):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df = df.rename(columns={'DeepHistory': 'open', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return pd.DataFrame()

def run_simulation():
    print("# North Star Strategy: 180-Day Actual Returns (1% Risk)\n")
    print(f"**Period**: Last 180 Days | **Risk**: 1% per trade | **Setup**: High Quality Reversals\n")
    print("| Asset | Trades | Win Rate | Total Return (%) | Max DD (%) |")
    print("|---|---|---|---|---|")
    
    grand_total_r = 0
    
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=180)
    
    for category, tickers in ASSETS.items():
        for ticker in tickers:
            df_1h = fetch_data(ticker)
            if df_1h.empty: continue
            
            # Filter for 180 Days *Data* (but we need pre-data for indicators)
            # We apply strategy on full 200d, then filter signals
            
            df_daily = CLSRangeStrategy.resample_daily_from_1h(df_1h)
            strat = CLSRangeStrategy()
            df_res = strat.apply_north_star(df_daily, df_1h)
            
            # Filter Signals after Cutoff
            signals = df_res[(df_res['signal_type'] != 'NONE') & (df_res.index >= cutoff_date)].copy()
            
            if signals.empty:
                print(f"| {ticker} | 0 | - | 0.00% | 0.00% |")
                continue
                
            balance = START_CAPITAL
            equity_curve = [START_CAPITAL]
            wins = 0
            total_r = 0
            
            # Simple Trade Loop
            for idx, row in signals.iterrows():
                # Entry
                entry_price = row['close']
                stop_loss = row['stop_loss']
                target = row['target_price']
                
                # Calculate R (Risk Amount)
                risk_dist = abs(entry_price - stop_loss)
                if risk_dist == 0: continue # Invalid SL
                
                reward_dist = abs(target - entry_price)
                rr_ratio = reward_dist / risk_dist
                
                # Skip bad R:R trades? Let's say we take all valid ones for "Actual" test
                if rr_ratio < 0.5: continue 
                
                # Determine Outcome (Look ahead)
                future = df_res.loc[idx:].iloc[1:120] # 5 days max hold
                if future.empty: continue
                
                pnl = 0
                r_outcome = 0
                
                if row['signal_type'] == 'CLS_LONG':
                    hit_tp = future[future['high'] >= target]
                    hit_sl = future[future['low'] <= stop_loss]
                    
                    tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                    sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                    
                    if tp_idx < sl_idx: # WIN
                        wins += 1
                        r_outcome = rr_ratio # Won R multiples
                    elif sl_idx < tp_idx: # LOSS
                        r_outcome = -1.0 # Lost 1R
                    else: # TIMEOUT (Exit at close)
                        exit_price = future['close'].iloc[-1]
                        r_outcome = (exit_price - entry_price) / risk_dist
                
                elif row['signal_type'] == 'CLS_SHORT':
                    hit_tp = future[future['low'] <= target]
                    hit_sl = future[future['high'] >= stop_loss]
                    
                    tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                    sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                    
                    if tp_idx < sl_idx: # WIN
                        wins += 1
                        r_outcome = rr_ratio
                    elif sl_idx < tp_idx: # LOSS
                        r_outcome = -1.0
                    else:
                        exit_price = future['close'].iloc[-1]
                        r_outcome = (entry_price - exit_price) / risk_dist
                        
                # Update Balance
                # 1% Risk = balance * 0.01
                risk_amt = balance * RISK_PER_TRADE
                pnl = risk_amt * r_outcome
                balance += pnl
                equity_curve.append(balance)
                total_r += r_outcome
            
            # Stats
            equity_curve = np.array(equity_curve)
            total_return_pct = ((balance - START_CAPITAL) / START_CAPITAL) * 100
            
            # Max DD
            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / peak
            max_dd = dd.max() * 100
            
            wr = (wins / len(signals)) * 100
            
            grand_total_r += total_r
            print(f"| {ticker} | {len(signals)} | {wr:.1f}% | {total_return_pct:.2f}% | {max_dd:.2f}% |")
            
    print(f"\n**Portfolio Summary**")
    print(f"Total R-Multiples Generated: {grand_total_r:.2f}R")
    print(f"Estimated Portfolio Return (if trading all): {grand_total_r * 1:.2f}%")

if __name__ == "__main__":
    run_simulation()

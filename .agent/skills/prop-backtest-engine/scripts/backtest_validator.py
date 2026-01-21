
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json

# Add Project Root to Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)

try:
    from strategy_scalp import WizardScalpStrategy
    from feature_engine import calculate_ml_features
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- CONFIG ---
# Strict Prop Firm Filters
MAX_DAILY_DD_PCT = 0.039 # 3.9%
MAX_TOTAL_DD_PCT = 0.079 # 7.9%
MIN_PROFIT_FACTOR = 1.3
ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD']

def fetch_data(symbol, days=90):
    """Fetch 1h data for backtest."""
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d", interval="1h")
        if df.empty: return pd.DataFrame()
        
        df = df.reset_index()
        df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df.set_index('datetime', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def run_simulation():
    print("=== PROP FIRM BACKTEST VALIDATOR (90 Days) ===")
    print(f"Strategy: Improved WizardScalp (Crypto)\n")
    
    overall_stats = []
    
    # Init Strategy
    strat = WizardScalpStrategy(lookback=8)
    
    for symbol in ASSETS:
        print(f"Testing {symbol}...")
        df = fetch_data(symbol)
        
        if df.empty:
            print("  No Data.")
            continue
            
        # 1. Apply Strategy
        df = strat.apply(df)
        
        # 2. Simulate Trades
        initial_balance = 100000.0
        equity = initial_balance
        peak_equity = initial_balance
        daily_start_equity = initial_balance
        current_day = None
        
        balance_curve = [initial_balance]
        daily_dd_breach = False
        max_dd_breach = False
        
        trades = []
        position = 0 # 0, 1 (Long), -1 (Short)
        entry_price = 0.0
        
        # Filter Logic (Simulating Signal Execution)
        for idx, row in df.iterrows():
            # Day Check for Daily DD
            day_str = str(idx.date())
            if current_day != day_str:
                daily_start_equity = equity
                current_day = day_str
            
            # Check Daily DD
            day_dd = (daily_start_equity - equity) / daily_start_equity
            if day_dd > MAX_DAILY_DD_PCT:
                daily_dd_breach = True
                
            # Check Max DD
            if equity > peak_equity: peak_equity = equity
            total_dd = (peak_equity - equity) / peak_equity
            if total_dd > MAX_TOTAL_DD_PCT:
                max_dd_breach = True
                
            # Trading Logic
            sig = row.get('signal_type', 'NONE')
            price = row['close']
            
            # EXIT
            if position != 0:
                pnl = 0
                closed = False
                
                # Exit Rule: Cloud Crossing (Scalp)
                # Or Hard TP/SL (1.5% / 0.75%)
                
                if position == 1: # Long
                    # Stop Loss (0.75% strict)
                    if row['low'] <= entry_price * 0.9925:
                        pnl = (entry_price * 0.9925 - entry_price) * (equity / entry_price) # Full port logic
                        equity += pnl
                        trades.append({'PnL': -0.75, 'Result': 'Loss'})
                        position = 0
                        closed = True
                    # Take Profit (1.5%)
                    elif row['high'] >= entry_price * 1.015:
                        pnl = (entry_price * 1.015 - entry_price) * (equity / entry_price)
                        equity += pnl
                        trades.append({'PnL': 1.5, 'Result': 'Win'})
                        position = 0
                        closed = True
                    # Trend Reversal
                    elif row['close'] < row['cloud_bottom']:
                         pnl = (price - entry_price) * (equity / entry_price)
                         equity += pnl
                         trades.append({'PnL': (price/entry_price - 1)*100, 'Result': 'Win' if pnl>0 else 'Loss'})
                         position = 0
                         closed = True
                         
                elif position == -1: # Short
                    # Stop Loss
                    if row['high'] >= entry_price * 1.0075:
                        pnl = (entry_price - entry_price * 1.0075) * (equity / entry_price)
                        equity += pnl
                        trades.append({'PnL': -0.75, 'Result': 'Loss'})
                        position = 0
                        closed = True
                    # Take Profit
                    elif row['low'] <= entry_price * 0.985:
                        pnl = (entry_price - entry_price * 0.985) * (equity / entry_price)
                        equity += pnl
                        trades.append({'PnL': 1.5, 'Result': 'Win'})
                        position = 0
                        closed = True
                     # Reversal
                    elif row['close'] > row['cloud_top']:
                         pnl = (entry_price - price) * (equity / entry_price)
                         equity += pnl
                         trades.append({'PnL': (entry_price/price - 1)*100, 'Result': 'Win' if pnl>0 else 'Loss'})
                         position = 0
                         closed = True
                
            # ENTRY (Only if Flat)
            if position == 0:
                if 'SCALP_LONG' in sig:
                    position = 1
                    entry_price = price
                elif 'SCALP_SHORT' in sig:
                    position = -1
                    entry_price = price
                    
        # Metrics
        total_trades = len(trades)
        wins = len([t for t in trades if t['Result'] == 'Win'])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = (equity - initial_balance) / initial_balance * 100
        
        status = "PASS"
        if daily_dd_breach: status = "FAIL (Daily DD)"
        elif max_dd_breach: status = "FAIL (Max DD)"
        elif total_pnl < 0: status = "FAIL (Unprofitable)"
        
        print(f"  Result: {status}")
        print(f"  PnL: {total_pnl:.2f}% | Win Rate: {win_rate:.1f}% ({total_trades} trades)")
        if daily_dd_breach: print("  [!] Breached Daily Drawdown Limit")
        
        overall_stats.append({'Asset': symbol, 'Status': status, 'PnL': total_pnl})

    print("\n=== SUMMARY REPORT ===")
    print(pd.DataFrame(overall_stats))

if __name__ == "__main__":
    run_simulation()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import Data Loader
loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agent/skills/market-data-manager/scripts"))
sys.path.append(loader_path)
from loader import load_data

# Import The Strategy Class
# Make sure your strategy_cls.py is in the python path or import it directly
from strategy_cls import CLSRangeStrategy

def run_cls_backtest(symbol='BTC/USDT', timeframe='1h'):
    print(f"\n[+] Loading Data for {symbol} ({timeframe})...")
    # 1. Load Data (Instant!)
    df = load_data(symbol, timeframe, asset_type='crypto' if 'USDT' in symbol else 'tradfi')
    
    if df.empty:
        print("[!] No data found.")
        return

    print(f"[+] Loaded {len(df)} candles.")
    
    # 2. Initialize Strategy
    strat = CLSRangeStrategy() 
    
    # 3. Apply Logic
    print("[*] Applying CLS Logic...")
    # Note: strategy_cls currently expects to calculate Daily ranges internally 
    # or expects '1d' data. 
    # If the strategy applies logic on HTF data first, we need to replicate that.
    # Let's see how strat.apply() works. 
    # Usually it takes 'df' and assumes it's the timeframe we trade on, 
    # but CLS needs Daily context. 
    
    # For this test, we mimic the logic:
    # We need Daily Data for the Ranges
    df_daily = load_data(symbol, '1d', 'crypto' if 'USDT' in symbol else 'tradfi')
    
    # Calculate Ranges on Daily
    # (We might need to refactor strategy_cls.py to separate range calc from signal gen if not already)
    
    # Let's use the strategy's apply method if it handles multi-timeframe 
    # OR manually invoke the logic if the class is rigid.
    
    # Assuming standard pattern: 
    # CLS typically:
    # 1. Calc Daily Sweep Levels
    # 2. Check 1H reclaim
    
    # RUNNING NATIVE APPLY FOR NOW
    results = strat.apply(df) 
    
    # 4. Filter Signals
    signals = results[results['signal_type'] != 'NONE'].copy()
    
    print(f"[=] Signals Found: {len(signals)}")
    if len(signals) > 0:
        print(signals[['close', 'signal_type']].tail(10))
        
        # 5. Simple PnL Simulation (Fixed Risk)
        # Assume 1% Risk, 2R Target
        wins = 0
        losses = 0
        
        # Very rough simulation loop
        for idx, row in signals.iterrows():
            entry = row['close']
            is_long = 'LONG' in row['signal_type']
            tp = entry * 1.02 if is_long else entry * 0.98
            sl = entry * 0.99 if is_long else entry * 1.01
            
            # Look forward 48 bars
            future = df.loc[idx:].iloc[1:49]
            if future.empty: continue
            
            if is_long:
                hit_tp = future['high'] >= tp
                hit_sl = future['low'] <= sl
            else:
                hit_tp = future['low'] <= tp
                hit_sl = future['high'] >= sl
                
            first_tp = hit_tp.idxmax() if hit_tp.any() else None
            first_sl = hit_sl.idxmax() if hit_sl.any() else None
            
            if first_tp and (not first_sl or first_tp < first_sl):
                wins += 1
            elif first_sl:
                losses += 1
                
        total = wins + losses
        wr = (wins/total)*100 if total > 0 else 0
        print(f"\n[WIN] Win Rate (Estimated 2R): {wr:.2f}% ({wins}/{total})")
        
        return {
            "Symbol": symbol,
            "Total": total,
            "Wins": wins, 
            "Losses": losses,
            "WinRate": wr
        }
    return None

if __name__ == "__main__":
    assets = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 
        '^GSPC', '^NDX', 'GC=F', 'SI=F', 'EURUSD=X', 'GBPUSD=X'
    ]
    results = []
    
    print(f"[*] Starting Portfolio Backtest on {len(assets)} assets...")
    
    for asset in assets:
        try:
            res = run_cls_backtest(asset, "1h")
            if res: results.append(res)
        except Exception as e:
            print(f"[!] Error on {asset}: {e}")
            
    # Summary
    print("\n" + "="*50)
    print(f"{'SYMBOL':<10} {'TRADES':<8} {'WIN RATE':<10} {'WINS':<6} {'LOSS':<6}")
    print("-" * 50)
    
    avg_wr = 0
    total_trades = 0
    
    for r in results:
        print(f"{r['Symbol']:<10} {r['Total']:<8} {r['WinRate']:>6.2f}%    {r['Wins']:<6} {r['Losses']:<6}")
        if r['Total'] > 0:
            avg_wr += r['WinRate'] * r['Total']
            total_trades += r['Total']
            
    if total_trades > 0:
        print("-" * 50)
        print(f"{'TOTAL':<10} {total_trades:<8} {avg_wr/total_trades:>6.2f}%")
        
    print("="*50)

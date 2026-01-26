import pandas as pd
import numpy as np
import os
import sys

# Import Data Loader
loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agent/skills/market-data-manager/scripts"))
sys.path.append(loader_path)
from loader import load_data

# Import Strategy
from strategy_cls import CLSRangeStrategy

def run_backtest(symbol):
    # 1. Load Data
    # HTF = 4H, LTF = 15m
    asset_type = 'crypto' if 'USDT' in symbol else 'tradfi'
    
    df_htf = load_data(symbol, '4h', asset_type)
    df_ltf = load_data(symbol, '15m', asset_type)
    
    if df_htf.empty or df_ltf.empty:
        return None
        
    # 2. Strategy
    # We might want shorter swing lookback for 4H? Default is 10.
    strat = CLSRangeStrategy(swing_window=10) 
    
    # 3. Apply Multi-Timeframe Logic
    try:
        df = strat.apply_4h_15m(df_htf, df_ltf)
    except Exception as e:
        # print(f"Error logic {symbol}: {e}")
        return None
        
    if 'signal_type' not in df.columns: return None
        
    signals = df[df['signal_type'] != 'NONE'].copy()
    
    if signals.empty:
        return {
            "Symbol": symbol, "Total": 0, "Wins": 0, "Losses": 0, "WinRate": 0
        }
        
    # 4. Simulate
    wins = 0
    losses = 0
    
    for idx, row in signals.iterrows():
        entry = row['close']
        s_type = row['signal_type']
        
        # 2R Targets
        tp = entry * 1.02 if 'LONG' in s_type else entry * 0.98
        sl = entry * 0.99 if 'LONG' in s_type else entry * 1.01
        
        # Look forward 48 bars (12 hours)
        future = df.loc[idx:].iloc[1:49]
        if future.empty: continue
        
        if 'LONG' in s_type:
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
    
    return {
        "Symbol": symbol,
        "Total": total,
        "Wins": wins, 
        "Losses": losses,
        "WinRate": wr
    }

if __name__ == "__main__":
    assets = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'XRP/USDT', 'ADA/USDT',
        '^GSPC', '^NDX', 'GC=F'
    ]
    
    print(f"\n[*] Running 4H/15m CLS Backtest on {len(assets)} assets...")
    print(f"{'SYMBOL':<10} {'TRADES':<8} {'WIN RATE':<10} {'WINS':<6} {'LOSS':<6}")
    print("-" * 50)
    
    total_trades = 0
    avg_wr = 0
    count = 0
    
    for asset in assets:
        res = run_backtest(asset)
        if res:
            print(f"{res['Symbol']:<10} {res['Total']:<8} {res['WinRate']:>6.2f}%    {res['Wins']:<6} {res['Losses']:<6}")
            if res['Total'] > 0:
                total_trades += res['Total']
                avg_wr += res['WinRate'] * res['Total']
        else:
            print(f"{asset:<10} N/A (No Data)")
            
    if total_trades > 0:
        print("-" * 50)
        print(f"{'TOTAL':<10} {total_trades:<8} {avg_wr/total_trades:>6.2f}%")
    print("=" * 50)

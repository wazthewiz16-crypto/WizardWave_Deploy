import pandas as pd
import numpy as np
import os
import sys
import json

# Import Utilities
loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agent/skills/market-data-manager/scripts"))
sys.path.append(loader_path)
from loader import load_data

# Import Strategies
from strategy_ichimoku import IchimokuStrategy
from strategy_ichimoku_pro import IchimokuProStrategy

def run_sim(df, pt_pct, sl_pct):
    signals = df[df['signal_type'] != 'NONE'].copy()
    if signals.empty: return []
    
    trades = []
    for idx, row in signals.iterrows():
        entry = row['close']
        s_type = row['signal_type']
        
        direction = 1 if 'LONG' in s_type else -1
        tp = entry * (1 + (pt_pct * direction))
        sl = entry * (1 - (sl_pct * direction))
        
        # Max hold 72 bars (3 days on 1h, 12 days on 4h)
        future = df.loc[idx:].iloc[1:73]
        if future.empty: continue
        
        if direction == 1:
            hit_tp = future['high'] >= tp
            hit_sl = future['low'] <= sl
        else:
            hit_tp = future['low'] <= tp
            hit_sl = future['high'] >= sl
            
        t_tp = hit_tp.idxmax() if hit_tp.any() else None
        t_sl = hit_sl.idxmax() if hit_sl.any() else None
        
        if t_tp and (not t_sl or t_tp < t_sl):
            trades.append(pt_pct - 0.001)
        elif t_sl:
            trades.append(-sl_pct - 0.001)
        else:
            ret = (future.iloc[-1]['close'] - entry)/entry if direction == 1 else (entry - future.iloc[-1]['close'])/entry
            trades.append(ret - 0.001)
            
    return trades

def main():
    # Focus on the 'Problem' assets from the master backtest
    assets = ['GC=F', 'SI=F', 'EURUSD=X', 'BTC/USDT']
    tf = '4h'
    
    print(f"\n{'='*70}\nICHIMOKU BATTLE: STANDARD VS KUMO KING (PRO)\n{'='*70}")
    print(f"{'ASSET':<12} | {'STANDARD RET':<15} {'KUMO KING RET':<15} {'WINNER'}")
    print("-" * 70)
    
    # Defaults
    pt = 0.02
    sl = 0.01

    total_std = 0
    total_pro = 0

    for symbol in assets:
        try:
            asset_type = 'crypto' if 'USDT' in symbol else 'tradfi'
            df = load_data(symbol, tf, asset_type)
            if df.empty: continue
                
            # 1. Standard
            std_strat = IchimokuStrategy(tenkan=20, kijun=60, span_b=120, displacement=30, adx_threshold=22)
            df_std = std_strat.apply_strategy(df)
            std_signals = df_std[df_std['signal_type'].notna()]
            std_trades = run_sim(df_std[df_std['signal_type'].notna()], pt, sl) if not std_signals.empty else []
            std_ret = sum(std_trades) * 100 if std_trades else 0
            
            # 2. Pro
            # Thickness and ADX are already defaults in the class or passed here
            pro_strat = IchimokuProStrategy(tenkan=20, kijun=60, span_b=120, displacement=30, adx_threshold=22, thickness_mult=0.6)
            df_pro = pro_strat.apply_strategy(df)
            pro_signals = df_pro[df_pro['signal_type'] != 'NONE']
            pro_trades = run_sim(df_pro[df_pro['signal_type'] != 'NONE'], pt, sl) if not pro_signals.empty else []
            pro_ret = sum(pro_trades) * 100 if pro_trades else 0
            
            winner = "PRO" if pro_ret > std_ret else "STD"
            print(f"{symbol:<12} | {std_ret:>12.1f}% {pro_ret:>12.1f}%   {winner} (Sigs: {len(std_signals)} vs {len(pro_signals)})")
            
            total_std += std_ret
            total_pro += pro_ret
            
        except Exception as e:
            pass

    print("-" * 70)
    print(f"{'TOTAL PORTFOLIO':<12} | {total_std:>13.1f}% {total_pro:>13.1f}%")
    final_winner = "KUMO KING (PRO)" if total_pro > total_std else "STANDARD"
    print(f"VERDICT: {final_winner}")
    print("="*70)

if __name__ == "__main__":
    main()

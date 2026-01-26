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
from strategy import WizardWaveStrategy
from strategy_wizard_pro import WizardWaveProStrategy

def run_champion_sim(df, tp_pct=0.09, sl_pct=0.033):
    """Original WizardWave Logic (Fixed SL/TP)"""
    signals = df[df['signal_type'] != 'NONE'].copy()
    trades = []
    
    for idx, row in signals.iterrows():
        entry = row['close']
        direction = 1 if 'LONG' in row['signal_type'] else -1
        
        tp = entry * (1 + (tp_pct * direction))
        sl = entry * (1 - (sl_pct * direction))
        
        # Max hold 50 bars
        future = df.loc[idx:].iloc[1:51]
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
            trades.append(tp_pct - 0.001)
        elif t_sl:
            trades.append(-sl_pct - 0.001)
        else:
            # Time exit
            ret = (future.iloc[-1]['close'] - entry)/entry if direction == 1 else (entry - future.iloc[-1]['close'])/entry
            trades.append(ret - 0.001)
            
    return trades

def main():
    assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', '^GSPC', 'GC=F']
    timeframes = ['4h', '12h']
    
    print(f"\n{'='*70}\nHEAD-TO-HEAD BATTLE: CHAMPION VS PRO-MAX\n{'='*70}")
    print(f"{'ASSET':<12} {'TF':<5} | {'CHAMPION RET':<15} {'PRO-MAX RET':<15} {'WINNER'}")
    print("-" * 70)
    
    champ_total_ret = 0
    pro_total_ret = 0
    
    for tf in timeframes:
        for symbol in assets:
            try:
                asset_type = 'crypto' if 'USDT' in symbol else 'tradfi'
                df = load_data(symbol, tf, asset_type)
                if df.empty or len(df) < 200: continue
                
                # 1. Champion
                champ_strat = WizardWaveStrategy()
                df_champ = champ_strat.apply(df)
                champ_trades = run_champion_sim(df_champ)
                champ_ret = sum(champ_trades) * 100
                
                # 2. Pro-Max
                pro_strat = WizardWaveProStrategy(atr_mult=2.5) # Dynamic SL
                pro_trades = pro_strat.run_simulation(df)
                pro_ret = sum(pro_trades) * 100
                
                winner = "PRO-MAX" if pro_ret > champ_ret else "CHAMPION"
                print(f"{symbol:<12} {tf:<5} | {champ_ret:>12.1f}% {pro_ret:>12.1f}%   {winner} (Signals: {len(champ_trades)} vs {len(pro_trades)})")
                
                champ_total_ret += champ_ret
                pro_total_ret += pro_ret
                
            except Exception as e:
                print(f"Error {symbol} {tf}: {e}")

    print("-" * 70)
    print(f"{'OVERALL TOTAL RESULT':<18} | {champ_total_ret:>12.1f}% {pro_total_ret:>12.1f}%")
    final_winner = "PRO-MAX" if pro_total_ret > champ_total_ret else "CHAMPION"
    print(f"FINAL VERDICT: {final_winner}")
    print("="*70)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# Import Utilities
loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agent/skills/market-data-manager/scripts"))
sys.path.append(loader_path)
from loader import load_data

# Import Strategies
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from strategy_cls import CLSRangeStrategy
from strategy_ichimoku import IchimokuStrategy
from strategy_ichimoku_pro import IchimokuProStrategy
from strategy_wizard_pro import WizardWaveProStrategy

def run_simulation(df, signal_col, tp_pct, sl_pct, time_limit=24):
    """Simple 2R or Fixed TP/SL Simulation"""
    signals = df[df[signal_col].notna() & (df[signal_col] != 'NONE')].copy()
    if signals.empty: return []
    
    trades = []
    for idx, row in signals.iterrows():
        entry = row['close']
        s_type = row[signal_col]
        
        # Determine TP/SL
        if 'LONG' in s_type:
            tp = entry * (1 + tp_pct)
            sl = entry * (1 - sl_pct)
        else:
            tp = entry * (1 - tp_pct)
            sl = entry * (1 + sl_pct)
            
        future = df.loc[idx:].iloc[1:time_limit+1]
        if future.empty: continue
        
        # Result
        if 'LONG' in s_type:
            hit_tp = future['high'] >= tp
            hit_sl = future['low'] <= sl
        else:
            hit_tp = future['low'] <= tp
            hit_sl = future['high'] >= sl
            
        t_tp = hit_tp.idxmax() if hit_tp.any() else None
        t_sl = hit_sl.idxmax() if hit_sl.any() else None
        
        pnl = 0
        if t_tp and (not t_sl or t_tp < t_sl):
            pnl = tp_pct
        elif t_sl:
            pnl = -sl_pct
        else:
            # Time exit
            exit_price = future.iloc[-1]['close']
            pnl = (exit_price - entry)/entry if 'LONG' in s_type else (entry - exit_price)/entry
            
        trades.append(pnl - 0.001) # Fees
        
    return trades

def main():
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
        
    assets = config['assets']
    models = config['models']
    
    results = []
    
    print(f"\n{'='*70}\nMASTER BACKTEST: ALL LIVE SIGNAL CONFIGS\n{'='*70}")
    
    for model_name, m_conf in models.items():
        name = model_name
        strat_type = m_conf['strategy']
        timeframes = m_conf['timeframes']
        tb = m_conf.get('triple_barrier', {})
        assets_filter = m_conf.get('assets_filter', assets)
        
        print(f"\n[*] Testing Model: {name} ({strat_type})")
        
        # Instantiate Strategy
        if strat_type == "WizardWave": strat = WizardWaveStrategy()
        elif strat_type == "WizardWaveProStrategy": strat = WizardWaveProStrategy(**m_conf.get('params', {}))
        elif strat_type == "WizardScalp": strat = WizardScalpStrategy()
        elif strat_type == "CLSRangeStrategy": strat = CLSRangeStrategy()
        elif strat_type == "IchimokuStrategy": strat = IchimokuStrategy(**m_conf.get('params', {}))
        elif strat_type == "IchimokuProStrategy": strat = IchimokuProStrategy(**m_conf.get('params', {}))
        else: continue
            
        for tf in timeframes:
            for symbol in assets_filter:
                try:
                    asset_type = 'crypto' if 'USDT' in symbol else 'tradfi'
                    
                    # Load Data
                    if strat_type == "CLSRangeStrategy":
                        # CLS needs HTF/LTF
                        htf_df = load_data(symbol, '1d', asset_type)
                        ltf_df = load_data(symbol, tf, asset_type)
                        if htf_df.empty or ltf_df.empty: continue
                        df = strat.apply_mtf(htf_df, ltf_df)
                    else:
                        df = load_data(symbol, tf, asset_type)
                        if df.empty: continue
                        if hasattr(strat, 'apply_strategy'): df = strat.apply_strategy(df)
                        else: df = strat.apply(df)
                        
                    if 'signal_type' not in df.columns: continue
                    
                    # TP/SL from Config
                    tp = tb.get(f"{asset_type}_pt", 0.02)
                    sl = tb.get(f"{asset_type}_sl", 0.01)
                    limit = tb.get("time_limit_bars", 24)
                    
                    trades = run_simulation(df, 'signal_type', tp, sl, limit)
                    
                    if trades:
                        res = {
                            "Model": name,
                            "Asset": symbol,
                            "TF": tf,
                            "Trades": len(trades),
                            "WR": (sum(1 for t in trades if t > 0) / len(trades)) * 100,
                            "Return": sum(trades) * 100
                        }
                        results.append(res)
                        print(f"  > {symbol:<12} {tf:<4} | Trades: {len(trades):<4} WR: {res['WR']:>6.1f}% | Ret: {res['Return']:>7.1f}%")
                except Exception as e:
                    # print(f"Error {symbol}: {e}")
                    pass

    if not results:
        print("\n[!] No trades generated by any model.")
        return

    df_final = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print(f"{'SUMMARY BY MODEL':<30} {'TRADES':<10} {'AVG WR':<10} {'TOT RET':<10}")
    print("-" * 70)
    summary = df_final.groupby('Model').agg({'Trades': 'sum', 'WR': 'mean', 'Return': 'sum'})
    for model, row in summary.sort_values(by='Return', ascending=False).iterrows():
        print(f"{model:<30} {int(row['Trades']):<10} {row['WR']:>7.1f}%    {row['Return']:>8.1f}%")
    print("="*70)
    
    print(f"\n[DONE] Tested {len(results)} Asset/TF Pairs.")

if __name__ == "__main__":
    main()

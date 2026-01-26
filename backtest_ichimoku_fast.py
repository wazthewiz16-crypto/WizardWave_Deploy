import pandas as pd
import numpy as np
import os
import sys

# Import Data Loader
loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agent/skills/market-data-manager/scripts"))
sys.path.append(loader_path)
from loader import load_data

# Import Strategy
from strategy_ichimoku import IchimokuStrategy

# Configuration (from report_ichimoku_90d.py)
TRADFI_CONFIG = {"tenkan": 20, "kijun": 60, "span_b": 120, "disp": 30}
CRYPTO_CONFIG = {"tenkan": 7, "kijun": 21, "span_b": 42, "disp": 21}
TIMEFRAMES = ["4h", "1d"] # Focus on most common

def run_asset_backtest(symbol, timeframe):
    asset_type = 'crypto' if 'USDT' in symbol else 'tradfi'
    df = load_data(symbol, timeframe, asset_type)
    
    if df.empty or len(df) < 200:
        return []
    
    cfg = CRYPTO_CONFIG if asset_type == 'crypto' else TRADFI_CONFIG
    ichi = IchimokuStrategy(
        tenkan=cfg['tenkan'],
        kijun=cfg['kijun'],
        span_b=cfg['span_b'],
        displacement=cfg['disp']
    )
    
    df = ichi.apply_strategy(df)
    signals = df[df['signal_type'].notna()].copy()
    
    trades = []
    for entry_time, row in signals.iterrows():
        signal_type = row['signal_type']
        entry_price = row['close']
        
        # Simulation
        future_df = df.loc[entry_time:].iloc[1:]
        exit_price = entry_price
        exit_time = future_df.index[-1] if not future_df.empty else entry_time
        exit_reason = "Open"
        
        for t, f_row in future_df.iterrows():
            if signal_type == "LONG" and f_row['close'] < f_row['kijun']:
                exit_price = f_row['close']
                exit_time = t
                exit_reason = "Kijun Exit"
                break
            elif signal_type == "SHORT" and f_row['close'] > f_row['kijun']:
                exit_price = f_row['close']
                exit_time = t
                exit_reason = "Kijun Exit"
                break
                
        pnl = (exit_price - entry_price) / entry_price if signal_type == "LONG" else (entry_price - exit_price) / entry_price
        pnl -= 0.001 # 0.1% slippage/fees
        
        trades.append({
            "Asset": symbol,
            "Timeframe": timeframe,
            "Signal": signal_type,
            "PnL": pnl,
            "EntryTime": entry_time
        })
    return trades

if __name__ == "__main__":
    assets = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 
        '^GSPC', '^NDX', 'GC=F', 'SI=F', 'EURUSD=X', 'GBPUSD=X'
    ]
    
    print(f"[*] Starting Fast Ichimoku Backtest on {len(assets)} assets...")
    
    all_trades = []
    for asset in assets:
        for tf in TIMEFRAMES:
            try:
                trades = run_asset_backtest(asset, tf)
                all_trades.extend(trades)
            except Exception as e:
                print(f"[!] Error on {asset} {tf}: {e}")
                
    if not all_trades:
        print("No trades found.")
        sys.exit()
        
    df_results = pd.DataFrame(all_trades)
    
    # Summary
    print("\n" + "="*60)
    print(f"{'ASSET':<12} {'TF':<5} {'TRADES':<8} {'WIN RATE':<10} {'AVG PnL':<10} {'SUM PnL':<10}")
    print("-" * 60)
    
    summary = df_results.groupby(['Asset', 'Timeframe']).agg({
        'PnL': ['count', 'mean', 'sum', lambda x: (x > 0).mean()]
    })
    summary.columns = ['Count', 'Avg', 'Sum', 'WinRate']
    
    for (asset, tf), row in summary.iterrows():
        print(f"{asset:<12} {tf:<5} {int(row['Count']):<8} {row['WinRate']:>7.2%}    {row['Avg']:>8.2%}    {row['Sum']:>8.2%}")
        
    print("-" * 60)
    print(f"TOTAL TRADES: {len(df_results)}")
    print(f"OVERALL WIN RATE: {(df_results['PnL'] > 0).mean():.2%}")
    print(f"OVERALL AVG PNL: {df_results['PnL'].mean():.2%}")
    print(f"OVERALL SUM PNL: {df_results['PnL'].sum():.2%}")
    print("="*60)

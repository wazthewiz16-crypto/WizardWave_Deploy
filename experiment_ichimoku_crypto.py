import pandas as pd
import concurrent.futures
from data_fetcher import fetch_data
from strategy_ichimoku import IchimokuStrategy

# Only Crypto Assets for Optimization
CRYPTO_ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "DOGE/USDT"]
TEST_TIMEFRAMES = ["4h", "1d"]

CONFIGS = [
    {"name": "Slow (Baseline)", "tenkan": 20, "kijun": 60, "span_b": 120, "disp": 30},
    {"name": "Standard (Fast)", "tenkan": 9, "kijun": 26, "span_b": 52, "disp": 26},
    {"name": "Crypto Optimized (Medium)", "tenkan": 10, "kijun": 30, "span_b": 60, "disp": 30},
    {"name": "Crypto Turbo (Very Fast)", "tenkan": 7, "kijun": 21, "span_b": 42, "disp": 21}, # 3 weeks logic
]

def run_backtest_config(config):
    ichi = IchimokuStrategy(
        tenkan=config['tenkan'],
        kijun=config['kijun'],
        span_b=config['span_b'],
        displacement=config['disp']
    )
    
    trades = []
    
    for asset in CRYPTO_ASSETS:
        for tf in TEST_TIMEFRAMES:
            try:
                # Fetch Data (2000 bars)
                df = fetch_data(asset, "crypto", tf, limit=2000)
                if df is None or df.empty or len(df) < 150: continue
                
                # Apply
                df = ichi.apply_strategy(df, tf)
                signals = df[df['signal_type'].notna()].copy()
                
                if signals.empty: continue
                
                # Simulation
                for entry_time, row in signals.iterrows():
                    signal_type = row['signal_type']
                    entry_price = row['close']
                    
                    # Exit (Kijun Cross)
                    future_df = df.loc[entry_time:].iloc[1:]
                    exit_price = entry_price
                    exit_time = future_df.index[-1] if not future_df.empty else entry_time
                    
                    for t, f_row in future_df.iterrows():
                        if signal_type == "LONG" and f_row['exit_long']:
                            exit_price = f_row['close']
                            exit_time = t
                            break
                        elif signal_type == "SHORT" and f_row['exit_short']:
                            exit_price = f_row['close']
                            exit_time = t
                            break
                    
                    # PnL
                    if signal_type == "LONG": pnl = (exit_price - entry_price) / entry_price
                    else: pnl = (entry_price - exit_price) / entry_price
                    
                    pnl -= 0.001 # Fees
                    
                    trades.append({
                        "Config": config['name'],
                        "Asset": asset,
                        "Timeframe": tf,
                        "PnL": pnl
                    })
                    
            except Exception as e:
                pass
                
    return trades

def main():
    print("Running Crypto Ichimoku Optimization...")
    
    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_backtest_config, cfg): cfg for cfg in CONFIGS}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            all_results.extend(res)
            
    if not all_results:
        print("No trades generated.")
        return

    df = pd.DataFrame(all_results)
    
    print("\n" + "="*50)
    print("CRYPTO ICHIMOKU OPTIMIZATION RESULTS")
    print("="*50)
    
    # Analyze by Config
    stats = df.groupby('Config')['PnL'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()])
    stats.columns = ['Trades', 'Avg PnL', 'Total Return', 'Win Rate']
    stats['Win Rate'] = stats['Win Rate'].apply(lambda x: f"{x:.2%}")
    stats['Avg PnL'] = stats['Avg PnL'].apply(lambda x: f"{x:.2%}")
    stats['Total Return'] = stats['Total Return'].apply(lambda x: f"{x:.2%}") # Simple sum of PnL % (Compounding approx)
    
    print(stats.sort_values(by='Trades', ascending=False))
    
    # Best Per Timeframe
    print("\n--- Breakdown by Timeframe ---")
    tf_stats = df.groupby(['Config', 'Timeframe'])['PnL'].agg(['count', 'mean'])
    print(tf_stats)

    df.to_csv("crypto_ichi_opt.csv", index=False)

if __name__ == "__main__":
    main()

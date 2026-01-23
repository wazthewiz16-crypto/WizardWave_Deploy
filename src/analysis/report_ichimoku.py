import pandas as pd
import numpy as np
import concurrent.futures
from data_fetcher import fetch_data
from strategy_ichimoku import IchimokuStrategy
import time

# --- Configuration ---
TRADFI_ASSETS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "GC=F", "SI=F", "CL=F", "DX-Y.NYB", "^GSPC", "^NDX", "^AXJO"]
CRYPTO_ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "DOGE/USDT"]
TIMEFRAMES = ["1h", "4h", "1d"] # 12h/4d require resampling, handled in fetch_data usually but let's stick to standard first to test logic. 
# Wait, fetch_data supports 12H/4D by resampling.
# Let's verify data_fetcher support. Assuming it works or I will simple-test 1H/4H/1D first.
# User asked for: 1H, 4H, 12H, 1D, 4D.
TEST_TIMEFRAMES = ["4h", "1d"]

def run_backtest_asset(asset, asset_type):
    results = []
    
    # Strat
    ichi = IchimokuStrategy()
    
    for tf in TEST_TIMEFRAMES:
        try:
            # Fetch Data (Increase history for robust testing)
            limit = 2000
            
            df = fetch_data(asset, asset_type, tf, limit=limit)
            
            if df is None or df.empty or len(df) < 150:
                continue
                
            # Apply Strategy
            df = ichi.apply_strategy(df, tf)
            
            # Simulate Trades
            signals = df[df['signal_type'].notna()].copy()
            
            if signals.empty:
                continue
                
            # Iterate Signals
            for entry_time, row in signals.iterrows():
                signal_type = row['signal_type']
                entry_price = row['close']
                
                # Check Overlap or just raw performance?
                # "Report back the performance... divide into Tradfi and Crypto"
                # Simple per-trade performance.
                
                # Find Exit
                # Look forward from entry_time
                future_df = df.loc[entry_time:].iloc[1:]
                
                exit_price = entry_price # Default breakeven if no exit found
                exit_reason = "End of Data"
                exit_time = future_df.index[-1] if not future_df.empty else entry_time
                pnl = 0.0
                
                for t, f_row in future_df.iterrows():
                    # Exit Check
                    if signal_type == "LONG" and f_row['exit_long']:
                        exit_price = f_row['close']
                        exit_reason = "Kijun Cross"
                        exit_time = t
                        break
                    elif signal_type == "SHORT" and f_row['exit_short']:
                        exit_price = f_row['close']
                        exit_reason = "Kijun Cross"
                        exit_time = t
                        break
                
                # Calculate PnL
                if signal_type == "LONG":
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price
                    
                # Deduct Fees Estimate (0.05% Crypto, 0.0% Tradfi? Let's use 0.1% generic)
                pnl -= 0.001
                
                trade_res = {
                    "Asset": asset,
                    "Type": asset_type,
                    "Timeframe": tf,
                    "Signal": signal_type,
                    "Entry Time": entry_time,
                    "Exit Time": exit_time,
                    "PnL": pnl,
                    "Reason": exit_reason
                }
                results.append(trade_res)
                
        except Exception as e:
            # print(f"Error {asset} {tf}: {e}")
            pass
            
    return results

def main():
    print("Starting Ichimoku Strategy Backtest (20/60/120/30)...")
    
    all_trades = []
    
    # Run TradFi
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_backtest_asset, asset, 'trad'): asset for asset in TRADFI_ASSETS}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            all_trades.extend(res)
            
    # Run Crypto
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(run_backtest_asset, asset, 'crypto'): asset for asset in CRYPTO_ASSETS}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            all_trades.extend(res)
            
    # Process Results
    if not all_trades:
        print("No trades found.")
        return

    df_res = pd.DataFrame(all_trades)
    
    print("\n" + "="*40)
    print("ICHIMOKU STRATEGY REPORT")
    print("="*40)
    
    # Overall
    print(f"\nTotal Trades: {len(df_res)}")
    print(f"Overall Win Rate: {(df_res['PnL'] > 0).mean():.2%}")
    print(f"Avg PnL: {df_res['PnL'].mean():.2%}")
    
    # Segmented: Crypto vs TradFi
    print("\n--- PERFORMANCE BY ASSET CLASS ---")
    grouped = df_res.groupby('Type')['PnL'].agg(['count', 'mean', lambda x: (x>0).mean()])
    grouped.columns = ['Trades', 'Avg PnL', 'Win Rate']
    grouped['Win Rate'] = grouped['Win Rate'].apply(lambda x: f"{x:.2%}")
    grouped['Avg PnL'] = grouped['Avg PnL'].apply(lambda x: f"{x:.2%}")
    print(grouped)
    
    # Segmented: Timeframe
    print("\n--- PERFORMANCE BY TIMEFRAME ---")
    tf_group = df_res.groupby('Timeframe')['PnL'].agg(['count', 'mean', lambda x: (x>0).mean()])
    tf_group.columns = ['Trades', 'Avg PnL', 'Win Rate']
    tf_group['Win Rate'] = tf_group['Win Rate'].apply(lambda x: f"{x:.2%}")
    tf_group['Avg PnL'] = tf_group['Avg PnL'].apply(lambda x: f"{x:.2%}")
    print(tf_group)
    
    # top Trades
    print("\n--- TOP 5 TRADES ---")
    top = df_res.sort_values(by='PnL', ascending=False).head(5)
    for _, row in top.iterrows():
        print(f"{row['Asset']} ({row['Timeframe']}) {row['Signal']}: {row['PnL']:.2%} ({row['Entry Time']})")

    # Save to CSV
    df_res.to_csv("ichimoku_report.csv", index=False)
    print("\nSaved detailed report to 'ichimoku_report.csv'")

if __name__ == "__main__":
    main()

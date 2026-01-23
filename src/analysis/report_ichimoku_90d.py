import pandas as pd
import concurrent.futures
from data_fetcher import fetch_data
from strategy_ichimoku import IchimokuStrategy
import datetime

# --- CONFIGURATION (Hybrid) ---
TRADFI_ASSETS = ["EURUSD=X", "GBPUSD=X", "AUDUSD=X", "GC=F", "SI=F", "CL=F", "DX-Y.NYB", "^GSPC", "^NDX", "^AXJO"]
CRYPTO_ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "DOGE/USDT"]

# Hybrid Settings
TRADFI_CONFIG = {"tenkan": 20, "kijun": 60, "span_b": 120, "disp": 30} # Slow
CRYPTO_CONFIG = {"tenkan": 7, "kijun": 21, "span_b": 42, "disp": 21}   # Turbo

# 90 Day Limit logic
# We need to fetch enough data for the Ichimoku Calculation (approx 150 bars back from 90 days ago)
# then filter results for signals starting > Now - 90 Days.
DAYS_BACK = 180
TIMEFRAMES = ["4h", "12h", "1d"]

def run_backtest_asset(asset, asset_type):
    trades = []
    
    # Configure Strategy
    if asset_type == 'crypto':
        cfg = CRYPTO_CONFIG
    else:
        cfg = TRADFI_CONFIG
        
    ichi = IchimokuStrategy(
        tenkan=cfg['tenkan'],
        kijun=cfg['kijun'],
        span_b=cfg['span_b'],
        displacement=cfg['disp']
    )
    
    cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=DAYS_BACK)
    
    for tf in TIMEFRAMES:
        try:
            # Fetch Data (safe 1000 bars)
            # yfinance returns timezone-unaware often, assuming UTC or local NY.
            # handled by fetch_data, usually returns tz-aware or tz-naive.
            df = fetch_data(asset, asset_type, tf, limit=1000)
            
            if df is None or df.empty or len(df) < 150: continue
            
            # Apply
            df = ichi.apply_strategy(df, tf)
            signals = df[df['signal_type'].notna()].copy()
            
            if signals.empty: continue
            
            # Filter for last 90 days
            # Handle TZ: Convert cutoff to df index tz if present
            if df.index.tz is not None:
                cutoff_date = cutoff_date.tz_convert(df.index.tz)
            else:
                cutoff_date = cutoff_date.tz_localize(None) # Make naive
            
            signals = signals[signals.index >= cutoff_date]
            
            if signals.empty: continue

            # Simulation
            for entry_time, row in signals.iterrows():
                signal_type = row['signal_type']
                entry_price = row['close']
                
                # Exit (Kijun Cross)
                future_df = df.loc[entry_time:].iloc[1:]
                exit_price = entry_price
                exit_time = future_df.index[-1] if not future_df.empty else entry_time
                exit_reason = "Open"
                
                # Kijun Trail Simulation
                for t, f_row in future_df.iterrows():
                    # Dynamic Kijun Exit
                    if signal_type == "LONG" and f_row['close'] < f_row['kijun']:
                        exit_price = f_row['close']
                        exit_time = t
                        exit_reason = "Kijun SL"
                        break
                    elif signal_type == "SHORT" and f_row['close'] > f_row['kijun']:
                        exit_price = f_row['close']
                        exit_time = t
                        exit_reason = "Kijun SL"
                        break
                
                # Calc PnL
                if signal_type == "LONG": pnl = (exit_price - entry_price) / entry_price
                else: pnl = (entry_price - exit_price) / entry_price
                
                pnl -= 0.001 # Fees
                
                trades.append({
                    "Asset": asset,
                    "Type": asset_type,
                    "Timeframe": tf,
                    "Signal": signal_type,
                    "Entry Time": entry_time,
                    "PnL": pnl,
                    "Exit Reason": exit_reason
                })
                
        except Exception as e:
            pass
            
    return trades

def main():
    print(f"Running Hybrid Ichimoku Backtest (Last {DAYS_BACK} Days)...")
    print(f"TradFi: {TRADFI_CONFIG}")
    print(f"Crypto: {CRYPTO_CONFIG}")
    
    all_trades = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit TradFi
        f1 = {executor.submit(run_backtest_asset, a, 'trad'): a for a in TRADFI_ASSETS}
        # Submit Crypto
        f2 = {executor.submit(run_backtest_asset, a, 'crypto'): a for a in CRYPTO_ASSETS}
        
        for future in concurrent.futures.as_completed({**f1, **f2}):
            res = future.result()
            all_trades.extend(res)
            
    if not all_trades:
        print("No trades found in last 90 days.")
        return

    df = pd.DataFrame(all_trades)
    
    print("\n" + "="*50)
    print(f"ICHIMOKU 90-DAY PERFORMANCE REPORT")
    print("="*50)
    
    # Overall Stats
    print(f"Total Trades: {len(df)}")
    print(f"Avg PnL: {df['PnL'].mean():.2%}")
    print(f"Win Rate: {(df['PnL'] > 0).mean():.2%}")
    print(f"Total Return (Sum): {df['PnL'].sum():.2%}")
    
    # By Asset Class
    print("\n--- By Asset Class ---")
    grp = df.groupby('Type')['PnL'].agg(['count', 'mean', 'sum', lambda x: (x>0).mean()])
    grp.columns = ['Trades', 'Avg PnL', 'Total Return', 'Win Rate']
    grp['Win Rate'] = grp['Win Rate'].apply(lambda x: f"{x:.2%}")
    grp['Avg PnL'] = grp['Avg PnL'].apply(lambda x: f"{x:.2%}")
    grp['Total Return'] = grp['Total Return'].apply(lambda x: f"{x:.2%}")
    print(grp)
    
    # Top Trades
    print("\n--- Top 5 Winners ---")
    top = df.sort_values(by='PnL', ascending=False).head(5)
    for _, row in top.iterrows():
        print(f"{row['Asset']} {row['Timeframe']} {row['Signal']}: {row['PnL']:.2%} ({row['Entry Time']})")

    df.to_csv("ichimoku_90d.csv", index=False)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import json
import os
import sys
# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.core.data_fetcher import fetch_data
from src.strategies.strategy_monday import MondayRangeStrategy
from src.core.pipeline import get_asset_type
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

def run_report():
    print("Generating Monday Range Strategy Report (Last 30 Days)...")
    
    # Load Config
    with open(os.path.join(root_path, 'config', 'strategy_config.json'), 'r') as f:
        config = json.load(f)
        
    assets = config['assets']
    results = []
    
    strat = MondayRangeStrategy()
    
    # Timeframe: 1h for sufficient resolution
    tf = '1h'
    # Timeframe: 1h for sufficient resolution
    tf = '1h'
    limit = 24 * 75 # 75 Days buffer (for 60 days report)
    
    crypto_whitelist = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'LINK/USDT']
    
    for symbol in assets:
        try:
            # Asset Type
            asset_type = get_asset_type(symbol)
            fetch_type = 'trad' if asset_type == 'forex' or asset_type == 'trad' else 'crypto'
            if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
            
            # Filter Crypto Assets
            if fetch_type == 'crypto' and symbol not in crypto_whitelist:
                # print(f"Skipping {symbol} (Not in whitelist)")
                continue

            print(f"Processing {symbol}...")
            df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=limit)
            
            if df.empty:
                print(f"No data for {symbol}")
                continue
                
            # Filter Last 60 Days strictly
            cutoff = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=60)
            # We need slightly more history for Monday calc, so we apply first then filter signals
            
            # Apply Strategy
            df = strat.apply(df)
            
            # Extract Signals
            # Signals are where signal_type contains MONDAY
            sig_mask = df['signal_type'].str.contains('MONDAY')
            signals = df[sig_mask].copy()
            
            # Filter solely for signals occurring in last 30 days
            signals = signals[signals.index >= cutoff]
            
            for time, row in signals.iterrows():
                entry_price = row['close']
                s_type = row['signal_type']
                target = row['target_price']
                target_mid = row['target_mid']
                sl_price = row['stop_loss']
                
                # Look at future data
                future = df.loc[df.index > time]
                
                if pd.isna(sl_price):
                    outcome = "INVALID/NO SL"
                    pnl_pct = 0.0
                else:
                    hit_tp1 = False
                    hit_tp2 = False
                    hit_sl = False
                    
                    if "LONG" in s_type:
                        # 1. Check for TP1 vs Initial SL
                        tp1_mask = future['high'] >= target_mid
                        sl_mask = future['low'] <= sl_price
                        
                        idx_tp1 = future[tp1_mask].index.min()
                        idx_sl = future[sl_mask].index.min()
                        
                        ts_tp1 = idx_tp1 if pd.notna(idx_tp1) else pd.Timestamp.max.tz_localize(idx_tp1.tz if hasattr(idx_tp1, 'tz') else None)
                        ts_sl = idx_sl if pd.notna(idx_sl) else pd.Timestamp.max.tz_localize(idx_sl.tz if hasattr(idx_sl, 'tz') else None)
                        
                        if pd.isna(idx_tp1) and pd.isna(idx_sl):
                            outcome = "OPEN"
                            curr = future.iloc[-1]['close']
                            pnl_pct = (curr - entry_price) / entry_price
                        elif ts_tp1 < ts_sl:
                            # Hit TP1 First!
                            pnl_locked = (target_mid - entry_price) / entry_price
                            
                            # Remaining 50%: Break Even Stop (Entry Price) -> Target 2
                            future_post_tp1 = future.loc[future.index > idx_tp1]
                            
                            if future_post_tp1.empty:
                                outcome = "HIT TP1 -> OPEN"
                                pnl_pct = (0.50 * pnl_locked) + (0.50 * pnl_locked)
                            else:
                                tp2_mask = future_post_tp1['high'] >= target
                                be_mask = future_post_tp1['low'] <= entry_price
                                
                                idx_tp2 = future_post_tp1[tp2_mask].index.min()
                                idx_be = future_post_tp1[be_mask].index.min()
                                
                                ts_tp2 = idx_tp2 if pd.notna(idx_tp2) else pd.Timestamp.max.tz_localize(None)
                                ts_be = idx_be if pd.notna(idx_be) else pd.Timestamp.max.tz_localize(None)
                                
                                if pd.isna(idx_tp2) and pd.isna(idx_be):
                                    outcome = "HIT TP1 -> OPEN"
                                    curr = future_post_tp1.iloc[-1]['close']
                                    rem_pnl = (curr - entry_price) / entry_price
                                elif ts_tp2 < ts_be:
                                    outcome = "HIT TP2 (Full)"
                                    rem_pnl = (target - entry_price) / entry_price
                                else:
                                    outcome = "HIT TP1 -> BE"
                                    rem_pnl = 0.0
                                
                                pnl_pct = (0.50 * pnl_locked) + (0.50 * rem_pnl)
                        else:
                            outcome = "HIT SL"
                            pnl_pct = (sl_price - entry_price) / entry_price

                    else: # SHORT
                        # 1. Check for TP1 vs Initial SL
                        tp1_mask = future['low'] <= target_mid
                        sl_mask = future['high'] >= sl_price
                        
                        idx_tp1 = future[tp1_mask].index.min()
                        idx_sl = future[sl_mask].index.min()
                        
                        ts_tp1 = idx_tp1 if pd.notna(idx_tp1) else pd.Timestamp.max.tz_localize(idx_tp1.tz if hasattr(idx_tp1, 'tz') else None)
                        ts_sl = idx_sl if pd.notna(idx_sl) else pd.Timestamp.max.tz_localize(idx_sl.tz if hasattr(idx_sl, 'tz') else None)
                        
                        if pd.isna(idx_tp1) and pd.isna(idx_sl):
                            outcome = "OPEN"
                            curr = future.iloc[-1]['close']
                            pnl_pct = (entry_price - curr) / entry_price
                        elif ts_tp1 < ts_sl:
                            # Hit TP1 First!
                            pnl_locked = (entry_price - target_mid) / entry_price
                            
                            # Remainder
                            future_post_tp1 = future.loc[future.index > idx_tp1]
                            
                            if future_post_tp1.empty:
                                outcome = "HIT TP1 -> OPEN"
                                pnl_pct = (0.50 * pnl_locked) + (0.50 * pnl_locked)
                            else:
                                tp2_mask = future_post_tp1['low'] <= target
                                be_mask = future_post_tp1['high'] >= entry_price
                                
                                idx_tp2 = future_post_tp1[tp2_mask].index.min()
                                idx_be = future_post_tp1[be_mask].index.min()
                                
                                ts_tp2 = idx_tp2 if pd.notna(idx_tp2) else pd.Timestamp.max.tz_localize(None)
                                ts_be = idx_be if pd.notna(idx_be) else pd.Timestamp.max.tz_localize(None)
                                
                                if pd.isna(idx_tp2) and pd.isna(idx_be):
                                    outcome = "HIT TP1 -> OPEN"
                                    curr = future_post_tp1.iloc[-1]['close']
                                    rem_pnl = (entry_price - curr) / entry_price
                                elif ts_tp2 < ts_be:
                                    outcome = "HIT TP2 (Full)"
                                    rem_pnl = (entry_price - target) / entry_price
                                else:
                                    outcome = "HIT TP1 -> BE"
                                    rem_pnl = 0.0
                                
                                pnl_pct = (0.50 * pnl_locked) + (0.50 * rem_pnl)
                        else:
                            outcome = "HIT SL"
                            pnl_pct = (entry_price - sl_price) / entry_price

                results.append({
                    "Asset": symbol,
                    "Class": "Crypto" if asset_type == 'crypto' else "TradFi",
                    "Date": str(time),
                    "Type": s_type,
                    "Entry": entry_price,
                    "Target": target,
                    "Result": outcome,
                    "PnL %": round(pnl_pct * 100, 2)
                })
                
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Results to DF
    if results:
        res_df = pd.DataFrame(results)
        print("\n=== MONDAY RANGE STRATEGY REPORT (LAST 60 DAYS) ===")
        # Reorder cols
        cols = ['Asset', 'Class', 'Date', 'Type', 'Entry', 'Target', 'Result', 'PnL %']
        print(res_df[cols].to_string(index=False))
        
        # Function to print stats
        def print_stats(name, df_sub):
            if df_sub.empty:
                print(f"\n[{name}] No setups found.")
                return
            
            total = len(df_sub)
            total = len(df_sub)
            wins = len(df_sub[df_sub['PnL %'] > 0])
            losses = len(df_sub[df_sub['PnL %'] < 0])
            be = len(df_sub[df_sub['PnL %'] == 0])
            opens = len(df_sub[df_sub['Result'].str.contains("OPEN")])
            
            # Win rate excluding opens? Or just Wins / Total?
            # Usually Wins / (Wins + Losses)
            denom = wins + losses + be
            win_rate = (wins / denom) * 100 if denom > 0 else 0
            avg_pnl = df_sub['PnL %'].mean()
            
            print(f"\n--- {name} PERFORMANCE ---")
            print(f"Total Signals: {total}")
            print(f"Wins: {wins}")
            print(f"Losses: {losses}")
            print(f"Breakeven: {be}")
            print(f"Open: {opens}")
            print(f"Win Rate (Closed): {win_rate:.2f}%")
            print(f"Avg PnL per Trade: {avg_pnl:.2f}%")

        # Split
        df_crypto = res_df[res_df['Class'] == 'Crypto']
        df_trad = res_df[res_df['Class'] == 'TradFi']
        
        print("\n=== SUMMARY BY ASSET CLASS ===")
        print_stats("CRYPTO", df_crypto)
        print_stats("TRADFI/FOREX", df_trad)
        
        print("\n=== OVERALL SUMMARY ===")
        print_stats("OVERALL", res_df)
        
        res_df.to_csv("monday_range_report.csv", index=False)
        print("\nReport saved to monday_range_report.csv")
    else:
        print("No signals found in the last 30 days.")

if __name__ == "__main__":
    run_report()

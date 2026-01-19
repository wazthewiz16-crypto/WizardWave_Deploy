import pandas as pd
import numpy as np
import json
import os
import pandas_ta as ta
from data_fetcher import fetch_data
from strategy_cls import CLSRangeStrategy
from pipeline import get_asset_type
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

def run_report():
    print("Generating CLS Range Strategy Report (Comprehensive Matrix)...")
    
    # Load Config
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
        
    assets = config['assets']
    
    # Test Matrix
    # 1. Configurations
    configs = [
        {'name': 'A (Baseline: 2.0x/10)', 'atr': 2.0, 'swing': 10, 'vol': 1.0},
        {'name': 'B (Sensitive: 1.5x/10)', 'atr': 1.5, 'swing': 10, 'vol': 1.0},
        {'name': 'C (Winner: 1.5x/10/1.5v)', 'atr': 1.5, 'swing': 10, 'vol': 1.5}
    ]
    
    # 2. Timeframe Pairs (HTF Trigger -> LTF Entry)
    # limit_htf should be enough for swing lookback (e.g. 50-100 bars)
    # limit_ltf should cover the 90 day report window
    tf_pairs = [
        {'label': 'Daily -> 1H', 'htf': '1d', 'ltf': '1h', 'h_lim': 500, 'l_lim': 24*120},
        {'label': 'Weekly -> 4H', 'htf': '1w', 'ltf': '4h', 'h_lim': 100, 'l_lim': 6*120} 
    ]
    
    all_results = []
    crypto_whitelist = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'LINK/USDT']
    
    for symbol in assets:
        try:
            # Asset Type
            asset_type = get_asset_type(symbol)
            fetch_type = 'trad' if asset_type == 'forex' or asset_type == 'trad' else 'crypto'
            if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
            
            # Filter Crypto Assets
            if fetch_type == 'crypto' and symbol not in crypto_whitelist:
                continue

            print(f"Processing {symbol}...")
            
            # Optimization: Cache fetches? 
            # Since TFs change, we must fetch inside the TF loop.
            
            for tf_set in tf_pairs:
                label = tf_set['label']
                
                # Fetch Data
                df_htf = fetch_data(symbol, asset_type=fetch_type, timeframe=tf_set['htf'], limit=tf_set['h_lim'])
                df_ltf = fetch_data(symbol, asset_type=fetch_type, timeframe=tf_set['ltf'], limit=tf_set['l_lim'])
                
                if df_htf.empty or df_ltf.empty:
                     continue
                     
                # Filter Last 90 Days (Based on LTF)
                cutoff = pd.Timestamp.now(tz=df_ltf.index.tz) - pd.Timedelta(days=90)
                
                for cfg in configs:
                    strat = CLSRangeStrategy(swing_window=cfg['swing'], atr_multiplier=cfg['atr'], volume_multiplier=cfg['vol'])
                    
                    df = strat.apply_mtf(df_htf, df_ltf)
                    
                    if 'signal_type' not in df.columns:
                        continue
                        
                    sig_mask = df['signal_type'].str.contains('CLS')
                    signals = df[sig_mask].copy()
                    signals = signals[signals.index >= cutoff]
                    
                    signals = signals.sort_index()
                    
                    # Initialize last_exit using same tz
                    tz_info = signals.index.tz
                    last_exit_time = pd.Timestamp.min
                    if tz_info:
                        last_exit_time = last_exit_time.tz_localize(tz_info)

                    for time, row in signals.iterrows():
                        # 1. Overlap Check
                        if time <= last_exit_time:
                            continue

                        entry_price = row['close']
                        s_type = row['signal_type']
                        target = row['target_price']
                        target_mid = row['target_mid']
                        sl_price = row['stop_loss']
                        
                        future = df.loc[df.index > time]
                        
                        # PnL Calculation Logic (Standard)
                        pnl_pct = 0.0
                        outcome = "INVALID"
                        trade_exit_time = None
                        
                        if pd.notna(sl_price) and pd.notna(target):
                            if "LONG" in s_type:
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
                                    trade_exit_time = future.index.max()
                                elif ts_tp1 < ts_sl:
                                    pnl_locked = (target_mid - entry_price) / entry_price
                                    future_post = future.loc[future.index > idx_tp1]
                                    if future_post.empty: # HIT TP1 -> OPEN
                                       rem = (future.iloc[-1]['close'] - entry_price) / entry_price
                                       outcome = "HIT TP1 -> OPEN"
                                       pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                       trade_exit_time = future.index.max()
                                    else:
                                        tp2_mask = future_post['high'] >= target
                                        be_mask = future_post['low'] <= entry_price
                                        idx_tp2 = future_post[tp2_mask].index.min()
                                        idx_be = future_post[be_mask].index.min()
                                        ts_tp2 = idx_tp2 if pd.notna(idx_tp2) else pd.Timestamp.max.tz_localize(None)
                                        ts_be = idx_be if pd.notna(idx_be) else pd.Timestamp.max.tz_localize(None)
                                        
                                        if pd.isna(idx_tp2) and pd.isna(idx_be):
                                            rem = (future_post.iloc[-1]['close'] - entry_price) / entry_price
                                            pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                            trade_exit_time = future_post.index.max()
                                        elif ts_tp2 < ts_be:
                                            rem = (target - entry_price) / entry_price
                                            pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                            trade_exit_time = idx_tp2
                                        else:
                                            rem = 0.0 # BE
                                            pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                            trade_exit_time = idx_be
                                else:
                                    outcome = "HIT SL" 
                                    pnl_pct = (sl_price - entry_price) / entry_price
                                    trade_exit_time = idx_sl
                            else: # SHORT
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
                                    trade_exit_time = future.index.max()
                                elif ts_tp1 < ts_sl:
                                    pnl_locked = (entry_price - target_mid) / entry_price
                                    future_post = future.loc[future.index > idx_tp1]
                                    if future_post.empty:
                                       rem = (entry_price - future.iloc[-1]['close']) / entry_price
                                       outcome = "HIT TP1 -> OPEN"
                                       pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                       trade_exit_time = future.index.max()
                                    else:
                                        tp2_mask = future_post['low'] <= target
                                        be_mask = future_post['high'] >= entry_price
                                        idx_tp2 = future_post[tp2_mask].index.min()
                                        idx_be = future_post[be_mask].index.min()
                                        ts_tp2 = idx_tp2 if pd.notna(idx_tp2) else pd.Timestamp.max.tz_localize(None)
                                        ts_be = idx_be if pd.notna(idx_be) else pd.Timestamp.max.tz_localize(None)
                                        
                                        if pd.isna(idx_tp2) and pd.isna(idx_be):
                                            rem = (entry_price - future_post.iloc[-1]['close']) / entry_price
                                            pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                            trade_exit_time = future_post.index.max()
                                        elif ts_tp2 < ts_be:
                                            rem = (entry_price - target) / entry_price
                                            pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                            trade_exit_time = idx_tp2
                                        else:
                                            rem = 0.0
                                            pnl_pct = (0.5 * pnl_locked) + (0.5 * rem)
                                            trade_exit_time = idx_be
                                else:
                                    outcome = "HIT SL"
                                    pnl_pct = (entry_price - sl_price) / entry_price
                                    trade_exit_time = idx_sl

                        if trade_exit_time:
                            last_exit_time = trade_exit_time

                        all_results.append({
                            "Setup": label,
                            "Config": cfg['name'],
                            "Asset": symbol,
                            "Class": "Crypto" if fetch_type == 'crypto' else "TradFi",
                            "Date": str(time),
                            "Type": s_type,
                            "PnL %": round(pnl_pct * 100, 2)
                        })

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if all_results:
        df_all = pd.DataFrame(all_results)
        
        # Grouped Summary
        # Group by [Setup, Class, Config]
        summary = df_all.groupby(['Setup', 'Class', 'Config']).agg(
            Signals=('Asset', 'count'),
            WinRate=('PnL %', lambda x: (x > 0).mean() * 100),
            AvgPnL=('PnL %', 'mean')
        ).reset_index()
        
        print("\n=== CLS STRATEGY MATRIX (Last 90 Days) ===")
        print(summary.to_string(index=False))
        
        # Save
        df_all.to_csv("cls_matrix_report.csv", index=False)
        print("\nDetailed report saved to cls_matrix_report.csv")
        
    else:
        print("No results found.")

if __name__ == "__main__":
    run_report()

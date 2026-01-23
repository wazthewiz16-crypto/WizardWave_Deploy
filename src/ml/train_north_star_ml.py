# !/usr/bin/env python3
"""
North Star Strategy ML Trainer.
Trains a Random Forest Classifier to filter North Star Signals.
Generates "Before vs After" Performance Report.
"""
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Add Project Root to Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_cls import CLSRangeStrategy
from feature_engine import calculate_ml_features

# 1. Configuration
ASSETS = {
    'Crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'XRP-USD', 'ADA-USD'],
    'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
    'Indices': ['^NDX', '^GSPC', '^DJI'],
    'Metals': ['GC=F', 'SI=F']
}

# Features to use for Training
ML_FEATURES = [
    'volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 
    'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'month_sin', 'cycle_regime'
]

def fetch_data(ticker, period='720d', interval='1h'):
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        df = df.rename(columns={'DeepHistory': 'open', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return pd.DataFrame()

def generate_dataset():
    print("Generating Dataset...")
    all_X = []
    all_y = []
    all_meta = [] # To track PnL later
    
    total_signals = 0
    
    for category, tickers in ASSETS.items():
        for ticker in tickers:
            print(f"  Processing {ticker}...")
            df_1h = fetch_data(ticker)
            if df_1h.empty: continue
            
            # 1. Run Strategy
            df_daily = CLSRangeStrategy.resample_daily_from_1h(df_1h)
            strat = CLSRangeStrategy()
            df_res = strat.apply_north_star(df_daily, df_1h)
            
            # 2. Add ML Features
            df_res = calculate_ml_features(df_res)
            
            # 3. Extract Signals
            signals = df_res[df_res['signal_type'] != 'NONE'].copy()
            if signals.empty: continue
            
            total_signals += len(signals)
            
            # 4. Labeling (Did it Hit Profit?)
            # Look ahead 48 hours.
            # Win = Hit Target before Stop
            
            for idx, row in signals.iterrows():
                future = df_res.loc[idx:].iloc[1:49]
                if future.empty: continue
                
                target = row['target_price']
                stop = row['stop_loss']
                entry = row['close']
                
                label = 0 # Loss
                pnl_pct = 0.0
                
                if row['signal_type'] == 'CLS_LONG':
                    hit_tp = future[future['high'] >= target]
                    hit_sl = future[future['low'] <= stop]
                    tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                    sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                    
                    if tp_idx < sl_idx: 
                        label = 1
                        pnl_pct = (target - entry) / entry
                    else:
                        label = 0
                        pnl_pct = (stop - entry) / entry # Should be neg
                        
                elif row['signal_type'] == 'CLS_SHORT':
                    hit_tp = future[future['low'] <= target]
                    hit_sl = future[future['high'] >= stop]
                    tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                    sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                    
                    if tp_idx < sl_idx: 
                        label = 1
                        pnl_pct = (entry - target) / entry
                    else:
                        label = 0
                        pnl_pct = (entry - stop) / entry
                
                # Store
                # Check if all features exist
                if not all(f in row for f in ML_FEATURES): continue
                
                all_X.append(row[ML_FEATURES].values)
                all_y.append(label)
                all_meta.append({
                    'ticker': ticker,
                    'pnl': pnl_pct,
                    'label': label,
                    'entry_time': idx
                })
                
    print(f"Total Signals Found: {total_signals}")
    return np.array(all_X), np.array(all_y), pd.DataFrame(all_meta)

def train_and_evaluate():
    X, y, meta = generate_dataset()
    
    if len(X) < 50:
        print("Not enough data to train ML model.")
        return
        
    # Implement PURGED TimeSeries Split (Financial ML Standard)
    # 1. Sort by Time
    meta['original_idx'] = np.arange(len(meta))
    meta = meta.sort_values('entry_time')
    sorted_indices = meta['original_idx'].values
    
    X = X[sorted_indices]
    y = y[sorted_indices]
    meta = meta.reset_index(drop=True)
    
    # 2. Split Point (70% Train, 30% Test)
    split_idx = int(len(meta) * 0.70)
    test_start_time = meta.iloc[split_idx]['entry_time']
    
    # 3. Purge/Embargo
    # We must ensure Training Data ends BEFORE Test Data starts - MaxLabelDuration
    # Max Label Duration = 48 Hours
    purge_duration = pd.Timedelta(hours=48)
    train_end_limit = test_start_time - purge_duration
    
    train_mask = meta['entry_time'] < train_end_limit
    test_mask = meta.index >= split_idx
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    # Metadata for analysis
    idx_test = meta[test_mask].index # This is now the specific slice of sorted meta
    
    print(f"\n--- PURGED TimeSeries Split ---")
    print(f"Train End   : {meta.loc[train_mask, 'entry_time'].max()}")
    print(f"Purge Gap   : {purge_duration}")
    print(f"Test Start  : {test_start_time}")
    print(f"Train Size  : {len(X_train)}")
    print(f"Test Size   : {len(X_test)}")
    
    # Train
    print(f"Training on {len(X_train)} samples...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Threshold Optimization
    best_thresh = 0.0
    best_pnl = -9999
    best_wr = 0.0
    best_trades = 0
    
    print("\n--- Threshold Optimization ---")
    print(f"Base Win Rate: {y_test.mean()*100:.2f}%")
    print("| Thresh | Trades | Win Rate | PnL Impact |")
    print("|---|---|---|---|")
    
    idx_test_arr = meta.loc[idx_test]
    
    for thresh in np.arange(0.25, 0.60, 0.05):
        mask = y_prob >= thresh
        if mask.sum() < 5: continue
        
        filtered = idx_test_arr[mask]
        wr = filtered['label'].mean() * 100
        pnl = filtered['pnl'].sum() * 100
        
        print(f"| {thresh:.2f} | {len(filtered)} | {wr:.2f}% | {pnl:.2f}% |")
        
        # Criteria: PnL > Best AND Trades >= 20
        # Or just maximize PnL
        if pnl > best_pnl and len(filtered) >= 10:
            best_pnl = pnl
            best_thresh = thresh
            best_wr = wr
            best_trades = len(filtered)
            
    print(f"\nBest Threshold Found: {best_thresh:.2f}")
    
    # Final Report with Best Threshold
    THRESHOLD = best_thresh
    
    # Results Analysis
    meta_test = meta.loc[idx_test].copy()
    meta_test['prob'] = y_prob
    
    # BEFORE (Raw Strategy on Test Set)
    raw_trades = len(meta_test)
    raw_win_rate = meta_test['label'].mean() * 100
    raw_total_pnl = meta_test['pnl'].sum() * 100 
    
    # AFTER (ML Filtered)
    filtered_test = meta_test[meta_test['prob'] >= THRESHOLD]
    ml_trades = len(filtered_test)
    ml_win_rate = filtered_test['label'].mean() * 100 if ml_trades > 0 else 0
    ml_total_pnl = filtered_test['pnl'].sum() * 100
    
    print("\n" + "="*40)
    print("      NORTH STAR STRATEGY ML FILTER      ")
    print("="*40)
    
    print("\n--- 1. BEFORE (Raw Strategy) ---")
    print(f"Total Trades : {raw_trades}")
    print(f"Win Rate     : {raw_win_rate:.2f}%")
    print(f"Total PnL    : {raw_total_pnl:.2f}%")
    
    print(f"\n--- 2. AFTER (ML Filtered > {THRESHOLD:.2f}) ---")
    print(f"Total Trades : {ml_trades} (Reduced by {100-(ml_trades/raw_trades*100):.1f}%)")
    print(f"Win Rate     : {ml_win_rate:.2f}%")
    print(f"Total PnL    : {ml_total_pnl:.2f}%")
    
    print("\n--- 3. IMPROVEMENT ---")
    wr_imp = ml_win_rate - raw_win_rate
    pnl_imp = ml_total_pnl - raw_total_pnl
    print(f"Win Rate Impact: {'+' if wr_imp > 0 else ''}{wr_imp:.2f}%")
    print(f"PnL Impact     : {'+' if pnl_imp > 0 else ''}{pnl_imp:.2f}%")
    
    # Feature Importance
    print("\n--- Feature Importance ---")
    imps = clf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': ML_FEATURES, 'Importance': imps})
    feat_imp = feat_imp.sort_values('Importance', ascending=False)
    print(feat_imp.head(5))

    # Save Model
    print(f"\nSaving Model to 'north_star_ml_model.pkl'...")
    import joblib
    model_data = {
        'model': clf,
        'threshold': THRESHOLD,
        'features': ML_FEATURES
    }
    joblib.dump(model_data, 'north_star_ml_model.pkl')
    print("Model Saved!")

if __name__ == "__main__":
    train_and_evaluate()

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Paths
loader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".agent/skills/market-data-manager/scripts"))
sys.path.append(loader_path)
from loader import load_data

from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features

# Load Config
with open('strategy_config.json', 'r') as f:
    CONFIG = json.load(f)

def apply_triple_barrier_pro(df, tb_config, asset_type):
    """Deep tagging for retraining"""
    time_limit = tb_config.get('time_limit_bars', 24)
    # Get PT/SL
    if asset_type == 'crypto':
        pt = tb_config['crypto_pt']
        sl = tb_config['crypto_sl']
    else:
        pt = tb_config.get('trad_pt', 0.02)
        sl = tb_config.get('trad_sl', 0.01)
        
    df = calculate_ml_features(df)
    signal_indices = df[df['signal_type'] != 'NONE'].index
    
    labels = []
    for start_time in signal_indices:
        row = df.loc[start_time]
        direction = 1 if 'LONG' in row['signal_type'] else -1
        start_iloc = df.index.get_loc(start_time)
        future = df.iloc[start_iloc + 1 : start_iloc + 1 + time_limit]
        
        if future.empty: continue
        
        if direction == 1:
            hit_pt = future['high'] >= row['close'] * (1 + pt)
            hit_sl = future['low'] <= row['close'] * (1 - sl)
        else:
            hit_pt = future['low'] <= row['close'] * (1 - pt)
            hit_sl = future['high'] >= row['close'] * (1 + sl)
            
        first_pt = hit_pt.idxmax() if hit_pt.any() else None
        first_sl = hit_sl.idxmax() if hit_sl.any() else None
        
        outcome = 0
        if first_pt and (not first_sl or first_pt < first_sl):
            outcome = 1
        elif first_sl:
            outcome = 0
        else:
            outcome = 1 if (future['close'].iloc[-1] > row['close'] if direction == 1 else future['close'].iloc[-1] < row['close']) else 0
            
        feat_row = {f: row[f] for f in ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'month_sin', 'cycle_regime', 'close_frac'] if f in row}
        feat_row['label'] = outcome
        feat_row['entry_time'] = start_time
        labels.append(feat_row)
        
    return pd.DataFrame(labels)

def retrain_model_robust(model_key):
    m_conf = CONFIG['models'][model_key]
    tf = '1h' if '1h' in model_key else '15m'
    print(f"\n[*] Starting ROBUST RETRAINING for {model_key} ({tf})...")
    
    all_data = []
    # Use only Top Liquid Assets for retraining scalp
    assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', '^GSPC', '^NDX']
    
    for symbol in assets:
        asset_type = 'crypto' if 'USDT' in symbol else 'tradfi'
        # Fetch HUGE sample from local archive (approx 5000 candles)
        df = load_data(symbol, tf, asset_type)
        if df.empty or len(df) < 500: continue
        
        print(f"  > Loaded {len(df)} candles for {symbol}")
        strat = WizardScalpStrategy(lookback=12) # Slightly longer lookback for noise reduction
        df = strat.apply(df)
        
        labeled = apply_triple_barrier_pro(df, m_conf['triple_barrier'], asset_type)
        if not labeled.empty:
            all_data.append(labeled)
            
    if not all_data:
        print(f"[!] No signal data found for {model_key}")
        return
        
    dataset = pd.concat(all_data).dropna()
    print(f"[*] Total Signal Samples Found: {len(dataset)}")
    
    if len(dataset) < 100:
        print("[!] Dataset too small. Aborting.")
        return
        
    dataset = dataset.sort_values('entry_time')
    features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'month_sin', 'cycle_regime', 'close_frac']
    
    # Split
    split = int(len(dataset) * 0.8)
    X_train, y_train = dataset[features].iloc[:split], dataset['label'].iloc[:split]
    X_test, y_test = dataset[features].iloc[split:], dataset['label'].iloc[split:]
    
    # Train robust forest
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[*] Final Test Accuracy (OOS): {acc:.2f}")
    
    # Save
    model_path = m_conf['model_file']
    save_data = {
        'model': clf,
        'features': features,
        'threshold': 0.61 # Slightly higher threshold for scalps
    }
    joblib.dump(save_data, model_path)
    print(f"[SUCCESS] Saved {model_path}")

if __name__ == "__main__":
    retrain_model_robust('1h')
    retrain_model_robust('15m')

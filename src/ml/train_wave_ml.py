import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Add Project Root
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.strategies.strategy import WizardWaveStrategy
from src.core.feature_engine import calculate_ml_features
from src.utils.paths import get_model_path

def fetch_data(ticker, period='5y', interval='1d'):
    print(f"Fetching {ticker}...")
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty: return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']]

def generate_wave_dataset():
    # Diversified Portfolio for Swing training
    tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', # Crypto
        'GC=F', 'CL=F', # Commodities
        '^NDX', '^GSPC', # Indices
        'EURUSD=X', 'GBPUSD=X' # Forex
    ]
    all_X = []
    all_y = []
    all_meta = []
    
    strat = WizardWaveStrategy()
    # Load Macro Data
    dxy_df = fetch_data('DX-Y.NYB', period='2y', interval='1d')
    btc_df = fetch_data('BTC-USD', period='2y', interval='1d')
    
    for ticker in tickers:
        df = fetch_data(ticker)
        if df.empty or len(df) < 300: continue
        
        # 1. Apply Wave Strategy
        df = strat.apply(df)
        
        # 2. Calculate Features (Passing Macro)
        df = calculate_ml_features(df, macro_df=dxy_df, crypto_macro_df=btc_df)
        
        # 3. Triple Barrier Labeling
        # Swing Target: 9% Profit, 3% Stop, 21 bars (21 Days)
        tp_mult = 0.09
        sl_mult = 0.03
        horizon = 21
        
        # Adjust for Asset Class Volatility (Simplified)
        if '=X' in ticker or '^' in ticker: # Forex / Indices
            tp_mult = 0.04
            sl_mult = 0.015
        
        signals = df[df['signal_type'] != 'NONE'].copy()
        print(f"  {ticker}: Found {len(signals)} signals.")
        
        for idx, row in signals.iterrows():
            future = df.loc[idx:].iloc[1:horizon+1]
            if future.empty: continue
            
            entry = row['close']
            target = entry * (1 + tp_mult) if 'LONG' in row['signal_type'] else entry * (1 - tp_mult)
            stop = entry * (1 - sl_mult) if 'LONG' in row['signal_type'] else entry * (1 + sl_mult)
            
            label = 0
            pnl = 0
            
            if 'LONG' in row['signal_type']:
                hit_tp = future[future['high'] >= target]
                hit_sl = future[future['low'] <= stop]
                
                tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                
                if tp_idx < sl_idx: 
                    label = 1
                    pnl = tp_mult
                else: 
                    label = 0
                    # Check if hit SL or just time exit
                    if not hit_sl.empty: pnl = -sl_mult
                    else: pnl = (future['close'].iloc[-1] - entry) / entry
            else:
                hit_tp = future[future['low'] <= target]
                hit_sl = future[future['high'] >= stop]
                
                tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                
                if tp_idx < sl_idx: 
                    label = 1
                    pnl = tp_mult
                else: 
                    label = 0
                    if not hit_sl.empty: pnl = -sl_mult
                    else: pnl = (entry - future['close'].iloc[-1]) / entry
            
            # Features (Standard set including FML features)
            features = [
                'volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 
                'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 
                'month_sin', 'cycle_regime', 'close_frac',
                'dxy_ret', 'dxy_corr', 'dxy_dist',
                'btc_corr', 'btc_mom'
            ]
            if not all(f in row for f in features): continue
            
            all_X.append(row[features].values)
            all_y.append(label)
            all_meta.append({
                'ticker': ticker,
                'entry_time': idx,
                'pnl': pnl,
                'label': label
            })
            
    return np.array(all_X), np.array(all_y), pd.DataFrame(all_meta)

def train_wave_model():
    X, y, meta = generate_wave_dataset()
    if len(X) < 100:
        print("Not enough signals to train.")
        return

    # Sort by time for Purged Cross Validation
    meta = meta.sort_values('entry_time').reset_index(drop=True)
    split_idx = int(len(meta) * 0.8)
    
    # 5-Bar Purge Gap for Swing ( prevent look-ahead leakage)
    test_start = meta.iloc[split_idx]['entry_time']
    train_end_limit = test_start - pd.Timedelta(days=5)
    
    train_mask = meta['entry_time'] < train_end_limit
    test_mask = meta.index >= split_idx
    
    # Align X and y with masks using the sorted meta indices
    # We need to ensure X and y are mapped correctly if all_X was built in order of meta
    # Actually generate_dataset appends in loop, so we should zip them first
    
    # Re-aligning:
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTraining Wave Meta-Labeler (Train: {len(X_train)}, Test: {len(X_test)})")
    
    # Random Forest with Moderate Depth for Swing Stability
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Report
    print("\n[Evaluation on Test Set]")
    print(classification_report(y_test, (y_prob >= 0.5).astype(int)))
    
    # Threshold Optimization
    best_thresh = 0.5
    best_pnl = -1
    
    print("\n| Thresh | Win Rate | Trade Count |")
    for thresh in np.arange(0.35, 0.65, 0.05):
        mask = y_prob >= thresh
        if mask.sum() < 5: continue
        
        filtered_meta = meta[test_mask][mask]
        wr = filtered_meta['label'].mean()
        print(f"| {thresh:.2f} | {wr:.2f} | {mask.sum()} |")
        
        if wr > 0.60: # Target > 60% win rate for filtered signals
             best_thresh = thresh
             
    # Save
    model_data = {
        'model': clf,
        'threshold': best_thresh,
        'features': [
            'volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 
            'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 
            'month_sin', 'cycle_regime', 'close_frac',
            'dxy_ret', 'dxy_corr', 'dxy_dist',
            'btc_corr', 'btc_mom'
        ],
        'strategy': 'WizardWave',
        'fml_version': '1.0 (Purged+FracDiff)'
    }
    save_path = get_model_path('wizard_wave_ml_model.pkl')
    joblib.dump(model_data, save_path)
    print(f"\n[SUCCESS] Saved New Wave Model: {save_path}")
    print(f"Optimized Confidence Threshold: {best_thresh:.2f}")

if __name__ == "__main__":
    train_wave_model()

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Add Project Root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features

def fetch_data(ticker, period='2y', interval='1h'):
    print(f"Fetching {ticker}...")
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty: return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']]

def generate_scalp_dataset():
    tickers = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD', 
        'LINK-USD', 'BNB-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD',
        'EURUSD=X', 'GBPUSD=X', '^NDX', '^GSPC'
    ]
    all_X = []
    all_y = []
    all_meta = []
    
    strat = WizardScalpStrategy()
    # Load Macro Data
    dxy_df = fetch_data('DX-Y.NYB', period='1y', interval='1h')
    btc_df = fetch_data('BTC-USD', period='1y', interval='1h')
    
    for ticker in tickers:
        df = fetch_data(ticker)
        if df.empty or len(df) < 500: continue
        
        # 1. Apply Scalp Strategy
        df = strat.apply(df)
        
        # 2. Calculate Features (Passing Macro)
        df = calculate_ml_features(df, macro_df=dxy_df, crypto_macro_df=btc_df)
        
        # 3. Triple Barrier Labeling
        # Target: 1.5% Profit, 1.0% Stop, 12 bars (12 Hours)
        tp_mult = 0.015
        sl_mult = 0.010
        horizon = 12
        
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
                    pnl = -sl_mult
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
                    pnl = -sl_mult
            
            # Features
            features = [
                'volatility', 'rsi', 'ma_dist', 'adx', 'rvol', 'atr_pct', 'cycle_regime',
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

def train_scalp_model():
    X, y, meta = generate_scalp_dataset()
    if len(X) < 100:
        print("Not enough signals to train.")
        return

    # Purged TimeSeries Split
    meta = meta.sort_values('entry_time').reset_index(drop=True)
    split_idx = int(len(meta) * 0.75)
    
    # 24h Purge Gap
    test_start = meta.iloc[split_idx]['entry_time']
    train_end_limit = test_start - pd.Timedelta(hours=24)
    
    train_mask = meta['entry_time'] < train_end_limit
    test_mask = meta.index >= split_idx
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTraining Model (Train: {len(X_train)}, Test: {len(X_test)})")
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    if len(clf.classes_) < 2:
        print("Model only trained on one class. Cannot optimize threshold.")
        y_prob = np.zeros(len(X_test))
    else:
        y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Threshold Optimization
    best_thresh = 0.4
    best_pnl = -1
    
    print("\n| Thresh | Win Rate | PnL |")
    for thresh in np.arange(0.35, 0.65, 0.05):
        mask = y_prob >= thresh
        if mask.sum() < 5: continue
        
        filtered_meta = meta[test_mask][mask]
        wr = filtered_meta['label'].mean()
        pnl = filtered_meta['pnl'].sum()
        print(f"| {thresh:.2f} | {wr:.2f} | {pnl:.2f} |")
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_thresh = thresh
            
    # Save
    model_data = {
        'model': clf,
        'threshold': best_thresh,
        'features': [
            'volatility', 'rsi', 'ma_dist', 'adx', 'rvol', 'atr_pct', 'cycle_regime',
            'dxy_ret', 'dxy_corr', 'dxy_dist',
            'btc_corr', 'btc_mom'
        ]
    }
    joblib.dump(model_data, 'wizard_scalp_ml_model.pkl')
    print(f"\nSaved Scalp Model with Threshold {best_thresh:.2f}")

if __name__ == "__main__":
    train_scalp_model()

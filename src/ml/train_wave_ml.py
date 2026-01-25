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

from src.strategies.wizard_wave import WizardWaveStrategy
from src.core.feature_engine import calculate_ml_features
from src.utils.paths import get_model_path

def fetch_data(ticker, period='5y', interval='1d'):
    print(f"Fetching {ticker} ({interval})...")
    
    y_interval = interval
    resample = None
    if interval == '12h':
        y_interval = '1h'
        resample = '12H'
    elif interval == '4d':
        y_interval = '1d'
        resample = '4D'
    elif interval == '1w':
        y_interval = '1wk'
    
    df = yf.Ticker(ticker).history(period=period, interval=y_interval)
    if df.empty: return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    
    if resample:
        logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        df = df.resample(resample).apply(logic).dropna()

    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']]

def generate_wave_dataset(interval='1d'):
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
    dxy_df = fetch_data('DX-Y.NYB', period='5y', interval=interval)
    btc_df = fetch_data('BTC-USD', period='5y', interval=interval)
    
    # Adjust targets for 3R ratios
    # Default (Daily): 10% Profit, 3.3% Stop (3.03 RR)
    tp_mult = 0.10
    sl_mult = 0.033
    horizon = 21
    
    if interval == '1w':
        tp_mult, sl_mult, horizon = 0.20, 0.06, 12
    elif interval == '4d':
        tp_mult, sl_mult, horizon = 0.15, 0.05, 15
    elif interval in ['12h', '4h']:
        tp_mult, sl_mult, horizon = 0.06, 0.02, 24

    for ticker in tickers:
        df = fetch_data(ticker, interval=interval)
        if df.empty or len(df) < 300: continue
        
        # 1. Apply Wave Strategy
        df = strat.apply(df)
        
        # 2. Calculate Features
        df = calculate_ml_features(df, macro_df=dxy_df, crypto_macro_df=btc_df)
        
        # Adjust for Asset Class Volatility
        curr_tp = tp_mult
        curr_sl = sl_mult
        if '=X' in ticker or '^' in ticker: # Forex / Indices
            curr_tp, curr_sl = tp_mult * 0.5, sl_mult * 0.5
        
        signals = df[df['signal_type'] != 'NONE'].copy()
        print(f"  {ticker} ({interval}): Found {len(signals)} signals.")
        
        for idx, row in signals.iterrows():
            future = df.loc[idx:].iloc[1:horizon+1]
            if future.empty: continue
            
            entry = row['close']
            target = entry * (1 + curr_tp) if 'LONG' in row['signal_type'] else entry * (1 - curr_tp)
            stop = entry * (1 - curr_sl) if 'LONG' in row['signal_type'] else entry * (1 + curr_sl)
            
            label = 0
            pnl = 0
            
            if 'LONG' in row['signal_type']:
                hit_tp = future[future['high'] >= target]
                hit_sl = future[future['low'] <= stop]
                tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                if tp_idx < sl_idx: label, pnl = 1, curr_tp
                else: 
                  if not hit_sl.empty: pnl = -curr_sl
                  else: pnl = (future['close'].iloc[-1] - entry) / entry
            else:
                hit_tp = future[future['low'] <= target]
                hit_sl = future[future['high'] >= stop]
                tp_idx = hit_tp.index[0] if not hit_tp.empty else pd.Timestamp.max
                sl_idx = hit_sl.index[0] if not hit_sl.empty else pd.Timestamp.max
                if tp_idx < sl_idx: label, pnl = 1, curr_tp
                else:
                  if not hit_sl.empty: pnl = -curr_sl
                  else: pnl = (entry - future['close'].iloc[-1]) / entry
            
            # Features
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

def train_wave_model(interval='1d'):
    X, y, meta = generate_wave_dataset(interval)
    if len(X) < 80:
        print(f"Not enough {interval} signals to train.")
        return

    meta = meta.sort_values('entry_time').reset_index(drop=True)
    split_idx = int(len(meta) * 0.8)
    
    test_start = meta.iloc[split_idx]['entry_time']
    train_end_limit = test_start - pd.Timedelta(days=interval.replace('w','7d').replace('d','1d'))
    
    train_mask = meta['entry_time'] < train_end_limit
    test_mask = meta.index >= split_idx
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"\nTraining {interval} Wave Model (Train: {len(X_train)}, Test: {len(X_test)})")
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Save
    model_data = {
        'model': clf,
        'threshold': 0.55,
        'features': [
            'volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 
            'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 
            'month_sin', 'cycle_regime', 'close_frac',
            'dxy_ret', 'dxy_corr', 'dxy_dist',
            'btc_corr', 'btc_mom'
        ],
        'interval': interval
    }
    filename = f"model_{interval}.pkl"
    save_path = get_model_path(filename)
    joblib.dump(model_data, save_path)
    print(f"[SUCCESS] Saved {interval} Model: {save_path}")

if __name__ == "__main__":
    tfs = ['1w', '4d', '1d', '12h', '4h']
    if len(sys.argv) > 1:
        tfs = [sys.argv[1]]
    for tf in tfs:
        train_wave_model(tf)

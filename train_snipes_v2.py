import pandas as pd
import numpy as np
import concurrent.futures
from data_fetcher import fetch_data
from feature_snipes import SpellSnipesFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def get_triple_barrier_labels_rr(df, target_r=2.0, barrier_len=24, atr_mult=1.5):
    """
    Triple Barrier with Risk:Reward > 1.5.
    SL = ATR * 1.5 (Dynamic Volatility Based).
    TP = SL * target_r (e.g. 2.0).
    Outcome: 1 (TP Hit), -1 (SL Hit), 0 (Time)
    """
    labels = pd.Series(index=df.index, data=0)
    
    # Calculate ATR for Dynamic Stops
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    future_window = barrier_len
    
    for t in range(len(df) - future_window):
        if np.isnan(atr.iloc[t]): continue
        
        curr_slice = df.iloc[t]
        future_slice = df.iloc[t+1 : t+1+future_window]
        
        entry = curr_slice['close']
        
        # Dynamic Limits
        vol = atr.iloc[t]
        sl_dist = vol * atr_mult
        
        # Minimum SL distance (0.1% for TradFi, 0.3% for Crypto) to avoid noise
        # This prevents 0.2% tight stops in low vol
        min_sl = entry * 0.001
        if sl_dist < min_sl: sl_dist = min_sl
        
        tp_dist = sl_dist * target_r
        
        tp_price_long = entry + tp_dist
        sl_price_long = entry - sl_dist
        
        # Label for LONG Only first? Or assume direction based on future?
        # Standard: We want model to Predict Direction.
        # So we check BOTH.
        
        # Check Long
        l_ret = 0
        long_tp = future_slice[future_slice['high'] >= tp_price_long].first_valid_index()
        long_sl = future_slice[future_slice['low'] <= sl_price_long].first_valid_index()
        
        if long_tp and long_sl:
            if long_tp < long_sl: l_ret = 1
            else: l_ret = -1
        elif long_tp: l_ret = 1
        elif long_sl: l_ret = -1
        
        # Check Short
        s_ret = 0
        tp_price_short = entry - tp_dist
        sl_price_short = entry + sl_dist
        
        short_tp = future_slice[future_slice['low'] <= tp_price_short].first_valid_index()
        short_sl = future_slice[future_slice['high'] >= sl_price_short].first_valid_index()
        
        if short_tp and short_sl:
            if short_tp < short_sl: s_ret = 1
            else: s_ret = -1
        elif short_tp: s_ret = 1
        elif short_sl: s_ret = -1
        
        # Final Label
        # If Long connects, Label 1.
        # If Short connects, Label -1.
        # If Both connect (chop), Label 0 or ambiguous?
        # If Neither, 0.
        
        if l_ret == 1 and s_ret != 1:
            labels.iloc[t] = 1
        elif s_ret == 1 and l_ret != 1:
            labels.iloc[t] = -1
        else:
            labels.iloc[t] = 0
            
    return labels

def process_asset_snipes_dynamic(symbol, atype, timeframe='15m'):
    print(f"Processing {symbol} ({timeframe})...")
    limit = 6000 
    df = fetch_data(symbol, atype, timeframe, limit=limit)
    if df is None or len(df) < 500: return None
    
    df = SpellSnipesFeatures.add_features(df)
    df = df.dropna()
    if df.empty: return None
    
    # Labeling
    # 2.0R Target. ATR 1.5 SL.
    # Barrier 24 bars (6 hours for 15m)
    labels = get_triple_barrier_labels_rr(df, target_r=2.0, barrier_len=24, atr_mult=1.5)
    df['target'] = labels
    
    return df

def run_training_snipes_stratified():
    # Split Training: Crypto Model vs TradFi Model
    
    # 1. Crypto Model
    print("\n=== Training Crypto Snipes Model ===")
    crypto_assets = [('BTC/USDT', 'crypto'), ('AVAX/USDT', 'crypto'), ('ETH/USDT', 'crypto')]
    train_and_save(crypto_assets, "model_snipes_crypto.pkl")
    
    # 2. TradFi Model
    print("\n=== Training TradFi Snipes Model ===")
    trad_assets = [('^NDX', 'trad'), ('^GSPC', 'trad'), ('GC=F', 'trad'), ('EURUSD=X', 'forex')]
    train_and_save(trad_assets, "model_snipes_tradfi.pkl")

def train_and_save(assets, filename):
    all_data = []
    for sym, atype in assets:
        try:
            df = process_asset_snipes_dynamic(sym, atype)
            if df is not None: all_data.append(df)
        except Exception as e: print(e)
            
    if not all_data: return
    
    full_df = pd.concat(all_data)
    
    # Time Split
    split = int(len(full_df) * 0.8)
    train = full_df.iloc[:split]
    test = full_df.iloc[split:]
    
    exclude = ['open','high','low','close','volume','target','time','local_time']
    feats = [c for c in train.columns if c not in exclude and 'date' not in c]
    
    X_train = train[feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    y_train = train['target']
    X_test = test[feats].replace([np.inf,-np.inf],np.nan).fillna(0)
    y_test = test['target']
    
    # Class Balance check
    print(f"Target Dist: {y_train.value_counts(normalize=True)}")
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    print(f"Results for {filename}:")
    print(classification_report(y_test, clf.predict(X_test)))
    
    joblib.dump(clf, filename)

if __name__ == "__main__":
    run_training_snipes_stratified()

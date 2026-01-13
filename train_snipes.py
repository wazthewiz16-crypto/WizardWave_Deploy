import pandas as pd
import numpy as np
import concurrent.futures
from data_fetcher import fetch_data
from feature_snipes import SpellSnipesFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report
import joblib

def process_asset_snipes(symbol, atype, timeframe='15m'):
    print(f"Processing {symbol} ({timeframe})...")
    
    # Fetch Data (Max available for training)
    limit = 5000 # ~60 days for 15m
    df = fetch_data(symbol, atype, timeframe, limit=limit)
    
    if df is None or len(df) < 500: return None
    
    # Fearure Eng
    df = SpellSnipesFeatures.add_features(df)
    df = df.dropna()
    
    if df.empty: return None
    
    # Labeling (Triple Barrier)
    # Target: 0.5% move in 20 bars?
    # Crypto: 0.5% is easy. TradFi: 0.2%?
    tgt = 0.005 if atype == 'crypto' else 0.002
    
    labels = SpellSnipesFeatures.get_triple_barrier_labels(df, sl_tp_ratio=1.0, barrier_len=12, min_ret=tgt)
    
    df['target'] = labels
    
    # Drop valid labels (0s are Time Limit = Neutral? Or ignore?)
    # Triple Barrier: 1=Buy, -1=Sell (if shorting), 0=Hold/Exit
    # We want to predict Direction.
    # Let's train for 1 (Long) vs Rest? Or Multi-class?
    # Strategy says "Long/Short trading signals".
    # Multi-class: 1, -1, 0.
    
    return df

def run_training_snipes():
    assets = [
        ('BTC/USDT', 'crypto'),
        ('AVAX/USDT', 'crypto'), 
        ('^NDX', 'trad'), 
        ('^GSPC', 'trad'), 
        ('GC=F', 'trad'), 
        ('EURUSD=X', 'forex')
    ]
    
    all_data = []
    
    for sym, atype in assets:
        df = process_asset_snipes(sym, atype)
        if df is not None:
            # Add Symbol col
            all_data.append(df)
            
    if not all_data:
        print("No training data.")
        return
        
    full_df = pd.concat(all_data)
    
    # Train/Test Split (Time based?)
    # Just simplistic split for report
    # Last 60 days is "Test".
    # Wait, we only fetched ~60 days.
    # So we need to train on first 80%, test on 20%?
    # Or Train on other assets, test on target?
    # Let's do 80/20 sequential split.
    
    split_idx = int(len(full_df) * 0.8)
    train = full_df.iloc[:split_idx]
    test = full_df.iloc[split_idx:]
    
    # Features
    exclude = ['open','high','low','close','volume','target','time','local_time']
    feats = [c for c in train.columns if c not in exclude and 'date' not in c]
    
    print(f"Features: {feats}")
    
    X_train = train[feats]
    y_train = train['target']
    X_test = test[feats]
    y_test = test['target']
    
    # Clean inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    
    print("\n=== Spot Snipes Model Results (Test Set) ===")
    print(classification_report(y_test, preds))
    
    # Feature Importance
    imps = pd.Series(clf.feature_importances_, index=feats).sort_values(ascending=False)
    print("\nTop 5 Features:")
    print(imps.head(5))
    
    # Backtest Simulation on Test Set
    # Simple PnL: If pred=1, ret = next_bar_ret. If pred=-1, ret = -next_bar_ret.
    # Note: Triple Barrier labels are "Did we hit PT before SL?".
    # So if Pred=1 and Label=1 -> Win.
    # If Pred=1 and Label=-1 -> Loss.
    # If Pred=1 and Label=0 -> Scratch/TimeLimit.
    
    # Let's calc Win Rate per Class
    test_res = pd.DataFrame({'pred': preds, 'actual': y_test})
    
    # Longs
    longs = test_res[test_res['pred'] == 1]
    if not longs.empty:
        wr_long = len(longs[longs['actual'] == 1]) / len(longs)
        print(f"\nLong Win Rate (Precision): {wr_long:.1%}")
    
    # Shorts
    shorts = test_res[test_res['pred'] == -1]
    if not shorts.empty:
        wr_short = len(shorts[shorts['actual'] == -1]) / len(shorts)
        print(f"Short Win Rate (Precision): {wr_short:.1%}")
        
    # Save Model
    joblib.dump(clf, "model_snipes.pkl")
    print("Model saved to model_snipes.pkl. Ready for app integration.")

if __name__ == "__main__":
    run_training_snipes()

import pandas as pd
import numpy as np
import json
import joblib
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_fetcher import fetch_data
from strategy_ichimoku import IchimokuStrategy
from strategy_cls import CLSRangeStrategy
from feature_engine import calculate_ichi_features, calculate_cls_features

# Configurations
TRADFI_CFG = {"tenkan": 20, "kijun": 60, "span_b": 120, "displacement": 30}
CRYPTO_CFG = {"tenkan": 7, "kijun": 21, "span_b": 42, "displacement": 21}

def load_assets():
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
    return config['assets']

def get_asset_info(symbol):
    if any(x in symbol for x in ['=X', '=F', '^', '-']):
        return 'trad'
    return 'crypto'

def simulate_ichi_outcome(df, entry_idx, signal_type, is_crypto):
    # Kijun Trail Simulation
    entry_price = df.iloc[entry_idx]['close']
    future_df = df.iloc[entry_idx+1:]
    
    if future_df.empty: return 0
    
    exit_price = entry_price
    
    for i, row in future_df.iterrows():
        # Exit Condition: Close crosses Kijun
        if signal_type == "LONG" and row['close'] < row['kijun']:
            exit_price = row['close']
            break
        elif signal_type == "SHORT" and row['close'] > row['kijun']:
            exit_price = row['close']
            break
            
    # Calculate Return
    if signal_type == "LONG":
        ret = (exit_price - entry_price) / entry_price
    else:
        ret = (entry_price - exit_price) / entry_price
        
    ret -= 0.001 # Fees
    
    # Target Threshold
    threshold = 0.005 if is_crypto else 0.002
    return 1 if ret > threshold else 0

def simulate_cls_outcome(df, entry_idx, signal_type, tp, sl):
    entry_price = df.iloc[entry_idx]['close']
    future_df = df.iloc[entry_idx+1:]
    
    if future_df.empty: return 0
    
    for i, row in future_df.iterrows():
        # Check High/Low for hits
        if signal_type == "CLS_LONG":
            if row['low'] <= sl: return 0 # Hit SL
            if row['high'] >= tp: return 1 # Hit TP
        else: # SHORT
            if row['high'] >= sl: return 0 # Hit SL
            if row['low'] <= tp: return 1 # Hit TP
            
    # If unclosed, treat as 0 (timeout/stale)
    return 0

def run_training():
    assets = load_assets()
    
    X_ichi, y_ichi = [], []
    X_cls, y_cls = [], []
    
    print("Gathering Training Data...")
    
    for symbol in assets:
        asset_type = get_asset_info(symbol)
        is_crypto = (asset_type == 'crypto')
        
        # --- ICHIMOKU DATA ---
        cfg = CRYPTO_CFG if is_crypto else TRADFI_CFG
        ichi_strat = IchimokuStrategy(**cfg)
        
        target_tfs = ["1d"]
        if not is_crypto: target_tfs = ["4h", "1d"]
        
        for tf in target_tfs:
            try:
                df = fetch_data(symbol, asset_type, tf, limit=1000)
                if df is None or len(df) < 150: continue
                
                # Apply Strat
                df = ichi_strat.apply_strategy(df, tf)
                df = calculate_ichi_features(df)
                
                signals = df[df['signal_type'].notna()]
                
                # Simulate Checks
                for idx in range(len(signals)):
                    row = signals.iloc[idx]
                    # Find integer index in original df
                    try:
                        orig_idx = df.index.get_loc(row.name)
                        target = simulate_ichi_outcome(df, orig_idx, row['signal_type'], is_crypto)
                        
                        feat_row = row[['tk_gap', 'price_to_kijun', 'cloud_width', 'dist_to_cloud_top', 'chikou_mom', 'adx', 'rsi', 'volatility']]
                        
                        X_ichi.append(feat_row)
                        y_ichi.append(target)
                    except: pass
            except Exception as e:
                # print(f"Ichi Error {symbol}: {e}")
                pass

        # --- CLS DATA ---
        try:
            # Need HTF (1d) and LTF (1h)
            df_htf = fetch_data(symbol, asset_type, '1d', limit=1000)
            df_ltf = fetch_data(symbol, asset_type, '1h', limit=5000) # Deep fetch for LTF matches
            
            if df_htf is not None and df_ltf is not None and not df_htf.empty and not df_ltf.empty:
                cls_strat = CLSRangeStrategy()
                df_cls = cls_strat.apply_mtf(df_htf, df_ltf)
                df_cls = calculate_cls_features(df_cls)
                
                signals = df_cls[df_cls['signal_type'].isin(["CLS_LONG", "CLS_SHORT"])]
                
                for idx in range(len(signals)):
                    row = signals.iloc[idx]
                    try:
                        orig_idx = df_cls.index.get_loc(row.name)
                        target = simulate_cls_outcome(df_cls, orig_idx, row['signal_type'], row['target_price'], row['stop_loss'])
                        
                        feat_row = row[['dist_to_tp', 'dist_to_sl', 'rr_ratio', 'rsi', 'adx', 'atr_pct', 'dist_sma50']]
                        
                        X_cls.append(feat_row)
                        y_cls.append(target)
                    except: pass
        except Exception as e:
            # print(f"CLS Error {symbol}: {e}")
            pass
            
    # --- TRAINING ---
    
    # ICHIMOKU
    if X_ichi:
        print("\nTraining Ichimoku Model...")
        X_df = pd.DataFrame(X_ichi)
        y_df = pd.Series(y_ichi)
        
        print(f"Ichi Samples: {len(X_df)}. Win Rate: {y_df.mean():.2%}")
        
        clf_i = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=42)
        clf_i.fit(X_df, y_df)
        
        print(classification_report(y_df, clf_i.predict(X_df)))
        
        joblib.dump(clf_i, "model_ichimoku.pkl")
        with open("features_ichimoku.json", "w") as f:
            json.dump(list(X_df.columns), f)
            
    # CLS
    if X_cls:
        print("\nTraining CLS Model...")
        X_df = pd.DataFrame(X_cls)
        y_df = pd.Series(y_cls)
        
        print(f"CLS Samples: {len(X_df)}. Win Rate: {y_df.mean():.2%}")
        
        clf_c = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=42)
        clf_c.fit(X_df, y_df)
        
        print(classification_report(y_df, clf_c.predict(X_df)))
        
        joblib.dump(clf_c, "model_cls.pkl")
        with open("features_cls.json", "w") as f:
            json.dump(list(X_df.columns), f)

if __name__ == "__main__":
    run_training()

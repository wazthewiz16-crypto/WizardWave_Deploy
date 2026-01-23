import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report

# Import Project Modules
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features
from pipeline import get_asset_type

# Configuration
FEATURES_ALL = [
    'volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 
    'bb_width', 'candle_ratio', 'atr_pct', 'mfi',
    'mango_d1_dist', 'mango_d2_dist', 'upper_zone_dist', 'lower_zone_dist'
]

OLD_FEATURES = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']

def load_config():
    with open('strategy_config.json', 'r') as f:
        return json.load(f)

def get_data_for_training(config, mode='htf'):
    print(f"Fetching Training Data for {mode.upper()}...")
    all_X = []
    all_y = []
    
    # Select Timeframes
    tfs = config[mode]['timeframes']
    
    # Reduce asset list for speed if needed, but let's try full
    assets = config['assets']
    
    for symbol in assets:
        for tf in tfs:
            try:
                # Fetch Max Data
                asset_type = get_asset_type(symbol)
                fetch_type = 'trad' if asset_type in ['forex', 'trad'] else 'crypto'
                if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
                
                # Use '1h' etc based on config codes
                limit = 2000 # ~80 days for 1h, plenty
                
                df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=limit)
                if df.empty: continue
                
                # Apply Strategy
                # Apply Strategy
                if mode in ['htf', 'mtf']:
                    strat = WizardWaveStrategy()
                else:
                    strat = WizardScalpStrategy(lookback=8)
                    
                df = strat.apply(df)
                df = calculate_ml_features(df)
                
                # Labeling (Profit-First Target)
                if mode == 'htf':
                    horizon = 48
                    reward_mult = 3.5
                    risk_mult = 1.0
                elif mode == 'mtf':
                    horizon = 24
                    reward_mult = 1.75
                    risk_mult = 1.0
                else: # ltf
                    horizon = 12
                    reward_mult = 2.0
                    risk_mult = 1.0

                if 'atr_pct' not in df.columns: df['atr_pct'] = 0.01
                
                reward_thresh = df['atr_pct'] * reward_mult
                risk_thresh = df['atr_pct'] * risk_mult
                
                df['future_ret'] = (df['close'].shift(-horizon) - df['close']) / df['close']
                df['future_min_ret'] = (df['low'].rolling(horizon).min().shift(-horizon) - df['close']) / df['close']
                
                df['target'] = ((df['future_ret'] > reward_thresh) & (df['future_min_ret'] > -risk_thresh)).astype(int)
                
                # Drop NaNs
                df = df.dropna()
                
                if df.empty: continue
                
                all_X.append(df[FEATURES_ALL])
                all_y.append(df['target'])
                
            except Exception as e:
                print(f"Error {symbol} {tf}: {e}")
                
    if not all_X:
        return None, None
        
    X = pd.concat(all_X)
    y = pd.concat(all_y)
    
    return X, y

def train_and_optimize(X, y, mode='htf'):
    # Correlation Filter
    print("Running Correlation Filter...")
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    print(f"Dropping correlated features: {to_drop}")
    X_reduced = X.drop(columns=to_drop)
    
    final_features = list(X_reduced.columns)
    
    # Train
    print(f"Training {mode.upper()} Model on {len(X)} samples with {len(final_features)} features...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_reduced, y)
    
    return clf, final_features, to_drop

def backtest_model(old_model, new_model, final_new_features, config, mode='htf'):
    # Backtest on Recent Data (Last 24H equiv + 1 year sample)
    # We essentially simulate signals
    
    print(f"Backtesting {mode.upper()}...")
    
    old_signals = 0
    new_signals = 0
    old_wins = 0
    new_wins = 0
    
    # Just run on a few assets to represent "24 Hours" and "1 Year"
    # Actually, we can just split the test set
    
    return {
        "old_signals": 100, "new_signals": 85, # Placeholder simulation results
        "old_acc": 0.55, "new_acc": 0.60
    }

def run_experiment():
    config = load_config()
    
    results = {}
    
    for mode in ['htf', 'mtf', 'ltf']:
        # 1. Get Data
        X, y = get_data_for_training(config, mode)
        if X is None:
            print(f"No data for {mode}")
            continue
            
        # 2. Train New
        new_model, new_feats, dropped = train_and_optimize(X, y, mode)
        
        # 3. Save New Model (and feature list)
        joblib.dump(new_model, f"model_{mode}_v2.pkl")
        with open(f"features_{mode}.json", "w") as f:
            json.dump(new_feats, f)
            
        # 4. Compare (Simulated for speed in this context, or real check)
        # We check prediction distribution
        
        # Old Model Check
        try:
            old_model = joblib.load(f"model_{mode}.pkl")
            # Old model expects OLD_FEATURES
            # We must ensure X has them. X comes from calculate_ml_features which has ALL.
            # So selecting works.
            X_old = X[OLD_FEATURES]
            probs_old = old_model.predict_proba(X_old)[:, 1]
            signals_old = (probs_old > 0.40).sum()
        except:
            signals_old = 0
            
        # New Model Check
        X_new = X[new_feats]
        probs_new = new_model.predict_proba(X_new)[:, 1]
        signals_new = (probs_new > 0.40).sum()
        
        results[mode] = {
            "dropped": dropped,
            "final_features": new_feats,
            "signals_old_count": int(signals_old),
            "signals_new_count": int(signals_new),
            "change_pct": (signals_new - signals_old) / signals_old if signals_old > 0 else 0
        }
        
    print("\n--- EXPERIMENT REPORT ---")
    print(json.dumps(results, indent=2))
    
    # Overwrite Production Models
    print("Deploying models...")
    for mode in ['htf', 'mtf', 'ltf']:
        if os.path.exists(f"model_{mode}_v2.pkl"):
            # Backup
            if os.path.exists(f"model_{mode}.pkl"):
                os.rename(f"model_{mode}.pkl", f"model_{mode}_backup.pkl")
            # Deploy
            os.rename(f"model_{mode}_v2.pkl", f"model_{mode}.pkl")
            
    print("Models deployed.")

if __name__ == "__main__":
    run_experiment()

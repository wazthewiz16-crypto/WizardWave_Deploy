import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, classification_report, accuracy_score
from sklearn.base import clone

# Project Modules
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
from feature_engine import calculate_ml_features
from pipeline import get_asset_type

def load_config():
    with open('strategy_config.json', 'r') as f:
        return json.load(f)

def get_data_for_challenger(config, mode='4h', limit=3000):
    print(f"--- Gathering Data for {mode.upper()} Challenger ---")
    all_X = []
    all_y = []
    
    if mode not in config['models']:
        print(f"Error: Mode {mode} not found in config['models']")
        return None, None
        
    tfs = config['models'][mode]['timeframes']
    assets = config['assets'] 
    
    # Selection of diverse assets for robust testing
    test_assets = [a for a in assets if '/' in a or '^' in a or '=' in a][:5]
    
    for symbol in test_assets:
        for tf in tfs:
            try:
                asset_type = get_asset_type(symbol)
                fetch_type = 'trad' if asset_type in ['forex', 'trad'] else 'crypto'
                if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
                
                df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=limit)
                if df is None or df.empty: continue
                
                strat = WizardWaveStrategy()
                df = strat.apply(df)
                df = calculate_ml_features(df)
                
                # Labeling (Return-based)
                horizon = 24
                df['future_ret'] = (df['close'].shift(-horizon) - df['close']) / df['close']
                # Target: Price is higher in horizon periods
                df['target'] = (df['future_ret'] > 0.005).astype(int) 
                
                df = df.dropna()
                if df.empty: continue
                
                # Use standard features
                features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
                all_X.append(df[features])
                all_y.append(df['target'])
            except Exception as e:
                pass
                
    if not all_X: return None, None
    X = pd.concat(all_X)
    y = pd.concat(all_y)
    return X, y

def run_challenger_suit():
    config = load_config()
    X, y = get_data_for_challenger(config, mode='4h')
    
    if X is None:
        print("Error: Could not gather enough data for testing.")
        return

    # Time-based split (Mock for research)
    split = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "RandomForest (Current Champion)": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    print("\n--- ML Model Challenger Leaderboard ---")
    print("-" * 50)
    
    leaderboard = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        # We value Precision (Quality of "Buy" signal)
        precision = precision_score(y_test, preds, zero_division=0)
        accuracy = accuracy_score(y_test, preds)
        
        leaderboard.append({
            "Model": name,
            "Precision": precision,
            "Accuracy": accuracy
        })

    # Sort by Precision
    leaderboard = sorted(leaderboard, key=lambda x: x['Precision'], reverse=True)
    
    for i, entry in enumerate(leaderboard):
        print(f"{i+1}. {entry['Model']} - Precision: {entry['Precision']:.2%} | Accuracy: {entry['Accuracy']:.2%}")

    best_model = leaderboard[0]['Model']
    improvement = leaderboard[0]['Precision'] - leaderboard[1]['Precision']
    
    print("-" * 50)
    print(f"\nRECOMMENDATION: {best_model}")
    if improvement > 0.02:
        print(f"Switch recommended! +{improvement:.2%} improvement in signal quality.")
    else:
        print("Current champion remains stable. Improvements are marginal (<2%).")

if __name__ == "__main__":
    run_challenger_suit()

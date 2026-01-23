
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score

# Add Project Root to Path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)

try:
    from strategy import WizardWaveStrategy
    from feature_engine import calculate_ml_features
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
ASSETS = ['BTC-USD', 'ETH-USD', 'SOL-USD'] # Test Assets
TRAIN_MONTHS = 6
TEST_MONTHS = 1
# To Test Jan 2025, we train July 24 - Dec 24.
START_DATE = datetime(2024, 7, 1) 
ITERATIONS = 12 # Jan 2025 -> Dec 2025

def fetch_data(symbol, start_date):
    """Fetch data from start_date to cover training + testing buffers."""
    # We need enough buffer for lookback (strategy)
    buffer_start = start_date - timedelta(days=60)
    # End date covers through end of 2025
    df = yf.Ticker(symbol).history(start=buffer_start, end=datetime(2026, 1, 1), interval='1d')
    if df.empty: return df
    
    df = df.reset_index()
    df.rename(columns={'Date': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    
    # TZ Handling
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
        df.set_index('datetime', inplace=True)
        
    return df[['open', 'high', 'low', 'close', 'volume']]

def train_model(train_df):
    """Trains a Random Forest on the dataframe."""
    # 1. Target Generation: Simple Next Candle Close > Current Close (for demo) 
    # OR Strategy Target: Hit TP before SL?
    
    df = train_df.copy()
    
    # Features
    df = calculate_ml_features(df)
    
    # Target: 1 if Return in next 5 days > 4% (Align w/ Volatility)
    df['target'] = 0
    
    future_window = 7
    threshold_pct = 0.09 # Target 9% (Production 1D TP)
    
    # Vectorized Target (approx)
    indexer = df.index
    for i in range(len(df) - future_window):
        curr = df.iloc[i]['close']
        # Look forward
        future_high = df.iloc[i+1 : i+future_window+1]['high'].max()
        future_low = df.iloc[i+1 : i+future_window+1]['low'].min()
        
        # Simple specific labelling: Hit TP (9%) before SL (3%)
        # Production SL is 3%
        
        # Check simple first hit
        # This is a simplification for speed in WFO loop
        if future_high > curr * (1 + threshold_pct):
             # Did it hit SL first?
             # We can't know EXACTLY without intraday, but we check if low was violated
             if future_low > curr * 0.97: # SL not hit
                 df.at[indexer[i], 'target'] = 1
    
    # Clean
    # MATCHING PRODUCTION FEATURE SET
    features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'rvol', 'month_sin', 'month_cos', 'cycle_regime'] 
    # Check intersection
    valid_feats = [f for f in features if f in df.columns]
    
    df_clean = df.dropna(subset=valid_feats + ['target'])
    
    if len(df_clean) < 50:
        return None, valid_feats
        
    X = df_clean[valid_feats]
    y = df_clean['target']
    
    # PRODUCTION SETTINGS (from strategy_config.json)
    # n_estimators: 200, max_depth: 10
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    
    return model, valid_feats

def backtest_period(df, model, features, start_date, end_date):
    """Simulate trading on test data using Full Strategy Logic."""
    mask = (df.index >= start_date) & (df.index < end_date)
    test_df = df.loc[mask].copy()
    
    if test_df.empty: return 0.0, 0
    
    # 1. Apply Strategy Logic (The Cloud, KVO, etc.)
    # We default to WizardWave (1D) for WFO validation
    strategy = WizardWaveStrategy() 
    test_df = strategy.apply(test_df)

    # 2. Calculate ML Features
    test_df = calculate_ml_features(test_df)
    
    # Fill Nans
    for f in features:
        if f not in test_df.columns: test_df[f] = 0
    test_df[features] = test_df[features].fillna(0)
    
    # 3. Predict Probabilities
    if model:
        test_df['prob'] = model.predict_proba(test_df[features])[:, 1]
    else:
        test_df['prob'] = 0.5
    
    test_df['prob'] = test_df['prob'].fillna(0.0)

    # 4. Simulate Trades (Triple Barrier + Strategy Entry)
    balance = 1000.0
    position = 0
    entry_price = 0
    trades = 0
    pnl_pct_accum = 0.0
    
    # Production Threshold
    prod_threshold = 0.42
    
    # Production TP/SL (1D Crypto)
    tp_pct = 0.09
    sl_pct = 0.03
    
    for idx, row in test_df.iterrows():
        price = row['close']
        prob = row['prob']
        sig = row.get('signal_type', 'NONE')
        
        # EXIT
        if position != 0:
            pnl = 0
            exit_trade = False
            
            if position == 1: # LONG
                if price >= entry_price * (1 + tp_pct): # TP
                    pnl = (price - entry_price) / entry_price
                    exit_trade = True
                elif price <= entry_price * (1 - sl_pct): # SL
                    pnl = (price - entry_price) / entry_price
                    exit_trade = True
            elif position == -1: # SHORT
                if price <= entry_price * (1 - tp_pct): # TP
                    pnl = (entry_price - price) / entry_price
                    exit_trade = True
                elif price >= entry_price * (1 + sl_pct): # SL
                    pnl = (entry_price - price) / entry_price
                    exit_trade = True
            
            if exit_trade:
                pnl_pct_accum += pnl
                position = 0
        
        # ENTRY (Must match Strategy Signal + ML Confidence)
        if position == 0 and prob >= prod_threshold:
            if 'LONG' in sig: # Matches LONG_ZONE, LONG_REV
                position = 1
                entry_price = price
                trades += 1
            elif 'SHORT' in sig: # Matches SHORT_ZONE, SHORT_REV
                position = -1
                entry_price = price
                trades += 1
                
    return pnl_pct_accum, trades

def run_wfo():
    print(f"=== WALK FORWARD OPTIMIZATION ===")
    print(f"Train Window: {TRAIN_MONTHS} Months | Test Window: {TEST_MONTHS} Month")
    
    global_results = []
    
    for asset in ASSETS:
        print(f"\nProcessing {asset}...")
        df_full = fetch_data(asset, START_DATE)
        
        # Iterations
        curr_start = START_DATE
        
        for i in range(ITERATIONS):
            # Define Windows
            train_end = curr_start + relativedelta(months=TRAIN_MONTHS)
            test_end = train_end + relativedelta(months=TEST_MONTHS)
            
            print(f"  Ref {i+1}: Train [{curr_start.date()} -> {train_end.date()}] | Test [{train_end.date()} -> {test_end.date()}]")
            
            # Slice Train
            train_mask = (df_full.index >= curr_start) & (df_full.index < train_end)
            df_train = df_full.loc[train_mask]
            
            # Train Model
            model, feats = train_model(df_train)
            
            if model:
                # Test
                pnl, trades = backtest_period(df_full, model, feats, train_end, test_end)
                result_str = f"PnL: {pnl*100:.2f}% ({trades} trades)"
                print(f"    Result: {result_str}")
                global_results.append({'Asset': asset, 'Period': f"{train_end.date()}", 'PnL': pnl, 'Trades': trades})
            else:
                print("    Failed to train (insufficient data)")
            
            # Walk Forward
            curr_start = curr_start + relativedelta(months=1)
            
    # Summary
    print("\n=== SUMMARY ===")
    res_df = pd.DataFrame(global_results)
    if not res_df.empty:
        print(res_df.groupby('Asset')['PnL'].sum() * 100)
    else:
        print("No trades generated.")

if __name__ == "__main__":
    run_wfo()

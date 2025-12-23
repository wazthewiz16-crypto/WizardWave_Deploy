import json
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features

# Load Configuration
with open('strategy_config.json', 'r') as f:
    config = json.load(f)

def get_asset_type(symbol):
    """
    Determines if an asset is 'crypto', 'forex', or 'trad' based on the symbol.
    """
    # Known Crypto Identifiers
    crypto_kw = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'BNB', 'LINK', 'ARB', 'AVAX', 'ADA', 'USDT']
    if any(k in symbol.upper() for k in crypto_kw):
        return 'crypto'
    
    # Forex
    if '=X' in symbol:
        return 'forex'

    # Common TradFi Patterns (Indices, Futures, Stocks)
    if symbol.startswith('^') or symbol.endswith('=F') or '-' in symbol:
        return 'trad'
        
    return 'trad'

def apply_triple_barrier(df, symbol, tf, group_config):
    """
    Applies Triple Barrier Method to label signals based on group config.
    """
    labels = []
    
    asset_type = get_asset_type(symbol)
    tb_config = group_config['triple_barrier']
    
    # Get PT/SL based on Asset Type
    if asset_type == 'crypto':
        pt = tb_config['crypto_pt']
        sl = tb_config['crypto_sl']
    elif asset_type == 'forex':
        pt = tb_config.get('forex_pt', tb_config['trad_pt'])
        sl = tb_config.get('forex_sl', tb_config['trad_sl'])
    else:
        pt = tb_config['trad_pt']
        sl = tb_config['trad_sl']
    
    
    # Weight
    weights_map = group_config.get('weights', {})
    sample_weight = weights_map.get(tf, 1.0)
    
    # Calculate Volatility (Sigma) if needed for Dynamic Barrier
    # Optimization: Only strictly needed for Crypto if configured, but calculating for all is fast and safe.
    if tb_config.get('crypto_use_dynamic', False) or tb_config.get('use_dynamic_barrier', False):
         df['sigma'] = df['close'].pct_change().ewm(span=36, adjust=False).std()
         df['sigma'] = df['sigma'].fillna(method='bfill').fillna(0.01)
    
    # Time Limit Calculation
    if tb_config.get('use_dynamic_barrier', False) and 'dynamic_time_limit_bars' in tb_config:
        time_limit = tb_config['dynamic_time_limit_bars']
    elif 'time_limit_bars' in tb_config:
        time_limit = tb_config['time_limit_bars']
    else:
        # Convert days to bars (Legacy HTF logic)
        days = tb_config.get('time_limit_days', 40)
        if tf == '1d': bars_per_day = 1
        elif tf == '4d': bars_per_day = 0.25
        else: bars_per_day = 1 # Fallback
        time_limit = int(days * bars_per_day)
        
    time_limit = max(1, time_limit)
    
    # Features (Centralized in feature_engine.py)
    df = calculate_ml_features(df)
    
    # Filter for Signals
    signal_indices = df[df['signal_type'] != 'NONE'].index
    
    for start_time in signal_indices:
        row = df.loc[start_time]
        signal_type = row['signal_type']
        entry_price = row['close']
        
        # Determine Direction
        # Both strategies use similar naming conventions or substrings
        if 'LONG' in signal_type:
            direction = 1
        else: # SHORT
            direction = -1
            
        try:
            start_iloc = df.index.get_loc(start_time)
        except KeyError:
            continue

        end_iloc = min(start_iloc + time_limit, len(df))
        future_window = df.iloc[start_iloc + 1 : end_iloc]
        
        if future_window.empty:
            continue
            
        outcome = 0 
        raw_ret = 0.0
        
        # Barrier Logic
        is_crypto_dynamic = (asset_type == 'crypto' and tb_config.get('crypto_use_dynamic', False))
        
        if is_crypto_dynamic:
             # Dynamic Volatility Based
             sigma = row.get('sigma', 0.01)
             k_pt = tb_config.get('crypto_dyn_pt_k', 0.5)
             k_sl = tb_config.get('crypto_dyn_sl_k', 0.5)
             
             d_pt = k_pt * sigma
             d_sl = k_sl * sigma
             
             if direction == 1:
                 pt_price = entry_price * (1 + d_pt)
                 sl_price = entry_price * (1 - d_sl)
                 hit_pt = future_window['high'] >= pt_price
                 hit_sl = future_window['low'] <= sl_price
             else:
                 pt_price = entry_price * (1 - d_pt)
                 sl_price = entry_price * (1 + d_sl)
                 hit_pt = future_window['low'] <= pt_price
                 hit_sl = future_window['high'] >= sl_price
                 
             # Store dynamic PT/SL for debugging/logic if needed across loop
             # (not needed for triple barrier outcome, just calc)
             
        else:
             # Static Fixed %
             if direction == 1:
                 pt_price = entry_price * (1 + pt)
                 sl_price = entry_price * (1 - sl)
                 hit_pt = future_window['high'] >= pt_price
                 hit_sl = future_window['low'] <= sl_price
             else:
                 pt_price = entry_price * (1 - pt)
                 sl_price = entry_price * (1 + sl)
                 hit_pt = future_window['low'] <= pt_price
                 hit_sl = future_window['high'] >= sl_price
                 

            
        first_pt = hit_pt.idxmax() if hit_pt.any() else None
        first_sl = hit_sl.idxmax() if hit_sl.any() else None
        
        if first_pt and (not first_sl or first_pt < first_sl):
            outcome = 1 # Good
            raw_ret = pt * 100
        elif first_sl and (not first_pt or first_sl < first_pt):
            outcome = 0 # Bad
            raw_ret = -sl * 100
        else:
            # Time Limit
            exit_price = future_window['close'].iloc[-1]
            if direction == 1:
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price
            
            raw_ret = ret * 100
            outcome = 1 if raw_ret > 0 else 0
            
        labels.append({
            'entry_time': start_time,
            'symbol': symbol,
            'signal_type': signal_type,
            'volatility': row['volatility'],
            'rsi': row['rsi'],
            'ma_dist': row['ma_dist'],
            'adx': row['adx'],
            'mom': row['mom'],
            'rvol': row.get('rvol', 0),
            'bb_width': row.get('bb_width', 0),
            'candle_ratio': row.get('candle_ratio', 0),
            'atr_pct': row.get('atr_pct', 0),
            'mfi': row.get('mfi', 50),
            'raw_ret': raw_ret,
            'label': outcome,
            'weight': sample_weight
        })
        
    return pd.DataFrame(labels)

def train_model(group_key, group_config):
    print(f"\n=== Training Model for {group_key.upper()} ===")
    all_signals = []
    
    timeframes = group_config['timeframes']
    strat_name = group_config['strategy']
    
    for symbol in config['assets']:
        for tf in timeframes:
            print(f"Processing {symbol} ({tf})...")
            try:
                # determine fetch code
                if '-' in symbol or '^' in symbol or '=' in symbol:
                    fetch_type = 'trad'
                else:
                    fetch_type = get_asset_type(symbol)
                
                # Fetch
                df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=config['lookback_candles'])
                if df.empty: continue
                
                # Apply Strategy
                if strat_name == "WizardWave":
                    strat = WizardWaveStrategy()
                else:
                    strat = WizardScalpStrategy() # Default Low TF settings
                    
                df = strat.apply(df)
                
                # Label
                labeled = apply_triple_barrier(df, symbol, tf, group_config)
                all_signals.append(labeled)
                
            except Exception as e:
                print(f"Error {symbol} {tf}: {e}")
                
    if not all_signals:
        print(f"No signals for {group_key}.")
        return

    full_dataset = pd.concat(all_signals, ignore_index=True)
    full_dataset.dropna(inplace=True)
    print(f"Total Signals: {len(full_dataset)}")
    
    if len(full_dataset) < 50:
        print("Not enough data.")
        return

    # Train
    features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
    target = 'label'
    
    X = full_dataset[features]
    y = full_dataset[target]
    w = full_dataset['weight']
    
    clf = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'], # Use updated depth
        random_state=42,
        class_weight='balanced'
    )
    
    clf.fit(X, y, sample_weight=w)
    
    model_file = group_config['model_file']
    joblib.dump(clf, model_file)
    print(f"Saved {model_file}")
    
    # Simple Eval
    preds = clf.predict(X)
    print(f"Training Accuracy: {accuracy_score(y, preds):.2f}")

def run_pipeline():
    # Train HTF
    train_model('htf', config['htf'])
    
    # Train LTF
    train_model('ltf', config['ltf'])

if __name__ == "__main__":
    run_pipeline()

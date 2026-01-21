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

def apply_triple_barrier(df, symbol, tf, model_config):
    """
    Applies Triple Barrier Method to label signals based on model config.
    """
    labels = []
    
    asset_type = get_asset_type(symbol)
    tb_config = model_config['triple_barrier']
    
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
    
    # Weight - simplified, 1.0 for single TF models
    sample_weight = 1.0
    
    # Calculate Volatility (Sigma) if needed for Dynamic Barrier
    if tb_config.get('crypto_use_dynamic', False):
         df['sigma'] = df['close'].pct_change().ewm(span=36, adjust=False).std()
         df['sigma'] = df['sigma'].fillna(method='bfill').fillna(0.01)
    
    # Time Limit Calculation
    if 'time_limit_bars' in tb_config:
        time_limit = tb_config['time_limit_bars']
    elif 'time_limit_days' in tb_config:
        # Fallback for legacy support if needed, though we moved to bars
        days = tb_config['time_limit_days']
        if tf == '1d': bars_per_day = 1
        elif tf == '4d': bars_per_day = 0.25
        elif tf == '1h': bars_per_day = 24 # Crude approx
        else: bars_per_day = 1 
        time_limit = int(days * bars_per_day)
    else:
        time_limit = 20 # Default
        
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
            
        item = {
            'entry_time': start_time,
            'symbol': symbol,
            'signal_type': signal_type,
            'raw_ret': raw_ret,
            'label': outcome,
            'weight': sample_weight
        }
        # Dynamic Feature Gathering
        for f in ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'month_sin', 'cycle_regime', 'close_frac']:
            if f in row: item[f] = row[f]
            else: item[f] = 0.0
            
        labels.append(item)
        
    return pd.DataFrame(labels)

def train_model(model_name, model_config):
    print(f"\n=== Training Model: {model_name} ===")
    all_signals = []
    
    timeframes = model_config['timeframes']
    strat_name = model_config['strategy']
    
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
                df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=config.get('lookback_candles', 300))
                if df.empty: continue
                
                # Apply Strategy
                if strat_name == "WizardWave":
                    strat = WizardWaveStrategy()
                else:
                    strat = WizardScalpStrategy(lookback=8) # Default Low TF
                    
                df = strat.apply(df)
                
                # Label
                labeled = apply_triple_barrier(df, symbol, tf, model_config)
                all_signals.append(labeled)
                
            except Exception as e:
                print(f"Error {symbol} {tf}: {e}")
                
    if not all_signals:
        print(f"No signals for {model_name}.")
        return

    full_dataset = pd.concat(all_signals, ignore_index=True)
    full_dataset.dropna(inplace=True)
    print(f"Total Signals: {len(full_dataset)}")
    
    if len(full_dataset) < 10: # Lowered threshold for specific TF training
        print(f"Not enough data ({len(full_dataset)}) to train {model_name}.")
        return

    full_dataset.sort_values('entry_time', inplace=True)
    full_dataset.dropna(inplace=True)
    
    # Train
    features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'month_sin', 'cycle_regime', 'close_frac']
    target = 'label'
    
    # Purged Split (OOS Validation)
    split_idx = int(len(full_dataset) * 0.8)
    train_data = full_dataset.iloc[:split_idx]
    test_data = full_dataset.iloc[split_idx:]
    
    X_train = train_data[features]
    y_train = train_data[target]
    w_train = train_data['weight']
    
    clf = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        random_state=42,
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train, sample_weight=w_train)
    
    model_file = model_config['model_file']
    joblib.dump(clf, model_file)
    print(f"Saved {model_file}")
    
    # Validation on Test Set (Out-of-Sample)
    test_preds = clf.predict(test_data[features])
    print(f"Validation Accuracy: {accuracy_score(test_data[target], test_preds):.2f}")

def run_pipeline():
    models = config.get('models', {})
    for name, conf in models.items():
        train_model(name, conf)

if __name__ == "__main__":
    run_pipeline()

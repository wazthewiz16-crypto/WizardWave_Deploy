import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
import itertools
import json

# Global Configuration for Data
ASSETS = ["^NDX", "^GSPC", "GC=F"] # NAS100, SPX500, Gold
TIMEFRAME = "1d"
LOOKBACK = 5000
TEST_SIZE = 300

# Grid Search Space
# Geared towards TradFi (Lower Volatility than Crypto)
GRID = {
    "pt_pct": [0.02, 0.03, 0.05],   # 2%, 3%, 5% (Adjusted down for Indices)
    "sl_pct": [0.01, 0.02, 0.04],   # 1%, 2%, 4%
    "time_limit_bars": [10, 20, 40] 
}

def run_optimization():
    print("--- Starting Grid Search Optimization (TradFi) ---")
    
    # 1. Fetch Data ONCE
    data_map = {}
    for symbol in ASSETS:
        print(f"Fetching {symbol}...")
        try:
            # Determine asset type based on symbol format
            asset_type = 'trad' if '-' in symbol or '^' in symbol or '=' in symbol else 'crypto'
            df = fetch_data(symbol, asset_type=asset_type, timeframe=TIMEFRAME, limit=LOOKBACK)
            
            if df.empty:
                print(f"No Data for {symbol}")
                continue

            strat = WizardWaveStrategy()
            df = strat.apply(df)
            data_map[symbol] = df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    if not data_map:
        print("No data fetched. Exiting.")
        return

    # 2. Iterate Grid
    keys = GRID.keys()
    values = GRID.values()
    combinations = list(itertools.product(*values))
    
    best_ret = -np.inf
    best_config = None
    
    print(f"Testing {len(combinations)} combinations...")
    
    for i, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))
        
        all_signals = []
        for symbol, df in data_map.items():
            labeled = local_apply_triple_barrier(df.copy(), symbol, current_params)
            all_signals.append(labeled)
            
        full_dataset = pd.concat(all_signals, ignore_index=True)
        if full_dataset.empty:
            continue
            
        full_dataset.dropna(inplace=True)
        full_dataset.sort_values('entry_time', inplace=True)
        
        if len(full_dataset) < TEST_SIZE + 50:
            continue
            
        # Split
        train_df = full_dataset.iloc[:-TEST_SIZE]
        test_df = full_dataset.iloc[-TEST_SIZE:]
        
        # Train
        features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom']
        X_train = train_df[features]
        y_train = train_df['label']
        X_test = test_df[features]
        
        if len(y_train.unique()) < 2:
            continue
            
        clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predict
        probs = clf.predict_proba(X_test)[:, 1]
        test_df = test_df.copy()
        test_df['model_prob'] = probs
        test_df['action'] = test_df['model_prob'] > 0.6
        test_df['filtered_ret'] = np.where(test_df['action'], test_df['raw_ret'], 0.0)
        
        total_ret = test_df['filtered_ret'].sum()
        trades_count = test_df['action'].sum()
        
        print(f"[{i+1}/{len(combinations)}] Params: {current_params} | Return: {total_ret:.2f}% | Trades: {trades_count}")
        
        if total_ret > best_ret:
            best_ret = total_ret
            best_config = current_params

    print("\n--- Optimization Complete ---")
    print(f"Best Return: {best_ret:.2f}%")
    print("Best Parameters:")
    print(json.dumps(best_config, indent=4))
    
    # Save Best to Config
    if best_config:
        final_config = {
            "assets": ASSETS,
            "timeframe": TIMEFRAME,
            "lookback_candles": LOOKBACK,
            "triple_barrier": best_config,
            "model": {
                "type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 5,
                "confidence_threshold": 0.6
            },
            # Adjust test size if needed, but 300 is fine
            "backtest": {"test_size": TEST_SIZE}
        }
        with open('strategy_config.json', 'w') as f:
            json.dump(final_config, f, indent=4)
        print("Updated strategy_config.json with best parameters.")

def local_apply_triple_barrier(df, symbol, params):
    labels = []
    pt = params['pt_pct']
    sl = params['sl_pct']
    time_limit = params['time_limit_bars']
    
    # Features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['sma50'] = ta.sma(df['close'], length=50)
    df['ma_dist'] = (df['close'] / df['sma50']) - 1
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['adx'] = adx_df['ADX_14'] if not adx_df.empty else 0
    df['mom'] = ta.roc(df['close'], length=10)

    signal_indices = df[df['signal_type'] != 'NONE'].index
    
    for start_time in signal_indices:
        row = df.loc[start_time]
        signal_type = row['signal_type']
        entry_price = row['close']
        
        if 'LONG' in signal_type:
            direction = 1
            pt_price = entry_price * (1 + pt)
            sl_price = entry_price * (1 - sl)
        else:
            direction = -1
            pt_price = entry_price * (1 - pt)
            sl_price = entry_price * (1 + sl)
            
        start_iloc = df.index.get_loc(start_time)
        end_iloc = min(start_iloc + time_limit, len(df))
        # Important: use iloc relative
        
        # Check if we have enough data? No, min handles it.
        # But for 'future_window', we want rows AFTER start_time
        # start_iloc+1 to end_iloc
        if (start_iloc + 1) >= len(df):
             continue

        future_window = df.iloc[start_iloc + 1 : end_iloc]
        
        if future_window.empty:
            continue
            
        outcome = 0 
        raw_ret = 0.0
        
        if direction == 1:
            hit_pt = future_window['high'] >= pt_price
            hit_sl = future_window['low'] <= sl_price
        else:
            hit_pt = future_window['low'] <= pt_price
            hit_sl = future_window['high'] >= sl_price
            
        first_pt = hit_pt.idxmax() if hit_pt.any() else None
        first_sl = hit_sl.idxmax() if hit_sl.any() else None
        
        if first_pt and (not first_sl or first_pt < first_sl):
            outcome = 1
            raw_ret = pt * 100
        elif first_sl and (not first_pt or first_sl < first_pt):
            outcome = 0
            raw_ret = -sl * 100
        else:
            # Time Limit
            exit_price = future_window['close'].iloc[-1]
            ret = (exit_price - entry_price) / entry_price if direction == 1 else (entry_price - exit_price) / entry_price
            raw_ret = ret * 100
            outcome = 0

        labels.append({
            'entry_time': start_time,
            'symbol': symbol,
            'signal_type': signal_type,
            'volatility': row['volatility'],
            'rsi': row['rsi'],
            'ma_dist': row['ma_dist'],
            'adx': row['adx'],
            'mom': row['mom'],
            'raw_ret': raw_ret,
            'label': outcome
        })
        
    return pd.DataFrame(labels)

if __name__ == "__main__":
    run_optimization()

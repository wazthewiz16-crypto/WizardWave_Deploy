import json
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy

# Load Configuration
with open('strategy_config.json', 'r') as f:
    config = json.load(f)

def run_pipeline():
    print("--- Starting Meta-Labeling Pipeline ---")
    
    # 1. Data Ingestion & Signal Generation
    # 1. Data Ingestion & Signal Generation
    all_signals = []
    
    # Support multiple timeframes if list, else single
    timeframes = config.get('timeframes', [config.get('timeframe', '1d')])
    
    for symbol in config['assets']:
        for tf in timeframes:
            print(f"Processing {symbol} ({tf})...")
            try:
                asset_type = 'trad' if '-' in symbol or '^' in symbol or '=' in symbol else 'crypto'
                
                # Fetch Data
                df = fetch_data(symbol, asset_type=asset_type, timeframe=tf, limit=config['lookback_candles'])
                if df.empty:
                    continue
                    
                # Generate Signals
                strat = WizardWaveStrategy()
                df = strat.apply(df)
                
                # Labeling
                labeled_data = apply_triple_barrier(df, symbol, tf)
                all_signals.append(labeled_data)
                
            except Exception as e:
                print(f"Error processing {symbol} {tf}: {e}")
            
    if not all_signals:
        print("No signals generated.")
        return

    # Combine all signals
    full_dataset = pd.concat(all_signals, ignore_index=True)
    print(f"Total Signals Collected: {len(full_dataset)}")
    
    # Drop NaNs created by feature engineering
    full_dataset.dropna(inplace=True)
    
    # 2. Train/Test Split
    # We save the last N signals for the final backtest simulation as requested
    test_size = config['backtest']['test_size']
    
    if len(full_dataset) < test_size + 50:
        print("Not enough data for split.")
        return

    # Sort by time to avoid look-ahead bias in split
    full_dataset.sort_values('entry_time', inplace=True)
    
    train_df = full_dataset.iloc[:-test_size]
    test_df = full_dataset.iloc[-test_size:]
    
    print(f"Training Set: {len(train_df)} | Test Set: {len(test_df)}")
    
    # 3. Model Training
    features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom'] 
    target = 'label'
    
    X_train = train_df[features]
    y_train = train_df[target]
    w_train = train_df['weight']
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    clf = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'], 
        max_depth=config['model']['max_depth'], 
        random_state=42
    )
    
    clf.fit(X_train, y_train, sample_weight=w_train)
    
    # Save Model
    joblib.dump(clf, 'model.pkl')
    print("Model saved to model.pkl")
    
    # Evaluate
    train_preds = clf.predict(X_train)
    print("Training Accuracy:", accuracy_score(y_train, train_preds))
    
    # 4. Backtest Simulation (The "Act" Phase)
    print("\n--- Running Backtest Simulation on Test Set ---")
    
    # Get Probabilities
    probs = clf.predict_proba(X_test)[:, 1] # Probability of class 1 (Good)
    
    # Prepare Backtest Data
    results = test_df.copy()
    results['model_prob'] = probs
    threshold = config['model']['confidence_threshold']
    
    # Logic: If prob > threshold, take trade.
    results['action'] = results['model_prob'] > threshold
    
    # Calculate Returns
    # raw_ret is the return of the trade regardless of filter
    # filtered_ret is raw_ret if action is True, else 0
    results['filtered_ret'] = np.where(results['action'], results['raw_ret'], 0.0)
    
    # Cumulative Returns
    results['cum_raw'] = results['raw_ret'].cumsum()
    results['cum_filtered'] = results['filtered_ret'].cumsum()
    
    print("Backtest Complete.")
    print(f"Raw Strategy Return: {results['cum_raw'].iloc[-1]:.2f}%")
    print(f"Filtered Strategy Return: {results['cum_filtered'].iloc[-1]:.2f}%")
    print(f"Trades Taken: {results['action'].sum()} / {len(results)}")
    
    # 5. Serialization & Plotting
    plot_equity_curve(results)
    
def apply_triple_barrier(df, symbol, tf):
    """
    Applies Triple Barrier Method to label signals.
    """
    labels = []
    
    # Barriers
    pt = config['triple_barrier']['pt_pct']
    sl = config['triple_barrier']['sl_pct']
    
    # Weight
    weights_map = config.get('timeframe_weights', {})
    sample_weight = weights_map.get(tf, 1.0)
    
    # Dynamic Time Limit Calculation
    time_limit_days = config['triple_barrier'].get('time_limit_days', 40)
    
    # Convert days to bars
    if tf == '1h':
        bars_per_day = 24
    elif tf == '4h':
        bars_per_day = 6
    elif tf == '1d':
        bars_per_day = 1
    elif tf == '4d':
        bars_per_day = 0.25 # 1 bar per 4 days
    else:
        bars_per_day = 1 # Default
        
    time_limit = int(time_limit_days * bars_per_day)
    time_limit = max(1, time_limit) # Ensure at least 1 bar
    
    # Features (Same as before)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # 2. RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # 3. MA Distance (Price / SMA50 - 1)
    df['sma50'] = ta.sma(df['close'], length=50)
    df['ma_dist'] = (df['close'] / df['sma50']) - 1
    
    # 4. ADX
    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
    if not adx_df.empty:
      df['adx'] = adx_df['ADX_14']
    else:
      df['adx'] = 0

    # 5. Momentum (ROC)
    df['mom'] = ta.roc(df['close'], length=10)

    # Filter for Signals
    # We iterate through the DF to respect time
    
    # Optimization: Get indices of signals
    # WizardWaveStrategy puts 'signal_type' != 'NONE'
    signal_indices = df[df['signal_type'] != 'NONE'].index
    
    for start_time in signal_indices:
        row = df.loc[start_time]
        signal_type = row['signal_type']
        entry_price = row['close']
        
        # Determine Direction
        if 'LONG' in signal_type:
            direction = 1
        else:
            direction = -1
            
        # Barrier Search
        # Slice future data up to time_limit
        # Note: df.loc[start_time:] includes start_time, so we take [1:] relative to that or utilize iloc
        
        # Use simple integer location for speed if possible, but index is datetime.
        # Let's find integer loc
        try:
            start_iloc = df.index.get_loc(start_time)
        except KeyError: # Duplicate index protection
             # If duplicate exists, handle or skip. Simple skip for now.
             continue

        end_iloc = min(start_iloc + time_limit, len(df))
        future_window = df.iloc[start_iloc + 1 : end_iloc]
        
        if future_window.empty:
            continue
            
        outcome = 0 # Default Bad
        raw_ret = 0.0
        
        # Check PT/SL
        # We need to check which happened first.
        # Vectorized check within the window
        
        # Calculate returns relative to entry
        # If LONG: (Price - Entry) / Entry
        # If SHORT: (Entry - Price) / Entry
        
        if direction == 1:
            # LONG
            # Highs for PT, Lows for SL
            # Find first time High >= Entry * (1+PT)
            pt_price = entry_price * (1 + pt)
            sl_price = entry_price * (1 - sl)
            
            # Boolean series
            hit_pt = future_window['high'] >= pt_price
            hit_sl = future_window['low'] <= sl_price
            
        else:
            # SHORT
            # Lows for PT, Highs for SL
            pt_price = entry_price * (1 - pt)
            sl_price = entry_price * (1 + sl)
            
            hit_pt = future_window['low'] <= pt_price
            hit_sl = future_window['high'] >= sl_price
            
        # Get first occurrence
        first_pt = hit_pt.idxmax() if hit_pt.any() else None
        first_sl = hit_sl.idxmax() if hit_sl.any() else None
        
        # Determine Label
        if first_pt and (not first_sl or first_pt < first_sl):
            outcome = 1 # Good
            raw_ret = pt * 100 # Approx return (taking fixed PT)
            # Or use exact close of that bar? Standard is to take the PT amount.
        elif first_sl and (not first_pt or first_sl < first_pt):
            outcome = 0 # Bad
            raw_ret = -sl * 100 # Took SL
        else:
            # Time Limit Reached
            # Close position at the end of window
            exit_price = future_window['close'].iloc[-1]
            if direction == 1:
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price
            
            raw_ret = ret * 100
            
            # Labeling definition: "Good (1)" usually means we hit PT. 
            # If we timed out with profit, is it Good?
            # User said: "Barrier 1: Profit Take... Barrier 2: Stop Loss... Barrier 3: Time Limit"
            # "Label the historic signals as 'Good' (1) or 'Bad' (0)"
            # Strict definition: Only PT is 1. Everything else is 0. 
            # Or if Return > 0 at time limit?
            # I will stick to: 1 if PT hit. 0 else. (High precision approach)
            outcome = 1 if hit_pt.any() and (not hit_sl.any() or hit_pt.idxmax() < hit_sl.idxmax()) else 0
            
            # Actually, let's allow "Good" if TimeLimit and Profit > 0? 
            # No, standard Triple Barrier usually implies targeting the PT.
            # I will stick to the user's implicit "filter for profit".
            
        # Store Data
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
            'label': outcome,
            'weight': sample_weight
        })
        
    return pd.DataFrame(labels)

def plot_equity_curve(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results['cum_raw'].values, label='Raw Strategy')
    plt.plot(results['cum_filtered'].values, label='Meta-Labeled (Filtered)', linewidth=2)
    plt.title('Equity Curve: Raw vs Meta-Labeled')
    plt.xlabel('Trade Count')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_result.png')
    print("Equity curve saved to backtest_result.png")

if __name__ == "__main__":
    run_pipeline()

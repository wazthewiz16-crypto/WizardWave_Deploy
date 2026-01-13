
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import precision_score, classification_report
import joblib
import pandas_ta as ta
import warnings
from strategy import WizardWaveStrategy
from feature_engine import calculate_ml_features

warnings.simplefilter(action='ignore', category=FutureWarning)

def fetch_history(symbol, period='1y', interval='1d'):
    """Fetch history using yfinance."""
    clean_sym = symbol
    if "USDT" in symbol: clean_sym = symbol.replace("/", "-").replace("USDT", "USD")
    elif "USD" in symbol and "=" not in symbol: clean_sym = symbol.replace("/", "")
    
    try:
        df = yf.Ticker(clean_sym).history(period=period, interval=interval)
        if df.empty: return df
        if period == '1mo' and interval=='1h': # Try fetch more if 1h
             df = yf.Ticker(clean_sym).history(period='1y', interval=interval)

        df = df.reset_index()
        df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        # Clean TZ
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_convert(None)
            df.set_index('datetime', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except: return pd.DataFrame()

def prepare_data(assets, interval='1d', lookback_period='1y'):
    all_data = []
    
    strat = WizardWaveStrategy()
    
    print(f"Fetching data for {len(assets)} assets ({interval})...")
    
    for symbol in assets:
        df = fetch_history(symbol, period=lookback_period, interval=interval)
        if df.empty or len(df) < 100: continue
        
        # Strategy
        df = strat.apply(df)
        
        # Features (Updated with Regime & Cloud Dist)
        df = calculate_ml_features(df)
        
        # TARGET GENERATION (3-Barrier Method Simplified)
        # Look forward 12 bars (approx)
        # If hits TP (2x ATR) before SL (1x ATR) => 1
        # Else => 0
        df['atr_target'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        future_window = 12
        df['target'] = 0
        
        for i in range(len(df) - future_window):
            row = df.iloc[i]
            signal = row['signal_type']
            
            if signal == 'NONE': continue
            
            entry_price = row['close']
            atr = row['atr_target']
            
            # Dynamic Target
            tp_dist = 2.0 * atr 
            sl_dist = 1.0 * atr
            
            # Label
            label = 0
            
            if 'LONG' in signal:
                tp_price = entry_price + tp_dist
                sl_price = entry_price - sl_dist
                
                # Scan future
                for j in range(1, future_window + 1):
                    fut = df.iloc[i+j]
                    if fut['low'] <= sl_price: 
                        label = 0
                        break
                    if fut['high'] >= tp_price:
                        label = 1
                        break
            elif 'SHORT' in signal:
                tp_price = entry_price - tp_dist
                sl_price = entry_price + sl_dist
                
                for j in range(1, future_window + 1):
                    fut = df.iloc[i+j]
                    if fut['high'] >= sl_price:
                        label = 0
                        break
                    if fut['low'] <= tp_price:
                        label = 1
                        break
            
            df.at[df.index[i], 'target'] = label
            
        # Add to stack
        # Filter only rows with Signal != NONE for training
        df_sigs = df[df['signal_type'] != 'NONE'].copy()
        all_data.append(df_sigs)
        
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data)

def tune_hyperparameters(metrics_df, interval_name):
    # Features List (Updated)
    features = [
        'volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 
        'bb_width', 'candle_ratio', 'atr_pct', 'mfi',
        'mango_d1_dist', 'mango_d2_dist', 'upper_zone_dist', 'lower_zone_dist',
        'cloud_top_dist', 'cloud_bot_dist', 'regime_bull'
    ]
    
    # Clean
    metrics_df = metrics_df.dropna(subset=features + ['target'])
    
    X = metrics_df[features]
    y = metrics_df['target']
    
    print(f"Training Data Shape for {interval_name}: {X.shape}")
    print(f"Target Distribution: {y.value_counts(normalize=True)}")
    
    # TimeSeries Split for CV
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Parameter Grid
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Random Search (Faster than Grid)
    search = RandomizedSearchCV(
        rf, 
        param_grid, 
        n_iter=20, # Try 20 combos
        cv=tscv, 
        scoring='precision', # Prioritize Precision (Win Rate)
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    search.fit(X, y)
    
    print(f"\nBest Params for {interval_name}: {search.best_params_}")
    print(f"Best CV Precision: {search.best_score_:.4f}")
    
    # Train stats on full set with best params
    best_model = search.best_estimator_
    y_pred = best_model.predict(X)
    print(classification_report(y, y_pred))
    
    return best_model, search.best_params_

if __name__ == "__main__":
    import json
    with open('strategy_config.json', 'r') as f:
        CONF = json.load(f)
    assets = CONF['assets']
    
    # 1. Tune 1D (Swing)
    print("\n=== TUNING 1D MODEL ===")
    df_1d = prepare_data(assets, '1d', '2y') # Use 2y for daily to get enough samples
    if not df_1d.empty:
        model_1d, params_1d = tune_hyperparameters(df_1d, '1d')
        joblib.dump(model_1d, 'model_1d_tuned.pkl')
        # Save params to json?
    
    # 2. Tune 1H (Scalp)
    print("\n=== TUNING 1H MODEL ===")
    df_1h = prepare_data(assets, '1h', '6mo') # Use 6mo for hourly (speed + relevance)
    if not df_1h.empty:
         model_1h, params_1h = tune_hyperparameters(df_1h, '1h')
         joblib.dump(model_1h, 'model_1h_tuned.pkl')
         
    print("Done. Tuned models saved.")

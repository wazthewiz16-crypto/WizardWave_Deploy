
import sys
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier

# Add Project Root to Path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)

try:
    from strategy import WizardWaveStrategy
    from strategy_scalp import WizardScalpStrategy
    from feature_engine import calculate_ml_features
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- SETTINGS ---
WINDOW_MONTHS = 24 # 2 Year sliding window
CONFIG_FILE = os.path.join(project_root, 'strategy_config.json')

def load_config():
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def fetch_rolling_data(symbol, interval='1d'):
    """Fetch the rolling window of data."""
    end_date = datetime.now()
    if interval == '1h':
        # YFinance limit for 1h is approx 730 days
        start_date = end_date - timedelta(days=720)
    else:
        start_date = end_date - relativedelta(months=WINDOW_MONTHS)
    
    # Handle Symbol mapping
    clean_sym = symbol
    if "USDT" in symbol: clean_sym = symbol.replace("/", "-").replace("USDT", "USD")
    elif "USD" in symbol and "=" not in symbol: clean_sym = symbol.replace("/", "")
    
    print(f"  Fetching {clean_sym} ({interval})...")
    try:
        df = yf.Ticker(clean_sym).history(start=start_date, end=end_date, interval=interval)
        if df.empty: 
            print(f"    Warning: No data for {clean_sym}")
            return pd.DataFrame()
        
        df = df.reset_index()
        df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df.set_index('datetime', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return pd.DataFrame()

def prepare_dataset(assets, tf='1d'):
    all_rows = []
    
    # Select Strategy
    if tf == '1d': 
        strat = WizardWaveStrategy()
        tp_tr = 0.09 # Production TP for 1D
        sl_tr = 0.03 # Production SL for 1D
        bars = 14
    else: # 1h
        strat = WizardScalpStrategy(lookback=8)
        tp_tr = 0.02 # Production TP for 1H
        sl_tr = 0.01 # Production SL for 1H
        bars = 24

    print(f"Preparing dataset for {tf}...")
    
    for symbol in assets:
        df = fetch_rolling_data(symbol, interval=tf)
        if df.empty or len(df) < 100: continue
        
        # 1. Strategy
        df = strat.apply(df)
        
        # 2. Features
        df = calculate_ml_features(df)
        
        # 3. Targets (Triple Barrier simplified)
        df['target'] = 0
        indexer = df.index
        for i in range(len(df) - bars):
            curr = df.iloc[i]['close']
            sig = df.iloc[i]['signal_type']
            if sig == 'NONE': continue
            
            # Future Window
            future = df.iloc[i+1 : i+bars+1]
            if 'LONG' in sig:
                # Did it hit TP (9%) before SL (3%)?
                # Optimization: find index of first hit
                tp_price = curr * (1 + tp_tr)
                sl_price = curr * (1 - sl_tr)
                
                hit_tp = future[future['high'] >= tp_price].head(1)
                hit_sl = future[future['low'] <= sl_price].head(1)
                
                if not hit_tp.empty:
                    if hit_sl.empty or hit_tp.index[0] < hit_sl.index[0]:
                        df.at[indexer[i], 'target'] = 1
            else: # SHORT
                tp_price = curr * (1 - tp_tr)
                sl_price = curr * (1 + sl_tr)
                hit_tp = future[future['low'] <= tp_price].head(1)
                hit_sl = future[future['high'] >= sl_price].head(1)
                
                if not hit_tp.empty:
                    if hit_sl.empty or hit_tp.index[0] < hit_sl.index[0]:
                        df.at[indexer[i], 'target'] = 1
                        
        sigs = df[df['signal_type'] != 'NONE'].copy()
        all_rows.append(sigs)
        
    if not all_rows: return pd.DataFrame()
    return pd.concat(all_rows)

def retrain():
    config = load_config()
    assets = config['assets']
    
    # 1. Retrain 1D
    df_1d = prepare_dataset(assets, '1d')
    if not df_1d.empty:
        feats = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
        X = df_1d[feats]
        y = df_1d['target']
        
        print(f"Training 1D Model on {len(X)} signals...")
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)
        
        save_path = os.path.join(project_root, 'model_1d.pkl')
        joblib.dump(model, save_path)
        print(f"Successfully updated {save_path}")

    # 2. Retrain 1H
    df_1h = prepare_dataset(assets, '1h')
    if not df_1h.empty:
        feats = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
        X = df_1h[feats]
        y = df_1h['target']
        
        print(f"Training 1H Model on {len(X)} signals...")
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)
        
        save_path = os.path.join(project_root, 'model_1h.pkl')
        joblib.dump(model, save_path)
        print(f"Successfully updated {save_path}")

if __name__ == "__main__":
    retrain()

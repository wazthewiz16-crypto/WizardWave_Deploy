import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta

# Project Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features

# Load Thresholds from config
with open('strategy_config.json', 'r') as f:
    PROD_CONFIG = json.load(f)

ASSETS = ['BTC-USD', 'ETH-USD', 'GC=F', 'EURUSD=X', '^NDX']

# Model Pairs (Old vs New)
MODELS = {
    'Wizard Wave (1D)': {
        'old_file': 'model_1d.pkl',
        'new_file': 'wizard_wave_ml_model.pkl',
        'interval': '1d',
        'strategy': WizardWaveStrategy(),
        'old_features': ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'rvol', 'month_sin', 'month_cos', 'cycle_regime'],
        'new_features': ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'month_sin', 'cycle_regime', 'close_frac']
    },
    'Wizard Scalp (1H)': {
        'old_file': 'model_1h.pkl',
        'new_file': 'wizard_scalp_ml_model.pkl',
        'interval': '1h',
        'strategy': WizardScalpStrategy(),
        'old_features': ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'rvol'], 
        'new_features': ['volatility', 'rsi', 'ma_dist', 'adx', 'rvol', 'atr_pct', 'cycle_regime']
    }
}

def get_config_val(label, key, default):
    tf = '1d' if '1D' in label else '1h'
    return PROD_CONFIG['models'].get(tf, {}).get(key, default)

def fetch_data(symbol, interval):
    # Try local cache first
    cache_map = {
        'BTC-USD': 'BTC_USDT',
        'ETH-USD': 'ETH_USDT',
        'GC=F': 'GCF',
        'EURUSD=X': 'EURUSDX',
        '^NDX': 'NDX'
    }
    cache_name = cache_map.get(symbol, symbol.replace('-', '_'))
    cache_path = f"market_data_cache/{cache_name}_{interval}.csv"
    
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        # Handle 'datetime' or 'Date' from cache
        date_col = 'datetime' if 'datetime' in df.columns else 'Date'
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        # Ensure col names are lower
        df.columns = [c.lower() for c in df.columns]
        return df[['open', 'high', 'low', 'close', 'volume']]
        
    print(f"  Fetching {symbol} from YF...")
    df = yf.Ticker(symbol).history(period="180d", interval=interval)
    if df.empty: return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    return df[['open', 'high', 'low', 'close', 'volume']]

def simulate(df, model, features, threshold, exposure=0.3):
    if df.empty: return 0, 0, 0
    
    # 2. ML Filter
    sig_mask = df['signal_type'] != 'NONE'
    if sig_mask.any():
        X = df.loc[sig_mask, features].fillna(0).values
        # Handle n_features mismatch
        if X.shape[1] != model.n_features_in_: return -999, 0, 0 
            
        probs = model.predict_proba(X)[:, 1]
        prob_idx = 0
        for idx, row in df[sig_mask].iterrows():
            if probs[prob_idx] < threshold:
                df.at[idx, 'signal_type'] = "NONE"
            prob_idx += 1

    # 3. Micro Simulation (Simplified)
    equity = 100000
    peak = equity
    max_dd = 0
    trades = 0
    
    position = 0
    entry = 0
    
    for idx, row in df.iterrows():
        sig = row['signal_type']
        price = row['close']
        sl = row.get('stop_loss', 0)
        tp = row.get('target_price', 0)
        
        # Exit
        if position != 0:
            closed = False
            pnl = 0
            if position == 1:
                if sl > 0 and row['low'] <= sl:
                    pnl = (sl - entry) * (equity*exposure/entry)
                    closed = True
                elif tp > 0 and row['high'] >= tp:
                    pnl = (tp - entry) * (equity*exposure/entry)
                    closed = True
                elif 'SHORT' in sig:
                    pnl = (price - entry) * (equity*exposure/entry)
                    closed = True
            elif position == -1:
                if sl > 0 and row['high'] >= sl:
                    pnl = (entry - sl) * (equity*exposure/entry)
                    closed = True
                elif tp > 0 and row['low'] <= tp:
                    pnl = (entry - tp) * (equity*exposure/entry)
                    closed = True
                elif 'LONG' in sig:
                    pnl = (entry - price) * (equity*exposure/entry)
                    closed = True
            
            if closed:
                equity += pnl
                position = 0
                trades += 1
                
        # Entry
        if position == 0 and 'NONE' not in sig:
            position = 1 if 'LONG' in sig else -1
            entry = price
            
        # DD
        if equity > peak: peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd: max_dd = dd
        
    pnl_pct = (equity - 100000) / 100000 * 100
    return pnl_pct, max_dd * 100, trades

def run_comparison():
    report = []
    
    for label, cfg in MODELS.items():
        print(f"\n--- Comparing {label} ---")
        
        # Load Models
        try:
            old_model = joblib.load(cfg['old_file'])
            if isinstance(old_model, dict): old_model = old_model['model']
            
            new_model = joblib.load(cfg['new_file'])
            if isinstance(new_model, dict): new_model = new_model['model']
        except Exception as e:
            print(f"Error loading models for {label}: {e}")
            continue
            
        for symbol in ASSETS:
            print(f"  Testing {symbol}...", end="\r")
            raw_df = fetch_data(symbol, cfg['interval'])
            if raw_df.empty: continue
            
            # Apply Strategy Logic (Shared)
            strat_df = cfg['strategy'].apply(raw_df.copy())
            feat_df = calculate_ml_features(strat_df)
            
            # Thresholds
            old_thresh = 0.45
            new_thresh = get_config_val(label, 'confidence_threshold', 0.45)
            
            # Old Run
            pnl_old, dd_old, t_old = simulate(feat_df.copy(), old_model, cfg['old_features'], old_thresh)
            
            # New Run
            pnl_new, dd_new, t_new = simulate(feat_df.copy(), new_model, cfg['new_features'], new_thresh)
            
            report.append({
                'Strategy': label,
                'Asset': symbol,
                'Old PnL%': f"{pnl_old:.1f}%" if pnl_old != -999 else "ERR",
                'New PnL%': f"{pnl_new:.1f}%",
                'Old DD%': f"{dd_old:.1f}%" if pnl_old != -999 else "ERR",
                'New DD%': f"{dd_new:.1f}%",
                'Old Trades': t_old if pnl_old != -999 else 0,
                'New Trades': t_new
            })

    df_report = pd.DataFrame(report)
    print("\n\n" + "="*80)
    print("                      MODEL PERFORMANCE COMPARISON (180 DAYS)")
    print("="*80)
    print(df_report.to_string(index=False))
    print("="*80)
    print("Note: 'ERR' usually means feature count mismatch for the older model.")

if __name__ == "__main__":
    run_comparison()

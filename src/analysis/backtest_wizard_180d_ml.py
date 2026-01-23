
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import json
import joblib
import os
import warnings
import sys
# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.strategies.strategy import WizardWaveStrategy
from src.strategies.strategy_scalp import WizardScalpStrategy
from src.core.feature_engine import calculate_ml_features
from src.utils.paths import get_config_path, get_model_path
from datetime import datetime, timedelta

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

def fetch_history_yf(symbol, period='1y', interval='1d'):
    """Fetch history using yfinance for backtesting."""
    # Mapping
    clean_sym = symbol
    if "USDT" in symbol:
        clean_sym = symbol.replace("/USDT", "-USD")
    elif "BTC/USD" in symbol:
        clean_sym = "BTC-USD"
    elif "ETH/USD" in symbol:
        clean_sym = "ETH-USD"
    
    # Handle the =X and =F
    # Usually passed correctly, but just in case
    
    try:
        df = yf.Ticker(clean_sym).history(period=period, interval=interval)
        if df.empty:
            return df
        
        # Standardize
        df = df.reset_index()
        df.rename(columns={
            'Date': 'datetime', 
            'Datetime': 'datetime',
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Volume': 'volume'
        }, inplace=True)
        
        # TZ naive
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_convert(None)
            df.set_index('datetime', inplace=True)
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        # print(f"Error fetching {clean_sym}: {e}")
        return pd.DataFrame()

def run_backtest_ml(df, strategy, model, asset_class, timeframe_label='1D', threshold=0.5, tb_config={}):
    """Run simulation on the dataframe WITH ML FILTERS."""
    
    # 1. Apply Strategy
    df = strategy.apply(df)
    
    # 2. Calculate Features
    df = calculate_ml_features(df)
    
    # 3. Dynamic Sigma for Crypto if enabled
    use_dynamic = tb_config.get('crypto_use_dynamic', False) and (asset_class == 'Crypto')
    if use_dynamic:
         df['sigma'] = df['close'].pct_change().ewm(span=36, adjust=False).std()
         df['sigma'] = df['sigma'].bfill().fillna(0.01)

    # Predict
    # Vectorized predict is faster.
    # Align features (Must match model training features)
    # Note: 'rvol' was missing in retrain_models.py, so we exclude it here to match current model.
    feat_cols = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
    
    # Ensure cols exist
    for c in feat_cols:
        if c not in df.columns: df[c] = 0.0
            
    df_clean = df.dropna(subset=feat_cols)
    if df_clean.empty:
        return []

    # Batch Predict
    try:
        if model:
            probs = model.predict_proba(df_clean[feat_cols])[:, 1]
            df.loc[df_clean.index, 'prob'] = probs
        else:
            df['prob'] = 1.0 # No model, assume PASS
    except:
        df['prob'] = 0.0

    df['prob'] = df['prob'].fillna(0.0)

    trades = []
    
    position = None
    entry_price = 0.0
    entry_time = None
    sl_price = 0.0
    tp_price = 0.0
    
    # --- PARAMS FROM CONFIG ---
    if asset_class == 'Crypto':
        base_tp = tb_config.get('crypto_pt', 0.04)
        base_sl = tb_config.get('crypto_sl', 0.02)
    elif asset_class == 'Forex':
        base_tp = tb_config.get('forex_pt', 0.01)
        base_sl = tb_config.get('forex_sl', 0.005)
    else: # TradFi
        base_tp = tb_config.get('trad_pt', 0.02)
        base_sl = tb_config.get('trad_sl', 0.01)
        
    for index, row in df.iterrows():
        close = row['close']
        cloud_top = row.get('cloud_top', 0)
        cloud_bottom = row.get('cloud_bottom', 0)
        signal = row.get('signal_type', 'NONE')
        prob = row.get('prob', 0.0)
        
        # Exit Logic
        exit_trade = False
        pnl = 0.0
        exit_reason = ""
        
        if position == 'LONG':
            if close <= sl_price:
                exit_trade = True
                pnl = (sl_price - entry_price) / entry_price
                exit_reason = "SL"
            elif close >= tp_price:
                exit_trade = True
                pnl = (tp_price - entry_price) / entry_price
                exit_reason = "TP"
            elif close < cloud_bottom:
                exit_trade = True
                pnl = (close - entry_price) / entry_price
                exit_reason = "Trend"
                
        elif position == 'SHORT':
            if close >= sl_price:
                exit_trade = True
                pnl = (entry_price - sl_price) / entry_price
                exit_reason = "SL"
            elif close <= tp_price:
                exit_trade = True
                pnl = (entry_price - tp_price) / entry_price
                exit_reason = "TP"
            elif close > cloud_top:
                exit_trade = True
                pnl = (entry_price - close) / entry_price
                exit_reason = "Trend"

        if exit_trade:
            trades.append({
                'Asset': 'Test',
                'Type': position,
                'Entry Time': entry_time,
                'Exit Time': index,
                'Entry Price': entry_price,
                'Exit Price': tp_price if exit_reason == "TP" else (sl_price if exit_reason == "SL" else close),
                'PnL': pnl,
                'Reason': exit_reason,
                'Conf': df.loc[entry_time, 'prob'] if entry_time in df.index else 0
            })
            position = None
            
        # Entry Logic (If no position)
        if position is None and prob >= threshold:
            take_signal = False
            new_pos = None
            
            if 'LONG' in signal: # Catch LONG_ZONE, SCALP_LONG etc
                new_pos = 'LONG'
                take_signal = True
            elif 'SHORT' in signal:
                new_pos = 'SHORT'
                take_signal = True
                
            if take_signal:
                position = new_pos
                entry_price = close
                entry_time = index
                
                # Dynamic Logic
                curr_tp = base_tp
                curr_sl = base_sl
                
                if use_dynamic:
                    sigma = row.get('sigma', 0.01)
                    k_pt = tb_config.get('crypto_dyn_pt_k', 3.0)
                    k_sl = tb_config.get('crypto_dyn_sl_k', 1.5)
                    curr_tp = k_pt * sigma
                    curr_sl = k_sl * sigma
                    
                if position == 'LONG':
                    sl_price = entry_price * (1 - curr_sl)
                    tp_price = entry_price * (1 + curr_tp)
                else:
                    sl_price = entry_price * (1 + curr_sl)
                    tp_price = entry_price * (1 - curr_tp)

        # Reversal Logic (Flip with Confidence Check)
        elif position is not None and prob >= threshold:
            # Check for Flip
             if position == 'LONG' and 'SHORT' in signal:
                # Close Long
                trades.append({
                    'Asset': 'Test',
                    'Type': 'LONG',
                    'Entry Time': entry_time,
                    'Exit Time': index,
                    'Entry Price': entry_price,
                    'Exit Price': close,
                    'PnL': (close - entry_price) / entry_price,
                    'Reason': 'Flip',
                    'Conf': prob
                })
                # Open Short
                position = 'SHORT'
                entry_price = close
                entry_time = index
                # Recalc dyn
                curr_tp = base_tp
                curr_sl = base_sl
                if use_dynamic:
                    sigma = row.get('sigma', 0.01)
                    k_pt = tb_config.get('crypto_dyn_pt_k', 3.0)
                    k_sl = tb_config.get('crypto_dyn_sl_k', 1.5)
                    curr_tp = k_pt * sigma
                    curr_sl = k_sl * sigma
                    
                sl_price = entry_price * (1 + curr_sl)
                tp_price = entry_price * (1 - curr_tp)
        
             elif position == 'SHORT' and 'LONG' in signal:
                # Close Short
                trades.append({
                    'Asset': 'Test',
                    'Type': 'SHORT',
                    'Entry Time': entry_time,
                    'Exit Time': index,
                    'Entry Price': entry_price,
                    'Exit Price': close,
                    'PnL': (entry_price - close) / entry_price,
                    'Reason': 'Flip',
                    'Conf': prob
                })
                # Open Long
                position = 'LONG'
                entry_price = close
                entry_time = index
                # Recalc dyn
                curr_tp = base_tp
                curr_sl = base_sl
                if use_dynamic:
                    sigma = row.get('sigma', 0.01)
                    k_pt = tb_config.get('crypto_dyn_pt_k', 3.0)
                    k_sl = tb_config.get('crypto_dyn_sl_k', 1.5)
                    curr_tp = k_pt * sigma
                    curr_sl = k_sl * sigma

                sl_price = entry_price * (1 - curr_sl)
                tp_price = entry_price * (1 + curr_tp)
            
    return trades

def print_bucket_stats(label, trades):
    if not trades:
        return

    df_t = pd.DataFrame(trades)
    total_trades = len(df_t)
    wins = len(df_t[df_t['PnL'] > 0])
    win_rate = wins / total_trades * 100
    total_pnl = df_t['PnL'].sum() * 100
    avg_pnl = df_t['PnL'].mean() * 100
    
    print(f"--- {label} ---")
    print(f"Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f}%")
    print(f"Avg Trade: {avg_pnl:.2f}%")
    # print("-" * 20)

if __name__ == "__main__":
    print("BACKTESTING FULL WIZARD WAVE (Last 180 Days) - WITH ML MODELS...")
    
    # Load Config
    with open(get_config_path('strategy_config.json'), 'r') as f:
        CONFIG = json.load(f)
        
    assets = CONFIG['assets']
    models_cfg = CONFIG['models']
    
    # Load Models
    model_1d = None
    path_1d = get_model_path("model_1d.pkl")
    if os.path.exists(path_1d): model_1d = joblib.load(path_1d)
    
    model_1h = None
    path_1h = get_model_path("model_1h.pkl")
    if os.path.exists(path_1h): model_1h = joblib.load(path_1h)
    
    # Accumulators
    results_crypto_1d = []
    results_trad_1d = []
    results_forex_1d = []
    
    results_crypto_1h = []
    results_trad_1h = []
    results_forex_1h = []
    
    # Limit
    limit_assets = 30 # Cap to avoid 5 minute run if many assets
    
    for i, symbol in enumerate(assets):
        if i >= limit_assets: break
        
        # Determine Class
        aclass = 'Crypto'
        if "USD=X" in symbol or "=X" in symbol: aclass = 'Forex'
        elif "^" in symbol or "=" in symbol or "-" in symbol and "BTC" not in symbol and "ETH" not in symbol: aclass = 'TradFi'
        # Fix for TSLA/NVDA if in list as per user config?
        # User list has ^NDX, ^GSPC etc.
        
        print(f"Processing {symbol} ({aclass})...")
        
        # --- 1D BACKTEST ---
        cfg_1d = models_cfg['1d']
        strat_1d = WizardWaveStrategy() # Default 1D
        
        df_d = fetch_history_yf(symbol, period='1y', interval='1d')
        if not df_d.empty:
            trades = run_backtest_ml(
                df_d, strat_1d, model_1d, aclass, '1D', 
                threshold=cfg_1d.get('confidence_threshold', 0.45),
                tb_config=cfg_1d.get('triple_barrier', {})
            )
            if aclass == 'Crypto': results_crypto_1d.extend(trades)
            elif aclass == 'Forex': results_forex_1d.extend(trades)
            else: results_trad_1d.extend(trades)

        # --- 1H BACKTEST ---
        cfg_1h = models_cfg['1h']
        strat_1h = WizardScalpStrategy() # 1H uses Scalp
        
        df_h = fetch_history_yf(symbol, period='6mo', interval='1h')
        if not df_h.empty:
            trades = run_backtest_ml(
                df_h, strat_1h, model_1h, aclass, '1H', 
                threshold=cfg_1h.get('confidence_threshold', 0.60),
                tb_config=cfg_1h.get('triple_barrier', {})
            )
            if aclass == 'Crypto': results_crypto_1h.extend(trades)
            elif aclass == 'Forex': results_forex_1h.extend(trades)
            else: results_trad_1h.extend(trades)

    print("\n" + "="*40)
    print("AGGREGATE RESULTS (180 DAYS w/ ML FILTER)")
    print("="*40)
    
    print_bucket_stats("CRYPTO SWING (1D)", results_crypto_1d)
    print_bucket_stats("CRYPTO SCALP (1H)", results_crypto_1h)
    print("-" * 20)
    print_bucket_stats("TRADFI SWING (1D)", results_trad_1d)
    print_bucket_stats("TRADFI SCALP (1H)", results_trad_1h)
    print("-" * 20)
    print_bucket_stats("FOREX SWING (1D)", results_forex_1d)
    print_bucket_stats("FOREX SCALP (1H)", results_forex_1h)
    print("="*40)

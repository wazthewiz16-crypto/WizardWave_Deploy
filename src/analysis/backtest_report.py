import json
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import os

from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features
from pipeline import get_asset_type, apply_triple_barrier

def run_backtest():
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
        
    models = {}
    models_conf = config['models']
    
    # Load Models
    for name, conf in models_conf.items():
        if os.path.exists(conf['model_file']):
            try:
                models[name] = joblib.load(conf['model_file'])
                print(f"Loaded {name}")
            except:
                print(f"Failed to load {name}")
    
    if not models:
        print("No models loaded.")
        return

    results = []

    # Features
    features_list = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']

    # Fetch Data with larger limit
    LIMIT = 3000

    for model_name, model_conf in models_conf.items():
        if model_name not in models: continue
        
        model = models[model_name]
        timeframes = model_conf['timeframes']
        strat_name = model_conf['strategy']
        
        for tf in timeframes:
            for symbol in config['assets']:
                try:
                    asset_type = get_asset_type(symbol)
                    fetch_type = 'trad' if asset_type == 'forex' or asset_type == 'trad' else 'crypto'
                    if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
                    
                    df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=LIMIT)
                    if df.empty: continue
                    
                    if strat_name == "WizardWave":
                        strat = WizardWaveStrategy()
                    else:
                        strat = WizardScalpStrategy(lookback=8)
                        
                    df = strat.apply(df)
                    df = calculate_ml_features(df)
                    
                    # Apply Triple Barrier (Labels for training, but here we simulate)
                    # Actually we just want to Predict and Simulate using the same logic.
                    
                    # Calculate Sigma
                    tb = model_conf['triple_barrier']
                    if tb.get('crypto_use_dynamic', False) and asset_type == 'crypto':
                         df['sigma'] = df['close'].pct_change().ewm(span=36, adjust=False).std()
                         df['sigma'] = df['sigma'].fillna(method='bfill').fillna(0.01)
                    
                    df = df.dropna()
                    if df.empty: continue
                    
                    # Predict all rows
                    X = df[features_list]
                    probs = model.predict_proba(X)[:, 1]
                    
                    df['prob'] = probs
                    
                    # Simulate
                    # Filter for signals
                    threshold = model_conf.get('confidence_threshold', 0.5)
                    
                    for idx, row in df.iterrows():
                        signal_type = row['signal_type']
                        if signal_type == 'NONE': continue
                        
                        prob = row['prob']
                        if prob < threshold: continue
                        
                        entry_time = idx
                        entry_price = row['close']
                        
                        # TP/SL Calc
                        if 'LONG' in signal_type:
                            direction = 1
                            if tb.get('crypto_use_dynamic', False) and asset_type == 'crypto':
                                  sigma = row.get('sigma', 0.01)
                                  k_pt = tb.get('crypto_dyn_pt_k', 0.5)
                                  k_sl = tb.get('crypto_dyn_sl_k', 0.5)
                                  pt_pct = k_pt * sigma
                                  sl_pct = k_sl * sigma
                            else:
                                 sl_pct = tb['crypto_sl'] if asset_type=='crypto' else tb.get('forex_sl' if asset_type=='forex' else 'trad_sl', 0.01)
                                 pt_pct = tb['crypto_pt'] if asset_type=='crypto' else tb.get('forex_pt' if asset_type=='forex' else 'trad_pt', 0.02)
                            
                            tp_price = entry_price * (1+pt_pct)
                            sl_price = entry_price * (1-sl_pct)
                            
                        else:
                            direction = -1
                            if tb.get('crypto_use_dynamic', False) and asset_type == 'crypto':
                                  sigma = row.get('sigma', 0.01)
                                  k_pt = tb.get('crypto_dyn_pt_k', 0.5)
                                  k_sl = tb.get('crypto_dyn_sl_k', 0.5)
                                  pt_pct = k_pt * sigma
                                  sl_pct = k_sl * sigma
                            else:
                                sl_pct = tb['crypto_sl'] if asset_type=='crypto' else tb.get('forex_sl' if asset_type=='forex' else 'trad_sl', 0.01)
                                pt_pct = tb['crypto_pt'] if asset_type=='crypto' else tb.get('forex_pt' if asset_type=='forex' else 'trad_pt', 0.02)

                            tp_price = entry_price * (1-pt_pct)
                            sl_price = entry_price * (1+sl_pct)

                        # Check Outcome (Future window)
                        if 'time_limit_bars' in tb:
                            offset = tb['time_limit_bars']
                        else:
                            offset = 20 # default
                            
                        start_loc = df.index.get_loc(entry_time)
                        end_loc = min(start_loc + offset, len(df))
                        
                        future = df.iloc[start_loc+1 : end_loc]
                        
                        outcome_pnl = 0.0
                        exit_reason = "TIMEOUT"
                        
                        # Check bars
                        for f_idx, f_row in future.iterrows():
                            if direction == 1:
                                if f_row['low'] <= sl_price:
                                    outcome_pnl = -sl_pct
                                    exit_reason = "SL"
                                    break
                                if f_row['high'] >= tp_price:
                                    outcome_pnl = pt_pct
                                    exit_reason = "TP"
                                    break
                            else:
                                if f_row['high'] >= sl_price:
                                    outcome_pnl = -sl_pct
                                    exit_reason = "SL"
                                    break
                                if f_row['low'] <= tp_price:
                                    outcome_pnl = pt_pct
                                    exit_reason = "TP"
                                    break
                        
                        # Timeout logic
                        if exit_reason == "TIMEOUT" and not future.empty:
                            exit_price = future.iloc[-1]['close']
                            if direction == 1:
                                raw_ret = (exit_price - entry_price) / entry_price
                            else:
                                raw_ret = (entry_price - exit_price) / entry_price
                            outcome_pnl = raw_ret
                            
                        results.append({
                            "Entry Time": entry_time,
                            "Model": model_name,
                            "Asset": symbol,
                            "Direction": "LONG" if direction == 1 else "SHORT",
                            "PnL": outcome_pnl,
                            "Exit": exit_reason,
                            "Prob": prob
                        })
                        
                except Exception as e:
                    # print(f"Error {symbol} {tf}: {e}")
                    pass

    # --- REPORTING ---
    if not results:
        print("No signals generated in backtest period.")
        return

    res_df = pd.DataFrame(results)
    res_df['Entry Time'] = pd.to_datetime(res_df['Entry Time'])
    
    # Ensure TZ-aware if needed
    if res_df['Entry Time'].dt.tz is None:
        res_df['Entry Time'] = res_df['Entry Time'].dt.tz_localize('UTC') # Assume UTC
        
    now = pd.Timestamp.now(tz='UTC')

    def print_stats(period_name, delta):
        cutoff = now - delta
        subset = res_df[res_df['Entry Time'] >= cutoff]
        
        print(f"\n--- {period_name} ---")
        if subset.empty:
            print("No trades.")
            return
            
        count = len(subset)
        win_rate = (subset['PnL'] > 0).mean() * 100
        total_pnl = subset['PnL'].sum() * 100 # In percent
        
        print(f"Total Signals: {count}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Cumulative PnL: {total_pnl:.1f}%")
        
        # By Model
        print(subset.groupby('Model')['PnL'].count())
        print("PnL by Model:")
        print(subset.groupby('Model')['PnL'].sum() * 100)

    print_stats("Last 24 Hours", timedelta(hours=24))
    print_stats("Last 7 Days", timedelta(days=7))
    print_stats("Last 30 Days", timedelta(days=30))
    
    # Global
    print("\n--- ALL TIME (Sample) ---")
    print(f"Total: {len(res_df)}")
    print(f"Win Rate: {(res_df['PnL'] > 0).mean()*100:.1f}%")
    print(f"Total PnL: {res_df['PnL'].sum()*100:.1f}%")

if __name__ == "__main__":
    run_backtest()

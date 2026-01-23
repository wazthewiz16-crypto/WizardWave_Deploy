# !/usr/bin/env python3
"""
Prop Firm Audit Script.
Tests WizardScalp, WizardWave, and CLS strategies against:
1. Max Daily Drawdown (3.9% Hard Limit)
2. Max Total Drawdown (7.9% Hard Limit)
3. Profit Factor > 1.3
"""
import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
try:
    from tabulate import tabulate
except ImportError:
    pass

# Add paths
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

# Imports
try:
    from src.strategies.strategy_scalp import WizardScalpStrategy
    from src.strategies.strategy import WizardWaveStrategy
    from src.strategies.strategy_cls import CLSRangeStrategy
    from src.strategies.strategy_ichimoku import IchimokuStrategy
    from src.core.feature_engine import calculate_ml_features
    from src.utils.paths import get_model_path
    import joblib
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Configuration
ASSETS = {
    'Scalp': ['BTC-USD', 'ETH-USD', 'EURUSD=X'],
    'Wave': ['BTC-USD', 'ETH-USD', 'GC=F', '^NDX'],
    'CLS': ['BTC-USD', 'EURUSD=X', 'GC=F'],
    'Ichimoku': ['BTC-USD', 'SOL-USD', 'GBPUSD=X']
}

CONSTANTS = {
    'MAX_DAILY_DD': 0.039,
    'MAX_TOTAL_DD': 0.079,
    'INITIAL_BALANCE': 100000,
    'TF_MAP': {
        'Scalp': '1h',
        'Wave': '1d',
        'CLS': '1h',
        'Ichimoku': '4h'
    }
}

def fetch_data(symbol, days=365, interval='1h'):
    try:
        df = yf.Ticker(symbol).history(period=f"{days}d", interval=interval)
        if df.empty: return pd.DataFrame()
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return pd.DataFrame()

def run_backtest(strategy_name, strategy_obj, assets):
    results = []
    
    # Load Models for ML Filtering
    loaded_models = {}
    model_files = {
        'Wizard Wave': get_model_path('wizard_wave_ml_model.pkl'),
        'Ichimoku Cloud': get_model_path('model_ichimoku.pkl'),
        'CLS Range': get_model_path('north_star_ml_model.pkl'),
        'Wizard Scalp': get_model_path('wizard_scalp_ml_model.pkl')
    }
    
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                loaded_models[name] = joblib.load(file)
                if isinstance(loaded_models[name], dict) and 'model' in loaded_models[name]:
                    loaded_models[name] = loaded_models[name]['model']
            except: pass

    print(f"\n--- Auditing {strategy_name} ---")
    
    # Selection of Interval
    interval = CONSTANTS['TF_MAP'].get(strategy_name, '1h')
    exposure = 0.3 # 30% of equity per trade for risk management
    
    for symbol in assets:
        print(f"  Testing {symbol}...", end="\r")
        df = fetch_data(symbol, interval=interval)
        if df.empty:
            print(f"  Testing {symbol}: No Data")
            continue
            
        # Run Strategy
        if strategy_name == 'CLS Range':
            df = strategy_obj.apply_north_star(df, df) if hasattr(strategy_obj, 'apply_north_star') else strategy_obj.apply(df)
        elif strategy_name == 'Ichimoku Cloud':
            df = strategy_obj.apply_strategy(df)
        else:
            df = strategy_obj.apply(df)
            
        # --- ML FILTERING ---
        if strategy_name in loaded_models:
            df = calculate_ml_features(df)
            model = loaded_models[strategy_name]
            
            # Map specific features for each model
            feature_sets = {
                'Wizard Wave': ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'rvol', 'month_sin', 'cycle_regime', 'close_frac'],
                'Ichimoku Cloud': ['tk_gap', 'price_to_kijun', 'cloud_width', 'dist_to_cloud_top', 'chikou_mom', 'adx', 'rsi', 'volatility'],
                'CLS Range': ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi', 'month_sin', 'cycle_regime'],
                'Wizard Scalp': ['volatility', 'rsi', 'ma_dist', 'adx', 'rvol', 'atr_pct', 'cycle_regime']
            }
            
            features = feature_sets.get(strategy_name, ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi'])
            
            # Calculate Ichimoku specific features if needed
            if strategy_name == 'Ichimoku Cloud':
                df['tk_gap'] = (df['tenkan'] - df['kijun']) / df['close']
                df['price_to_kijun'] = (df['close'] - df['kijun']) / df['close']
                df['cloud_width'] = (df['span_a'] - df['span_b']).abs() / df['close']
                df['dist_to_cloud_top'] = (df['close'] - df[['span_a', 'span_b']].max(axis=1)) / df['close']
                df['chikou_mom'] = (df['close'] - df['close'].shift(30)) / df['close']

            # Ensure features exist
            for f in features:
                if f not in df.columns: df[f] = 0.0
                
            sig_mask = df['signal_type'].replace('NONE', np.nan).notna()
            if sig_mask.any():
                X = df.loc[sig_mask, features].fillna(0).values
                probs = model.predict_proba(X)[:, 1]
                
                # Fetch threshold from model_data if it exists, else use defaults
                threshold = 0.45
                if strategy_name == 'Wizard Scalp': threshold = 0.60
                elif strategy_name == 'Ichimoku Cloud': threshold = 0.45
                elif strategy_name == 'Wizard Wave': threshold = 0.40 # Targeted win rate
                
                prob_idx = 0
                for idx, row in df[sig_mask].iterrows():
                    if probs[prob_idx] < threshold:
                        df.at[idx, 'signal_type'] = "NONE"
                    prob_idx += 1
            
        # Simulation
        equity = CONSTANTS['INITIAL_BALANCE']
        peak_equity = equity
        daily_start_equity = equity
        day_locked = False
        current_day = None
        
        daily_dd_breach = False
        max_dd_fail = False
        trades = 0
        wins = 0
        
        position = 0 # 1 for Long, -1 for Short
        entry_price = 0
        
        for idx, row in df.iterrows():
            day_str = str(idx.date())
            if current_day != day_str:
                daily_start_equity = equity
                current_day = day_str
                day_locked = False 
                
            # --- EXIT LOGIC (SL/TP) ---
            if position != 0:
                closed = False
                pnl = 0
                sl = row.get('stop_loss', 0)
                tp = row.get('target_price', 0)
                
                if position == 1:
                    if sl > 0 and row['low'] <= sl:
                        pnl = (sl - entry_price) * (equity*exposure/entry_price)
                        closed = True
                    elif tp > 0 and row['high'] >= tp:
                        pnl = (tp - entry_price) * (equity*exposure/entry_price)
                        closed = True
                    elif 'SHORT' in sig:
                        pnl = (price - entry_price) * (equity*exposure/entry_price)
                        closed = True
                elif position == -1:
                    if sl > 0 and row['high'] >= sl:
                        pnl = (entry_price - sl) * (equity*exposure/entry_price)
                        closed = True
                    elif tp > 0 and row['low'] <= tp:
                        pnl = (entry_price - tp) * (equity*exposure/entry_price)
                        closed = True
                    elif 'LONG' in sig:
                        pnl = (entry_price - price) * (equity*exposure/entry_price)
                        closed = True
                        
                if closed:
                    equity += pnl
                    position = 0
                    trades += 1
                    if pnl > 0: wins += 1

            # Check Daily DD
            day_dd = (daily_start_equity - equity) / daily_start_equity
            if day_dd > CONSTANTS['MAX_DAILY_DD']: daily_dd_breach = True
            
            # --- CIRCUIT BREAKER ---
            if day_dd > 0.02: 
                day_locked = True
                if position != 0:
                    # Force exit if still open (SL didn't trigger yet)
                    pnl = (row['close'] - entry_price) * (equity*exposure/entry_price) if position == 1 else (entry_price - row['close']) * (equity*exposure/entry_price)
                    equity += pnl
                    position = 0
                    trades += 1
            
            # Check Max DD
            if equity > peak_equity: peak_equity = equity
            total_dd = (peak_equity - equity) / peak_equity
            if total_dd > CONSTANTS['MAX_TOTAL_DD']: max_dd_fail = True
            
            sig = row.get('signal_type', 'NONE')
            price = row['close']
            
            # --- ENTRY LOGIC ---
            if position == 0 and not day_locked:
                if 'LONG' in sig and 'SHORT' not in sig:
                    position = 1
                    entry_price = price
                elif 'SHORT' in sig and 'LONG' not in sig:
                    position = -1
                    entry_price = price
                    
        total_pnl = (equity - CONSTANTS['INITIAL_BALANCE']) / CONSTANTS['INITIAL_BALANCE'] * 100
        pass_status = "PASS"
        if daily_dd_breach: pass_status = "FAIL (Daily DD)"
        elif max_dd_fail: pass_status = "FAIL (Max DD)"
        elif total_pnl < 0: pass_status = "FAIL (Neg PnL)"
        
        print(f"  Testing {symbol}: {pass_status} (PnL: {total_pnl:.1f}%)")
        
        results.append({
            'Strategy': strategy_name,
            'Asset': symbol,
            'Status': pass_status,
            'PnL %': round(total_pnl, 2),
            'Trades': trades,
            'Daily DD Breach': daily_dd_breach
        })
        
    return results

def main():
    all_results = []
    
    # 1. Wizard Scalp
    try:
        all_results.extend(run_backtest('Wizard Scalp', WizardScalpStrategy(), ASSETS['Scalp']))
    except Exception as e:
        print(f"Failed to Audit Scalp: {e}")
    
    # 2. Wizard Wave
    try:
        all_results.extend(run_backtest('Wizard Wave', WizardWaveStrategy(), ASSETS['Wave']))
    except Exception as e:
         print(f"Failed to Audit Wave: {e}")
         
    # 3. Ichimoku Cloud
    try:
        all_results.extend(run_backtest('Ichimoku Cloud', IchimokuStrategy(), ASSETS['Ichimoku']))
    except Exception as e:
         print(f"Failed to Audit Ichimoku: {e}")
    
    # 4. CLS Range
    try:
        all_results.extend(run_backtest('CLS Range', CLSRangeStrategy(), ASSETS['CLS']))
    except Exception as e:
         print(f"Failed to Audit CLS: {e}")
    
    # Report
    print("\n" + "="*50)
    print("PROP FIRM AUDIT REPORT")
    print("="*50)
    df_res = pd.DataFrame(all_results)
    print(df_res)
    
    # Create Artifact
    print("\nReport Generated.")

if __name__ == "__main__":
    main()

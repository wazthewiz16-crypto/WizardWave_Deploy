import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features

def debug_scalp_eth():
    symbol = 'ETH-USD'
    df = yf.Ticker(symbol).history(period="180d", interval="1h")
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    strat = WizardScalpStrategy()
    df = strat.apply(df)
    
    # ML Filter
    model_file = 'wizard_scalp_ml_model.pkl'
    if os.path.exists(model_file):
        loaded = joblib.load(model_file)
        model = loaded['model']
        thresh = 0.60
        features = ['volatility', 'rsi', 'ma_dist', 'adx', 'rvol', 'atr_pct', 'cycle_regime']
        
        df = calculate_ml_features(df)
        sig_mask = df['signal_type'] != 'NONE'
        if sig_mask.any():
            X = df.loc[sig_mask, features].fillna(0).values
            probs = model.predict_proba(X)[:, 1]
            prob_idx = 0
            for idx, row in df[sig_mask].iterrows():
                if probs[prob_idx] < thresh:
                    df.at[idx, 'signal_type'] = "NONE"
                prob_idx += 1

    equity = 100000
    position = 0
    entry_price = 0
    
    print(f"| Date | Action | Price | SL | PnL | Equity |")
    
    for idx, row in df.iterrows():
        sig = row['signal_type']
        price = row['close']
        sl = row.get('stop_loss', 0)
        
        # Exit
        if position != 0:
            closed = False
            pnl_val = 0
            if position == 1:
                if sl > 0 and row['low'] <= sl:
                    pnl_val = (sl - entry_price) * (equity/entry_price)
                    closed = True
                elif 'SHORT' in sig:
                    pnl_val = (price - entry_price) * (equity/entry_price)
                    closed = True
            elif position == -1:
                if sl > 0 and row['high'] >= sl:
                    pnl_val = (entry_price - sl) * (equity/entry_price)
                    closed = True
                elif 'LONG' in sig:
                    pnl_val = (entry_price - price) * (equity/entry_price)
                    closed = True
            
            if closed:
                equity += pnl_val
                print(f"| {idx} | EXIT | {price:.2f} | - | {pnl_val:.2f} | {equity:.2f} |")
                position = 0
        
        # Entry
        if position == 0:
            if 'LONG' in sig:
                position = 1
                entry_price = price
                print(f"| {idx} | LONG | {price:.2f} | {sl:.2f} | - | - |")
            elif 'SHORT' in sig:
                position = -1
                entry_price = price
                print(f"| {idx} | SHORT | {price:.2f} | {sl:.2f} | - | - |")

if __name__ == "__main__":
    debug_scalp_eth()

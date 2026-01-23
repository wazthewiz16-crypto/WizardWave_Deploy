import pandas as pd
import numpy as np
import joblib
from train_snipes import process_asset_snipes
from sklearn.metrics import classification_report

def run_backtest_report():
    print("Loading Spell Snipes Model...")
    try:
        model = joblib.load("model_snipes.pkl")
    except:
        print("Model not found. Run train_snipes.py first.")
        return

    assets = [
        ('BTC/USDT', 'crypto'),
        ('AVAX/USDT', 'crypto'), 
        ('^NDX', 'trad'), 
        ('^GSPC', 'trad'), 
        ('GC=F', 'trad'),
        ('EURUSD=X', 'forex')
    ]
    
    results = []
    
    print("\nRunning Backtest on Last ~60 Days (15m Timeframe)...")
    print("Strategy: Triple Barrier (Dynamic Exit) | Confidence Threshold: 0.60")
    
    for sym, atype in assets:
        df = process_asset_snipes(sym, atype, '15m')
        if df is None: continue
        
        # Features
        exclude = ['open','high','low','close','volume','target','time','local_time']
        feats = [c for c in df.columns if c not in exclude and 'date' not in c]
        
        # Predict Proba
        X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        probs = model.predict_proba(X)
        
        # Classes: [-1, 0, 1] usually. Check model.classes_
        classes = model.classes_
        # Map class index
        try:
            idx_long = list(classes).index(1)
            idx_short = list(classes).index(-1)
        except:
             # Imbalanced classes might miss one?
             # Assuming standard training had all
             print(f"  Warning: Classes {classes} missing -1 or 1.")
             continue
        
        # Filter High Confidence
        threshold = 0.60
        
        long_sigs = probs[:, idx_long] > threshold
        short_sigs = probs[:, idx_short] > threshold
        
        df['pred_signal'] = 0
        df.loc[long_sigs, 'pred_signal'] = 1
        df.loc[short_sigs, 'pred_signal'] = -1
        
        signals = df[df['pred_signal'] != 0]
        
        if signals.empty:
            results.append({"Asset": sym, "Trades": 0, "Win Rate": "0%", "Net PnL": "0%"})
            continue
            
        # Simulate PnL (Triple Barrier Style)
        # We assume entry at Close (signal generated).
        # We hold for max 12 bars (from training barrier_len).
        # We exit if TP (0.5% / 0.2%) or SL (-0.5% / -0.2%) hit.
        
        wins = 0
        losses = 0
        net_pnl = 0.0
        
        tgt = 0.005 if atype == 'crypto' else 0.002
        barrier_len = 12
        
        for t, row in signals.iterrows():
            entry = row['close']
            sig = row['pred_signal']
            
            # Future 12 bars
            future = df.loc[t:].iloc[1:barrier_len+1]
            
            outcome_pnl = 0.0
            hit = False
            
            for ft, frow in future.iterrows():
                ret = (frow['close'] - entry) / entry
                
                if sig == 1:
                    if ret >= tgt:
                        outcome_pnl = tgt
                        hit = True
                        break
                    elif ret <= -tgt:
                        outcome_pnl = -tgt
                        hit = True
                        break
                else: # Short
                    if ret <= -tgt: # Price drop = Profit
                        outcome_pnl = tgt # Positive return
                        hit = True
                        break
                    elif ret >= tgt:
                        outcome_pnl = -tgt
                        hit = True
                        break
            
            if not hit:
                # Time Limit Exit
                last_price = future.iloc[-1]['close'] if not future.empty else entry
                ret = (last_price - entry) / entry
                if sig == 1: outcome_pnl = ret
                else: outcome_pnl = -ret
            
            # Fee
            outcome_pnl -= 0.001
            
            net_pnl += outcome_pnl
            if outcome_pnl > 0: wins += 1
            else: losses += 1
            
        count = wins + losses
        wr = wins/count if count > 0 else 0
        
        print(f"  {sym}: Trades: {count}, WR: {wr:.0%}, Net: {net_pnl:.2%}")
        results.append({
            "Asset": sym,
            "Trades": count,
            "Win Rate": f"{wr:.0%}",
            "Net PnL": f"{net_pnl:.2%}"
        })
        
    print("\n=== Spell Snipes Strategy Results (Last 60 Days) ===")
    print(pd.DataFrame(results).to_markdown())

if __name__ == "__main__":
    run_backtest_report()

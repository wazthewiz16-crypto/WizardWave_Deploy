import pandas as pd
import numpy as np
import joblib
import pandas_ta as ta
from train_snipes_v2 import process_asset_snipes_dynamic
from feature_snipes import SpellSnipesFeatures

def run_backtest_report_v2():
    print("Loading Models...")
    try:
        model_trad = joblib.load("model_snipes_tradfi.pkl")
        # Crypto model failed training, so we skip it or try loading
        # model_crypto = joblib.load("model_snipes_crypto.pkl") 
    except:
        print("Model not found. Run train_snipes_v2.py first.")
        return

    assets = [
        ('^NDX', 'trad'), 
        ('^GSPC', 'trad'), 
        ('GC=F', 'trad'),
        ('EURUSD=X', 'forex')
    ]
    
    results = []
    
    # Strict Confidence to boost precision
    CONF_THRESH = 0.65
    
    print(f"\nRunning Backtest (Last 60 Days, 15m)...")
    print(f"Strategy: Dynamic ATR Stops (1.5x) | Target 2.0R | Confidence > {CONF_THRESH}")
    
    for sym, atype in assets:
        df = process_asset_snipes_dynamic(sym, atype, '15m')
        if df is None: continue
        
        # Recalculate ATR for simulation (process_asset adds features, but not ATR explicitly in column maybe?)
        # SpellSnipesFeatures adds BB/VWAP. 
        # train_snipes_v2 calculated ATR inside the labeler.
        # We need ATR here for Trade Simulation.
        
        # Calc ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Features for Prediction
        exclude = ['open','high','low','close','volume','target','time','local_time','atr']
        feats = [c for c in df.columns if c not in exclude and 'date' not in c]
        
        # Predict
        X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Use TradFi model
        model = model_trad
        
        probs = model.predict_proba(X)
        classes = model.classes_
        
        # Map classes
        try:
            idx_long = list(classes).index(1)
            idx_short = list(classes).index(-1)
        except:
            print(f"Model missing classes: {classes}")
            continue
            
        long_sigs = probs[:, idx_long] > CONF_THRESH
        short_sigs = probs[:, idx_short] > CONF_THRESH
        
        df['pred_signal'] = 0
        df.loc[long_sigs, 'pred_signal'] = 1
        df.loc[short_sigs, 'pred_signal'] = -1
        
        signals = df[df['pred_signal'] != 0]
        
        if signals.empty:
            results.append({"Asset": sym, "Trades": 0, "Win Rate": "0%", "Net PnL": "0%"})
            continue
            
        wins = 0
        losses = 0
        net_pnl = 0.0
        
        # Simulation
        barrier_len = 24 # 6 hours
        atr_mult = 1.5
        target_r = 2.0
        
        for t, row in signals.iterrows():
            entry = row['close']
            sig = row['pred_signal']
            vol = row['atr']
            
            if pd.isna(vol) or vol == 0: continue
            
            # Set Dynamic Levels
            sl_dist = vol * atr_mult
            # Min SL check (0.1%)
            if sl_dist < entry * 0.001: sl_dist = entry * 0.001
            
            tp_dist = sl_dist * target_r
            
            future = df.loc[t:].iloc[1:barrier_len+1]
            
            outcome_pnl = 0.0
            hit = False
            
            for ft, frow in future.iterrows():
                if sig == 1: # LONG
                    # Hit TP?
                    if frow['high'] >= entry + tp_dist:
                        outcome_pnl = (tp_dist / entry)
                        hit = True
                        break
                    # Hit SL?
                    if frow['low'] <= entry - sl_dist:
                        outcome_pnl = -(sl_dist / entry)
                        hit = True
                        break
                else: # SHORT
                    # Hit TP? (Price Drops)
                    if frow['low'] <= entry - tp_dist:
                        outcome_pnl = (tp_dist / entry)
                        hit = True
                        break
                    # Hit SL? (Price Rises)
                    if frow['high'] >= entry + sl_dist:
                        outcome_pnl = -(sl_dist / entry)
                        hit = True
                        break
            
            if not hit:
                # Time Exit
                last_c = future.iloc[-1]['close'] if not future.empty else entry
                if sig == 1: outcome_pnl = (last_c - entry)/entry
                else: outcome_pnl = (entry - last_c)/entry
                
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

    print("\n=== Spell Snipes V2 (RR 2.0) Results ===")
    print(pd.DataFrame(results).to_markdown())

if __name__ == "__main__":
    run_backtest_report_v2()

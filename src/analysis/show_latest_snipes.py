import pandas as pd
import numpy as np
import joblib
from train_snipes_v2 import process_asset_snipes_dynamic
from feature_snipes import SpellSnipesFeatures

def show_latest_signals():
    print("Loading Model...")
    try:
        model = joblib.load("model_snipes_tradfi.pkl")
    except:
        print("Model not found.")
        return

    assets = ['GC=F', '^NDX']
    CONF_THRESH = 0.65
    
    for sym in assets:
        print(f"\nScanning {sym} for latest signals...")
        df = process_asset_snipes_dynamic(sym, 'trad', '15m')
        if df is None: continue
        
        # Calc ATR
        high, low, c = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = (high - c.shift(1)).abs()
        tr3 = (low - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Predict
        exclude = ['open','high','low','close','volume','target','time','local_time','atr']
        feats = [c for c in df.columns if c not in exclude and 'date' not in c]
        X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        probs = model.predict_proba(X)
        classes = model.classes_
        
        try:
            idx_long = list(classes).index(1)
            idx_short = list(classes).index(-1)
        except: continue
        
        # Signals
        df['prob_long'] = probs[:, idx_long]
        df['prob_short'] = probs[:, idx_short]
        
        long_sigs = df['prob_long'] > CONF_THRESH
        short_sigs = df['prob_short'] > CONF_THRESH
        
        df['signal'] = "NONE"
        df.loc[long_sigs, 'signal'] = "LONG"
        df.loc[short_sigs, 'signal'] = "SHORT"
        
        signals = df[df['signal'] != "NONE"].copy()
        
        if signals.empty:
            print("No signals found.")
            continue
            
        # Enrich with TP/SL/Outcome
        atr_mult = 1.5
        target_r = 2.0
        barrier_len = 24
        
        display_data = []
        
        # Look at last 10 signals
        recent_sigs = signals.tail(10)
        
        for t, row in recent_sigs.iterrows():
            entry = row['close']
            sig_type = row['signal']
            vol = row['atr']
            ts = t
            
            if pd.isna(vol): continue
            
            sl_dist = max(vol * atr_mult, entry * 0.001)
            tp_dist = sl_dist * target_r
            
            if sig_type == "LONG":
                tp = entry + tp_dist
                sl = entry - sl_dist
            else:
                tp = entry - tp_dist
                sl = entry + sl_dist
                
            # Outcome Check
            future = df.loc[t:].iloc[1:barrier_len+1]
            outcome = "OPEN/PENDING"
            pnl_pct = 0.0
            
            # Check price action
            for ft, frow in future.iterrows():
                if sig_type == "LONG":
                    if frow['high'] >= tp:
                        outcome = "WIN (TP)"
                        pnl_pct = (tp - entry)/entry
                        break
                    if frow['low'] <= sl:
                        outcome = "LOSS (SL)"
                        pnl_pct = (sl - entry)/entry
                        break
                else:
                    if frow['low'] <= tp:
                        outcome = "WIN (TP)"
                        pnl_pct = (entry - tp)/entry
                        break
                    if frow['high'] >= sl:
                        outcome = "LOSS (SL)"
                        pnl_pct = (entry - sl)/entry
                        break
            
            if outcome == "OPEN/PENDING" and not future.empty:
                # Check outcome if time expired
                if len(future) == barrier_len:
                    last_c = future.iloc[-1]['close']
                    outcome = "TIME LIMIT"
                    if sig_type == "LONG": pnl_pct = (last_c - entry)/entry
                    else: pnl_pct = (entry - last_c)/entry
            
            conf = row['prob_long'] if sig_type == "LONG" else row['prob_short']
            
            display_data.append({
                "Time": str(ts),
                "Type": sig_type,
                "Price": f"{entry:.2f}",
                "TP": f"{tp:.2f}",
                "SL": f"{sl:.2f}",
                "Confidence": f"{conf:.0%}",
                "Result": outcome,
                "PnL": f"{pnl_pct:.2%}"
            })
            
        print(f"\nLatest 10 Signals for {sym}:")
        print(pd.DataFrame(display_data).to_markdown())

if __name__ == "__main__":
    show_latest_signals()

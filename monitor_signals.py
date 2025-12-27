import time
import json
import os
import joblib
import pandas as pd
import random
import urllib.request
from datetime import datetime
import traceback

# Import Project Modules
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from feature_engine import calculate_ml_features
from pipeline import get_asset_type

# --- CONFIGURATION ---
try:
    with open('strategy_config.json', 'r') as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"Error loading strategy_config.json: {e}")
    CONFIG = {}

try:
    with open('discord_config.json', 'r') as f:
        DISCORD_CONFIG = json.load(f)
        WEBHOOK_URL = DISCORD_CONFIG.get('webhook_url')
except:
    WEBHOOK_URL = None

PROCESSED_FILE = 'processed_signals.json'

# --- HELPERS ---

def send_discord_alert(webhook_url, signal_data):
    """Sends a formatted trade signal to Discord."""
    try:
        def parse_float(val):
            try:
                if isinstance(val, str):
                    return float(val.replace(',', '').replace('%', ''))
                return float(val)
            except:
                return 0.0

        action = str(signal_data.get('Action','')).upper()
        asset = str(signal_data.get('Asset', 'Unknown'))
        timeframe = str(signal_data.get('Timeframe', 'N/A'))
        
        # Determine Direction and Color
        raw_type = str(signal_data.get('Type', '')).upper()
        raw_signal = str(signal_data.get('Signal', '')).upper()
        
        is_long = "LONG" in raw_type or "LONG" in raw_signal or "BULL" in raw_type
        direction_str = "🟢 LONG" if is_long else "🔴 SHORT"
        color = 65280 if is_long else 16711680 # Green or Red
        
        entry_price = parse_float(signal_data.get('Entry_Price', 0))
        take_profit = parse_float(signal_data.get('Take_Profit', 0))
        stop_loss = parse_float(signal_data.get('Stop_Loss', 0))
        confidence = signal_data.get('Confidence_Score', 0) 
        if confidence == 0 and 'Confidence' in signal_data:
             confidence = parse_float(signal_data['Confidence'])
             if confidence < 1.0: confidence *= 100 

        # Calculate R:R
        rr_str = "N/A"
        try:
            if entry_price > 0:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                if risk > 0:
                    rr_ratio = reward / risk
                    rr_str = f"{rr_ratio:.2f}R"
        except:
            pass
        
        embed = {
            "title": f"🔮 WIZARD PROPHECY: {asset} {direction_str}",
            "description": f"**Direction:** {direction_str}\n**Timeframe:** {timeframe}\n**Action:** {action}",
            "color": color,
            "fields": [
                {"name": "Entry Price", "value": f"{entry_price}", "inline": True},
                {"name": "Take Profit", "value": f"{take_profit}", "inline": True},
                {"name": "Stop Loss", "value": f"{stop_loss}", "inline": True},
                {"name": "Confidence", "value": f"{confidence:.1f}%", "inline": True},
                {"name": "R:R", "value": rr_str, "inline": True},
                {"name": "Entry Time", "value": f"{signal_data.get('Entry_Time', 'N/A')}", "inline": False}
            ],
            "footer": {"text": "WizardWave v1.0 • Automated Signal (Monitor)"}
        }

        data = {
            "username": "WizardWave Oracle",
            "embeds": [embed]
        }

        headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(webhook_url, data=json.dumps(data).encode('utf-8'), headers=headers)
        
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status not in [200, 204]:
                print(f"Failed to send Discord alert: {response.status}")
                
    except Exception as e:
        print(f"Error sending Discord alert: {e}")

def process_alerts(df):
    """Checks for new signals and sends alerts."""
    if not WEBHOOK_URL:
        print("No Webhook URL found.")
        return

    MAX_ALERT_AGE_HOURS = 3
    now_est = pd.Timestamp.now(tz='America/New_York')

    for _, row in df.iterrows():
        try:
            action_str = str(row.get('Action', '')).upper()
            is_take = "TAKE" in action_str or "✅" in action_str
            
            if not is_take:
                continue

            entry_time_str = str(row.get('Entry_Time', ''))
            asset = str(row.get('Asset', ''))
            tf = str(row.get('Timeframe', ''))
            
            sig_id = f"{asset}_{tf}_{entry_time_str}".replace(" ", "_")
            if sig_id == "__": continue

            # Freshness Check
            try:
                entry_dt = pd.to_datetime(entry_time_str)
                if entry_dt.tzinfo is None:
                    # Assume EST/New York if naive, or UTC -> convert
                    entry_dt = entry_dt.tz_localize('America/New_York', ambiguous='infer') # Approximation for now
                
                # Compare
                if (now_est - entry_dt).total_seconds() > (MAX_ALERT_AGE_HOURS * 3600):
                    # print(f"Skipping old signal: {sig_id}")
                    continue
            except:
                pass 

            # Deduplication
            current_ids = set()
            if os.path.exists(PROCESSED_FILE):
                try:
                    with open(PROCESSED_FILE, 'r') as f:
                        content = json.load(f)
                        if isinstance(content, list):
                            current_ids = set(content)
                except:
                    pass

            if sig_id not in current_ids:
                print(f"Sending ALERT for {sig_id}")
                send_discord_alert(WEBHOOK_URL, row)
                
                current_ids.add(sig_id)
                try:
                    with open(PROCESSED_FILE, 'w') as f:
                        json.dump(list(current_ids), f)
                except Exception as e:
                    print(f"Error saving processed ID: {e}")
                    
        except Exception as e:
            print(f"Error processing row: {e}")

# --- CORE LOGIC ---

def run_analysis_cycle():
    """Runs one full pass of data fetching and prediction."""
    print(f"Starting Analysis Cycle at {datetime.now()}")
    
    # Load Models
    try:
        model_htf = joblib.load(CONFIG['htf']['model_file'])
        model_ltf = joblib.load(CONFIG['ltf']['model_file'])
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    all_signals = []

    # 1. HTF
    for tf in CONFIG['htf']['timeframes']:
        for symbol in CONFIG['assets']:
            try:
                # Determine Fetch Type
                asset_type = get_asset_type(symbol)
                fetch_type = 'trad' if asset_type == 'forex' or asset_type == 'trad' else 'crypto'
                if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'

                # Fetch
                df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=300)
                if df.empty: continue

                # Strategy
                strat = WizardWaveStrategy()
                df = strat.apply(df)
                
                # Features
                df = calculate_ml_features(df)
                df = df.dropna()
                
                if df.empty: continue
                
                # Predict
                last_row = df.iloc[-1].copy() # Ensure copy
                features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
                
                X_new = pd.DataFrame([last_row[features]])
                prob = model_htf.predict_proba(X_new)[0][1]
                
                # Create Signal Object
                signal_type = last_row['signal_type']
                if signal_type != 'NONE':
                    # Populate Signal Data
                    # Calculate TP/SL from config for strict alerting?
                    # Or use the strategy defaults? 
                    # app.py seems to use pre-calc.
                    # We will re-calc simplified for alerting.
                    
                    price = last_row['close']
                    
                    # Logic from apply_triple_barrier (simplified for LIVE)
                    tb = CONFIG['htf']['triple_barrier']
                    if 'LONG' in signal_type:
                        sl_pct = tb['crypto_sl'] if asset_type=='crypto' else tb['trad_sl']
                        pt_pct = tb['crypto_pt'] if asset_type=='crypto' else tb['trad_pt']
                        sl_price = price * (1 - sl_pct)
                        pt_price = price * (1 + pt_pct)
                    else:
                        sl_pct = tb['crypto_sl'] if asset_type=='crypto' else tb['trad_sl']
                        pt_pct = tb['crypto_pt'] if asset_type=='crypto' else tb['trad_pt']
                        sl_price = price * (1 + sl_pct)
                        pt_price = price * (1 - pt_pct)

                    is_take = "✅ TAKE" if prob > 0.55 else "⚠️ WAIT"
                    
                    sig = {
                        "Asset": symbol,
                        "Timeframe": tf,
                        "Action": f"{is_take}", # Valid for existing parser
                        "Type": "LONG" if "LONG" in signal_type else "SHORT",
                        "Signal": signal_type,
                        "Entry_Price": price,
                        "Current_Price": price,
                        "Entry_Time": str(last_row.name), # Index is timestamp
                        "Confidence": f"{prob*100:.1f}%",
                        "Confidence_Score": prob*100,
                        "Take_Profit": round(pt_price, 4),
                        "Stop_Loss": round(sl_price, 4)
                    }
                    all_signals.append(sig)
            
            except Exception as e:
                # print(f"Error HTF {symbol} {tf}: {e}")
                pass

    # 2. LTF
    for tf in CONFIG['ltf']['timeframes']:
        for symbol in CONFIG['assets']:
            try:
                asset_type = get_asset_type(symbol)
                fetch_type = 'trad' if asset_type == 'forex' or asset_type == 'trad' else 'crypto'
                if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'

                df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=300)
                if df.empty: continue

                strat = WizardScalpStrategy()
                df = strat.apply(df)
                df = calculate_ml_features(df)
                df = df.dropna()
                
                if df.empty: continue

                last_row = df.iloc[-1].copy()
                features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
                X_new = pd.DataFrame([last_row[features]])
                prob = model_ltf.predict_proba(X_new)[0][1]
                
                signal_type = last_row['signal_type']
                if signal_type != 'NONE':
                    price = last_row['close']
                    tb = CONFIG['ltf']['triple_barrier']
                    
                    # Logic for LTF (Dynamic Handling)
                    is_crypto_dynamic = (asset_type == 'crypto' and tb.get('crypto_use_dynamic', False))
                    sigma = last_row.get('sigma', 0.01) # Usually calculated in pipeline
                    
                    # Need to calc sigma if missing
                    # Or just use static fallback for robustness
                    if 'sigma' not in last_row:
                        # Simple ATR aprox or fallback
                        sl_pct = tb['crypto_sl'] 
                        pt_pct = tb['crypto_pt']
                    else:
                         sl_pct = tb['crypto_sl'] # Placeholder if dynamic not fully implemented here
                         pt_pct = tb['crypto_pt']
                    
                    if 'LONG' in signal_type:
                        sl_price = price * (1 - sl_pct)
                        pt_price = price * (1 + pt_pct)
                    else:
                        sl_price = price * (1 + sl_pct)
                        pt_price = price * (1 - pt_pct)

                    is_take = "✅ TAKE" if prob > 0.55 else "⚠️ WAIT"

                    sig = {
                        "Asset": symbol,
                        "Timeframe": tf,
                        "Action": f"{is_take}",
                        "Type": "LONG" if "LONG" in signal_type else "SHORT",
                        "Signal": signal_type,
                        "Entry_Price": price,
                        "Current_Price": price,
                        "Entry_Time": str(last_row.name),
                        "Confidence": f"{prob*100:.1f}%",
                        "Confidence_Score": prob*100,
                        "Take_Profit": round(pt_price, 4),
                        "Stop_Loss": round(sl_price, 4)
                    }
                    all_signals.append(sig)

            except Exception as e:
                pass

    # Process
    if all_signals:
        df_sig = pd.DataFrame(all_signals)
        process_alerts(df_sig)
    else:
        print("No signals found in this cycle.")

if __name__ == "__main__":
    print("🔮 WizardWave Signal Monitor Started...")
    print(f"Webhook URL: {WEBHOOK_URL} (Loaded)")
    
    while True:
        try:
            run_analysis_cycle()
        except Exception as e:
            print(f"Cycle crashed: {e}")
            traceback.print_exc()
        
        # Wait 5 minutes
        print("Sleeping for 300s...")
        time.sleep(300)

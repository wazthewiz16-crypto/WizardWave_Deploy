import time
import json
import os
import requests
from datetime import datetime, timedelta

# --- CONFIG ---
RAW_DATA_FILE = "tv_raw_data.json"
HISTORY_FILE = "processed_alerts.json"

# Webhook URL
WEBHOOK_URL = "https://discord.com/api/webhooks/1321966675529924618/u0kYg0uT3o2J-A4nZYL-uAgV5oiET0SzkM1oJLpMfl2Z3-vN_V5vVp7U5r5d5b5l5e" 

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def send_discord_alert(payload):
    try:
        requests.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        print(f"  -> Discord Fail: {e}")

def process_logic():
    print(f"[{datetime.now()}] Checking for new data...")
    
    if not os.path.exists(RAW_DATA_FILE):
        print("  -> No raw data file found.")
        return

    try:
        with open(RAW_DATA_FILE, "r") as f:
            all_assets = json.load(f)
    except Exception as e:
        print(f"  -> JSON Load Error: {e}")
        return

    history = load_history()
    new_alerts_sent = 0

    for asset_entry in all_assets:
        name = asset_entry['asset']
        data_map = asset_entry['raw_data']
        
        # Confirmation Pairs
        pairs = [("15m", "1h"), ("1h", "4h"), ("4h", "1d"), ("1d", "4d")]
        
        for low_tf, high_tf in pairs:
            if low_tf not in data_map or high_tf not in data_map:
                continue
                
            curr = data_map[low_tf]
            htf = data_map[high_tf]
            
            p = curr['close']
            if not p: continue
            
            # --- MANGO PARSING LOGIC ---
            # Filter for Price-Like values (within 50% of current price)
            valid_vals = [v for v in curr['mango'] if abs(v - p) < p * 0.5]
            htf_vals_clean = [v for v in htf['mango'] if htf['close'] and abs(v - htf['close']) < htf['close'] * 0.5]

            # We need at least 2 values for Trend
            if len(valid_vals) < 2: 
                continue
            
            # Retrieve Values
            # User confirmed 5 values:
            # 0: MangoD1
            # 1: MangoD2
            # 2: BuyOp / SellOp (Trigger Price?)
            # 3: Entry Zone Upper
            # 4: Entry Zone Lower
            
            if len(curr['mango']) < 5: 
                continue
                
            def parse_mango_vals(vals):
                # Strict Mapping from Screenshot
                return {
                    "d1": vals[0],
                    "d2": vals[1],
                    "op": vals[2],
                    "entry_up": vals[3],
                    "entry_down": vals[4]
                }
            
            low_vals = parse_mango_vals(curr['mango'])
            
            # HTF might not have all 5 if data window is scrolled or different?
            # Assuming HTF has same structure.
            if len(htf['mango']) < 5: continue
            htf_vals = parse_mango_vals(htf['mango'])
            
            # --- TREND LOGIC ---
            def get_trend(price, v):
               upper = max(v['d1'], v['d2'])
               lower = min(v['d1'], v['d2'])
               if price > upper: return "Bullish"
               if price < lower: return "Bearish"
               return "Neutral"
               
            curr_trend = get_trend(curr['close'], low_vals)
            htf_trend = get_trend(htf['close'], htf_vals)
            
            sig_type = None
            sl_price = 0.0
            
            # LONG: Bullish + Price < Entry Top
            if curr_trend == "Bullish" and htf_trend == "Bullish":
                if p <= low_vals['entry_up']:
                    sig_type = "LONG"
                    # SL below the cloud
                    sl_price = min(low_vals['d1'], low_vals['d2']) * 0.995

            # SHORT: Bearish + Price > Entry Bottom
            if curr_trend == "Bearish" and htf_trend == "Bearish":
                if p >= low_vals['entry_down']:
                    sig_type = "SHORT"
                    # SL above the cloud
                    sl_price = max(low_vals['d1'], low_vals['d2']) * 1.005
            
            if sig_type:
                # --- DEDUPLICATION ---
                uid = f"{name}_{low_tf}_{sig_type}"
                last_ts_str = history.get(uid)
                
                should_alert = False
                if not last_ts_str:
                    should_alert = True
                else:
                    try:
                        last_ts = datetime.fromisoformat(last_ts_str)
                        hours = (datetime.now() - last_ts).total_seconds() / 3600
                        limit = 4
                        if "4h" in low_tf: limit = 12
                        if "1d" in low_tf: limit = 24
                        if hours > limit: should_alert = True
                    except: should_alert = True
                
                if should_alert:
                    print(f"  >>> SIGNAL FOUND: {name} {low_tf} {sig_type} @ {p}")
                    payload = {
                        "username": "Mango Oracle 🔮",
                        "embeds": [{
                            "title": f"🔮 {sig_type} Signal: {name}",
                            "description": f"**Timeframe:** {low_tf}\n**Price:** {p}\n**SL:** {sl_price:.4f}\n**Confirm:** {high_tf} Trend Aligned",
                            "color": 5763719 if sig_type == "LONG" else 15548997
                        }]
                    }
                    send_discord_alert(payload)
                    history[uid] = datetime.now().isoformat()
                    new_alerts_sent += 1

    save_history(history)
    print(f"  -> Cycle Done. Sent {new_alerts_sent} alerts.")

if __name__ == "__main__":
    import sys
    run_once = "--once" in sys.argv
    
    if run_once:
        process_logic()
    else:
        while True:
            try:
                process_logic()
            except Exception as e:
                print(f"Alert Loop Crash: {e}")
            print("Sleeping 60s...")
            time.sleep(60)

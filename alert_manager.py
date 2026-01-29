import time
import json
import os
import requests
from datetime import datetime, timedelta

# --- CONFIG ---
RAW_DATA_FILE = "tv_raw_data.json"
HISTORY_FILE = "processed_alerts.json"

# Webhook URL
WEBHOOK_URL = "https://discord.com/api/webhooks/1453411250548510730/sGwwz8eauP1VYw_6pMOfWwwJaBuxN8fJuUjKH5mIGSsrplyTBLhLSU07L5lQ84MS7qlF" 

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
            
            # --- TIME FILTER (User Habit: No Overnight) ---
            from datetime import timezone
            def get_est_time():
                # Force UTC-5 regardless of system headers
                return datetime.now(timezone.utc) - timedelta(hours=5)
            
            now_est = get_est_time()
            
            # Active Hours: 05:00 (5 AM) to 23:00 (11 PM)
            if now_est.hour < 5 or now_est.hour >= 23:
                # print("  (Night Mode - Skipping)")
                continue

            # LONG: Bullish + Price INSIDE Zone
            # Strict: Must be > EntryDown AND < EntryUp
            if curr_trend == "Bullish" and htf_trend == "Bullish":
                if p <= low_vals['entry_up'] and p >= low_vals['entry_down']:
                    sig_type = "LONG"
                    # SL Logic: Tighter for LTF, Wider for HTF
                    is_ltf = low_tf in ["15m", "1h", "4h"]
                    buffer = 0.005 if is_ltf else 0.02 
                    
                    cloud_bottom = min(low_vals['d1'], low_vals['d2'])
                    sl_price = cloud_bottom * (1 - buffer)

            # SHORT: Bearish + Price INSIDE Zone
            # Strict: Must be > EntryUp (Wait, usually Entry Zone is bracket. If price is inside [Down, Up], it's valid.)
            if curr_trend == "Bearish" and htf_trend == "Bearish":
                if p >= low_vals['entry_down'] and p <= low_vals['entry_up']:
                    sig_type = "SHORT"
                    # SL Logic
                    is_ltf = low_tf in ["15m", "1h", "4h"]
                    buffer = 0.005 if is_ltf else 0.02
                    
                    cloud_top = max(low_vals['d1'], low_vals['d2'])
                    sl_price = cloud_top * (1 + buffer)
            
            if sig_type:
                # --- CALCULATE TP (Risk:Reward) ---
                # LTF (15m, 1h, 4h) -> 2R
                # HTF (12h, 1d, 4d) -> 3R
                risk = abs(p - sl_price)
                rr_mult = 2 if low_tf in ["15m", "1h", "4h"] else 3
                
                if sig_type == "LONG":
                    tp_price = p + (risk * rr_mult)
                else:
                    tp_price = p - (risk * rr_mult)

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
                    
                    # Timestamp
                    entry_time_str = now_est.strftime('%Y-%m-%d %H:%M EST')
                    
                    # 1. Discord Alert
                    payload = {
                        "username": "Mango Oracle 🔮",
                        "embeds": [{
                            "title": f"🔮 {sig_type} Signal: {name}",
                            "description": f"**Timeframe:** {low_tf}\n**Entry:** {p}\n**TP ({rr_mult}R):** {tp_price:.4f}\n**SL:** {sl_price:.4f}\n**Time:** {entry_time_str}\n**Confirm:** {high_tf} Trend Aligned",
                            "color": 5763719 if sig_type == "LONG" else 15548997
                        }]
                    }
                    send_discord_alert(payload)
                    
                    # 2. Update Dedupe History
                    history[uid] = datetime.now().isoformat()
                    
                    # 3. Log to Dashboard JSON (oracle_signals.json)
                    signal_data = {
                        "Asset": name,
                        "Signal": sig_type, # LONG/SHORT
                        "Timeframe": low_tf,
                        "Entry_Price": p,
                        "Stop_Loss": sl_price,
                        "TP": tp_price,
                        "Entry_Time": entry_time_str,
                        "Confidence": "Oracle 🔮",
                        "Model": "Oracle",
                        "Method": "Fractal Alignment",
                        "RR": rr_mult,
                        "_sort_key": datetime.now().isoformat()
                    }
                    
                    # Append to file safely
                    ORACLE_LOG_FILE = "mango_oracle_signals.json"
                    existing_sigs = []
                    if os.path.exists(ORACLE_LOG_FILE):
                        try:
                            with open(ORACLE_LOG_FILE, "r") as f:
                                existing_sigs = json.load(f)
                        except: pass
                    
                    existing_sigs.append(signal_data)
                    # Keep last 100?
                    if len(existing_sigs) > 100: existing_sigs = existing_sigs[-100:]
                    
                    with open(ORACLE_LOG_FILE, "w") as f:
                        json.dump(existing_sigs, f, indent=4)
                        
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
            print("Sleeping 120s...")
            time.sleep(120)

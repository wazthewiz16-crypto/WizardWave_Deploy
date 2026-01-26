import time
import json
import os
import joblib
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import random
import urllib.request
from datetime import datetime, timedelta
import traceback

# Import Project Modules
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from strategy_cls import CLSRangeStrategy
from strategy_ichimoku import IchimokuStrategy
from strategy_wizard_pro import WizardWaveProStrategy
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

def format_asset_name(symbol):
    # Mapping
    mapping = {
        "DX-Y.NYB": "DXY 💵",
        "GC=F": "Gold 🟡",
        "CL=F": "Oil 🛢️",
        "SI=F": "Silver ⚪",
        "^GSPC": "SPX 📈",
        "^NDX": "NDX 💻",
        "^DJI": "DOW 🏭",
        "^AXJO": "AUS200 🇦🇺",
        "EURUSD=X": "EUR/USD 🇪🇺",
        "GBPUSD=X": "GBP/USD 🇬🇧",
        "AUDUSD=X": "AUD/USD 🇦🇺",
        "NZDUSD=X": "NZD/USD 🇳🇿",
        "USDCAD=X": "USD/CAD 🇨🇦",
        "USDCHF=X": "USD/CHF 🇨🇭",
        "USDJPY=X": "USD/JPY 🇯🇵"
    }
    if symbol in mapping: return mapping[symbol]
    
    # Generic Cleanup
    s = symbol.replace("=X", "")
    if s.endswith("USD") and len(s) == 6 and "/" not in s: 
         # FOREX
         return f"{s[:3]}/{s[3:]}"
    if s.endswith("USD") and "/" not in s:
         # Crypto likely e.g. BTCUSD
         return f"{s.replace('USD', '/USD')}"
         
    return s

def format_display_time(ts_str):
    try:
        dt = pd.to_datetime(ts_str)
        # Handle naive as UTC (yfinance standard) or imply UTC
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        
        dt_ny = dt.tz_convert('America/New_York')
        return dt_ny.strftime('%Y-%m-%d %I:%M %p EST')
    except:
        return ts_str

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
        asset_raw = str(signal_data.get('Asset', 'Unknown'))
        asset = format_asset_name(asset_raw)
        
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
        
        strategy_name = signal_data.get('Strategy', 'WizardWave')
        entry_time_display = format_display_time(signal_data.get('Entry_Time', 'N/A'))
        
        embed = {
            "title": f"🔮 {strategy_name.upper()}: {asset} {direction_str}",
            "description": f"**Direction:** {direction_str}\n**Timeframe:** {timeframe}\n**Action:** {action}",
            "color": color,
            "fields": [
                {"name": "Entry Price", "value": f"{entry_price}", "inline": True},
                {"name": "Take Profit", "value": f"{take_profit}", "inline": True},
                {"name": "Stop Loss", "value": f"{stop_loss}", "inline": True},
                {"name": "Confidence", "value": f"{confidence:.1f}%", "inline": True},
                {"name": "R:R", "value": rr_str, "inline": True},
                {"name": "Entry Time", "value": entry_time_display, "inline": False}
            ],
            "footer": {"text": f"{strategy_name} • Automated Signal (Monitor)"}
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
        # print("No Webhook URL found.")
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
                    entry_dt = entry_dt.tz_localize('America/New_York', ambiguous='infer') 
                
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
    loaded_models = {}
    models_config = CONFIG.get('models', {})
    
    for name, conf in models_config.items():
        try:
            if os.path.exists(conf['model_file']):
                loaded_models[name] = joblib.load(conf['model_file'])
            else:
                pass # print(f"Model file not found: {conf['model_file']}")
        except Exception as e:
            print(f"Error loading model {name}: {e}")

    if not loaded_models:
        print("No models loaded. Please ensure pipeline has been run.")
        return

    all_signals = []
    
    # Iterate through each model configuration
    for model_name, model_conf in models_config.items():
        strat_name = model_conf.get('strategy')
        model = loaded_models.get(model_name)
        is_rule_based = strat_name in ["IchimokuStrategy", "WizardWaveProStrategy"]
        
        if not model and not is_rule_based:
            continue
            
        timeframes = model_conf['timeframes']
        tb = model_conf.get('triple_barrier', {})
        conf_threshold = model_conf.get('confidence_threshold', 0.50)
        assets_to_scan = model_conf.get('assets_filter', CONFIG.get('assets', []))
        
        # Run Strategy & Features
        for tf in timeframes:
            # Fetch Macro Data (DXY & BTC) for this timeframe
            macro_df = None
            crypto_macro_df = None
            try:
                macro_df = fetch_data('DX-Y.NYB', asset_type='trad', timeframe=tf, limit=300)
                crypto_macro_df = fetch_data('BTC/USDT', asset_type='crypto', timeframe=tf, limit=300)
            except: pass

            for symbol in assets_to_scan:
                try:
                    # Asset Type
                    asset_type = get_asset_type(symbol)
                    fetch_type = 'trad' if asset_type == 'forex' or asset_type == 'trad' else 'crypto'
                    if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
                    
                    # FILTER: SKIP FOREX ON SWING TIMEFRAMES (Exception for Ichimoku)
                    # FILTER: SKIP FOREX ON SWING TIMEFRAMES (Exception for Ichimoku/Pro)
                    if asset_type == 'forex' and tf in ['4h', '12h', '1d', '4d'] and strat_name not in ["IchimokuStrategy", "WizardWaveProStrategy"]:
                        continue

                    # Fetch
                    df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=300)
                    if df.empty: continue
                    
                    # Strategy Factory
                    if strat_name == "WizardWave":
                        strat = WizardWaveStrategy()
                        df = strat.apply(df)
                    elif strat_name == "WizardWaveProStrategy":
                        params = model_conf.get('params', {})
                        strat = WizardWaveProStrategy(**params)
                        df = strat.apply(df)
                    elif strat_name == "IchimokuStrategy":
                        params = model_conf.get('params', {})
                        strat = IchimokuStrategy(**params)
                        df = strat.apply_strategy(df)
                    elif strat_name == "IchimokuProStrategy":
                        params = model_conf.get('params', {})
                        strat = IchimokuProStrategy(**params)
                        df = strat.apply_strategy(df)
                    else:
                        strat = WizardScalpStrategy(lookback=8)
                        df = strat.apply(df)
                    df = calculate_ml_features(df, macro_df=macro_df, crypto_macro_df=crypto_macro_df)
                    
                    # Predict / Signal Logic
                    if model:
                        df = calculate_ml_features(df, macro_df=macro_df, crypto_macro_df=crypto_macro_df)
                        
                        # Determine Dynamic Features from Model
                        if hasattr(model, 'feature_names_in_'):
                            features_list = list(model.feature_names_in_)
                        else:
                            features_list = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']

                        # Ensure all features exist
                        for f in features_list:
                            if f not in df.columns:
                                df[f] = 0.0
                                
                        # Sigma for Dynamic Barriers
                        crypto_use_dynamic = tb.get('crypto_use_dynamic', False)
                        if crypto_use_dynamic and asset_type == 'crypto':
                             df['sigma'] = df['close'].pct_change().ewm(span=36, adjust=False).std()
                             df['sigma'] = df['sigma'].bfill().fillna(0.01)
                        
                        df = df.dropna()
                        if df.empty: continue
                        
                        last_row = df.iloc[-1].copy()
                        
                        # Predict
                        X_new = pd.DataFrame([last_row[features_list]])
                        prob = model.predict_proba(X_new)[0][1]
                    else:
                        # Rule-based (e.g. Ichimoku)
                        if 'signal_type' in df.columns:
                            df_signals = df[df['signal_type'].notna() & (df['signal_type'] != 'NONE')]
                        else:
                            df_signals = pd.DataFrame()
                        
                        if df_signals.empty: continue
                        last_row = df_signals.iloc[-1].copy()
                        prob = 1.0 
                    
                    signal_type = last_row.get('signal_type', 'NONE')
                    if signal_type and signal_type != 'NONE':
                        price = last_row['close']
                        
                        # TP/SL Calculation
                        crypto_use_dynamic = tb.get('crypto_use_dynamic', False)
                        if 'LONG' in signal_type:
                            if crypto_use_dynamic and asset_type == 'crypto':
                                  sigma = last_row.get('sigma', 0.01)
                                  k_pt = tb.get('crypto_dyn_pt_k', 0.5)
                                  k_sl = tb.get('crypto_dyn_sl_k', 0.5)
                                  pt_pct = k_pt * sigma
                                  sl_pct = k_sl * sigma
                            else:
                                 sl_pct = tb.get('crypto_sl' if asset_type=='crypto' else ('forex_sl' if asset_type=='forex' else 'trad_sl'), 0.01)
                                 pt_pct = tb.get('crypto_pt' if asset_type=='crypto' else ('forex_pt' if asset_type=='forex' else 'trad_pt'), 0.02)
                                 
                            sl_price = price * (1 - sl_pct)
                            pt_price = price * (1 + pt_pct)
                        else: # SHORT
                            if crypto_use_dynamic and asset_type == 'crypto':
                                  sigma = last_row.get('sigma', 0.01)
                                  k_pt = tb.get('crypto_dyn_pt_k', 0.5)
                                  k_sl = tb.get('crypto_dyn_sl_k', 0.5)
                                  pt_pct = k_pt * sigma
                                  sl_pct = k_sl * sigma
                            else:
                                 sl_pct = tb.get('crypto_sl' if asset_type=='crypto' else ('forex_sl' if asset_type=='forex' else 'trad_sl'), 0.01)
                                 pt_pct = tb.get('crypto_pt' if asset_type=='crypto' else ('forex_pt' if asset_type=='forex' else 'trad_pt'), 0.02)

                            sl_price = price * (1 + sl_pct)
                            pt_price = price * (1 - pt_pct)

                        is_take = "✅ TAKE" if prob > conf_threshold else "⚠️ WAIT"
                        
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
                            "Stop_Loss": round(sl_price, 4),
                            "Strategy": f"{strat_name} ({tf})"
                        }
                        
                        all_signals.append(sig)

                except Exception as e:
                    pass

    # --- CLS STRATEGY SCAN ---
    print("Running Daily CLS Range Scan...")
    try:
        cls_strat = CLSRangeStrategy() 
        cls_config = models_config.get('1h_cls', {})
        cls_whitelist = cls_config.get('assets_filter', [])
        
        for symbol in CONFIG['assets']:
            try:
                asset_type = get_asset_type(symbol)
                is_tradfi = (asset_type == 'forex' or asset_type == 'trad') or ('-' in symbol or '^' in symbol or '=' in symbol)
                
                # Filter: Allow TradFi (Legacy) OR Whitelisted Crypto (BTC/ETH)
                is_whitelisted = symbol in cls_whitelist or symbol.replace("BINANCE:", "") in cls_whitelist
                
                if not (is_tradfi or is_whitelisted):
                    continue
                
                # Fetch MTF Data
                f_type = 'trad' if is_tradfi else 'crypto'
                
                df_htf = fetch_data(symbol, asset_type=f_type, timeframe='1d', limit=500)
                df_ltf = fetch_data(symbol, asset_type=f_type, timeframe='1h', limit=400)
                
                if df_htf.empty or df_ltf.empty: continue
                
                # Apply
                df = cls_strat.apply_mtf(df_htf, df_ltf)
                if df.empty or 'signal_type' not in df.columns: continue
                
                last = df.iloc[-1]
                s_type = last['signal_type']
                
                if isinstance(s_type, str) and "CLS" in s_type:
                     price = last['close']
                     tp = last['target_price']
                     sl = last['stop_loss']
                     
                     sig = {
                        "Asset": symbol,
                        "Timeframe": "1h", 
                        "Action": "✅ TAKE",
                        "Type": "LONG" if "LONG" in s_type else "SHORT",
                        "Signal": s_type,
                        "Entry_Price": price,
                        "Current_Price": price,
                        "Entry_Time": str(last.name),
                        "Confidence": "100.0%", 
                        "Confidence_Score": 100,
                        "Take_Profit": round(tp, 4) if pd.notna(tp) else 0,
                        "Stop_Loss": round(sl, 4) if pd.notna(sl) else 0,
                        "Strategy": "Daily CLS Range"
                    }
                     all_signals.append(sig)
                     print(f"CLS Signal Found: {symbol} {s_type}")

            except Exception as e:
                pass
    except Exception as e:
         print(f"CLS Init Error: {e}")

    # --- MANGO ORACLE INTEGRATION ---
    print("Checking Oracle Scraper Feed...")
    try:
        if os.path.exists("mango_oracle_signals.json"):
             with open("mango_oracle_signals.json", "r") as f:
                 oracle_sigs = json.load(f)
             
             # Load Cooldowns
             cooldown_file = "oracle_cooldowns.json"
             cooldowns = {}
             if os.path.exists(cooldown_file):
                 try:
                     with open(cooldown_file, "r") as f: cooldowns = json.load(f)
                 except: pass

             if oracle_sigs:
                 updated_cooldowns = False
                 now_ts = time.time()
                 
                 # TF to Seconds map
                 COOLDOWNS = {
                     "15M": 3600,       # 1 Hour
                     "1H": 14400,       # 4 Hours
                     "4H": 43200,       # 12 Hours
                     "1D": 86400,       # 24 Hours
                     "4D": 172800       # 48 Hours
                 }

                 for s in oracle_sigs:
                     # --- AGE CHECK (Fix Duplicate/Old Alert Spam) ---
                     ts_str = s.get('Timestamp')
                     if ts_str:
                          try:
                             # Timestamp is EST (UTC-5) from Scraper
                             s_dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
                             
                             # Current EST Time
                             now_est = datetime.utcnow() - timedelta(hours=5)
                             
                             # Calculate Age
                             age_seconds = (now_est - s_dt).total_seconds()
                             
                             # Filter out signals older than 4 hours (Generous buffer for 4H/Daily, tight enough for 15m)
                             # If it's a "Daily" signal, it might be valid for 24h, but we only want to ALERT when it's NEW.
                             # If scraper keeps it "Sticky", timestamp remains old.
                             # So if timestamp is > 4h old, we assume we already alerted or missed the boat.
                             if age_seconds > 14400: 
                                 # print(f"Skipping Old Signal: {s['Asset']} {age_seconds}s old")
                                 continue
                          except Exception as e: 
                             print(f"Time Parse Error: {e}")
                             pass

                     # Cooldown Key
                     c_key = f"{s['Asset']}_{s['Timeframe']}_{s['Type']}"
                     last_sent = cooldowns.get(c_key, 0)
                     
                     # Use timeframe-specific limits
                     limit = COOLDOWNS.get(s['Timeframe'].upper(), 3600)
                     
                     # Check Eligibility (Time.time() is system agnostic, used for relative cooldown)
                     if (now_ts - last_sent) < limit:
                         continue
                     
                     # Update Cooldown
                     cooldowns[c_key] = now_ts
                     updated_cooldowns = True
                     
                     price = s.get('Price', 0.0)
                     if price is None: price = 0.0
                     sig = {
                        "Asset": s['Asset'],
                        "Timeframe": s['Timeframe'],
                        "Action": "✅ TAKE",
                        "Type": s['Type'],
                        "Signal": f"Dynamic {s['Type'].title()}", # e.g. Dynamic Long
                        "Entry_Price": price,
                        "Current_Price": price,
                        "Entry_Time": s.get('Timestamp', datetime.now().isoformat()),
                        "Confidence": "Indicator", # String for display
                        "Confidence_Score": 100.0, # High prio
                        "Take_Profit": 0.0, # Dynamic exit
                        "Stop_Loss": s.get('Stop_Loss', 0.0),
                        "Strategy": "Mango Oracle 🔮"
                     }
                     all_signals.append(sig)
                     print(f"Added NEW Oracle Signal: {s['Asset']} {s['Timeframe']}")
                 
                 if updated_cooldowns:
                     with open(cooldown_file, "w") as f:
                         json.dump(cooldowns, f)
             else:
                 print("Oracle feed empty.")
    except Exception as e:
        print(f"Oracle Feed Error: {e}")

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

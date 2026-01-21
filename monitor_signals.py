import time
import json
import os
import joblib
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import random
import urllib.request
from datetime import datetime
import traceback

# Import Project Modules
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
from strategy_cls import CLSRangeStrategy
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
        if model_name not in loaded_models:
            continue
            
        model = loaded_models[model_name]
        timeframes = model_conf['timeframes']
        strat_name = model_conf['strategy']
        tb = model_conf['triple_barrier']
        conf_threshold = model_conf.get('confidence_threshold', 0.50)
        
        # Run Strategy & Features
        for tf in timeframes:
            # Fetch Macro Data (DXY & BTC) for this timeframe
            macro_df = None
            crypto_macro_df = None
            try:
                macro_df = fetch_data('DX-Y.NYB', asset_type='trad', timeframe=tf, limit=300)
                crypto_macro_df = fetch_data('BTC/USDT', asset_type='crypto', timeframe=tf, limit=300)
            except: pass

            for symbol in CONFIG['assets']:
                try:
                    # Determine Dynamic Features from Model
                    if hasattr(model, 'feature_names_in_'):
                        features_list = list(model.feature_names_in_)
                    else:
                        features_list = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
                    
                    # Asset Type
                    asset_type = get_asset_type(symbol)
                    fetch_type = 'trad' if asset_type == 'forex' or asset_type == 'trad' else 'crypto'
                    if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
                    
                    # --- FILTER: SKIP FOREX ON SWING TIMEFRAMES ---
                    if asset_type == 'forex' and tf in ['4h', '12h', '1d', '4d']:
                        continue

                    # Fetch
                    df = fetch_data(symbol, asset_type=fetch_type, timeframe=tf, limit=300)
                    if df.empty: continue
                    
                    # Strategy
                    if strat_name == "WizardWave":
                        strat = WizardWaveStrategy()
                    else:
                        strat = WizardScalpStrategy(lookback=8)
                    
                    df = strat.apply(df)
                    df = calculate_ml_features(df, macro_df=macro_df, crypto_macro_df=crypto_macro_df)
                    
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
                    
                    signal_type = last_row['signal_type']
                    if signal_type != 'NONE':
                        price = last_row['close']
                        
                        # TP/SL Calculation
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
                                sl_pct = tb['crypto_sl'] if asset_type=='crypto' else tb.get('forex_sl' if asset_type=='forex' else 'trad_sl', 0.01)
                                pt_pct = tb['crypto_pt'] if asset_type=='crypto' else tb.get('forex_pt' if asset_type=='forex' else 'trad_pt', 0.02)

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
                            "Stop_Loss": round(sl_price, 4)
                        }
                        
                        all_signals.append(sig)

                except Exception as e:
                    pass

    # --- CLS STRATEGY SCAN (TradFi Only) ---
    print("Running Daily CLS Range Scan...")
    try:
        cls_strat = CLSRangeStrategy() 
        
        for symbol in CONFIG['assets']:
            try:
                # Filter TradFi
                asset_type = get_asset_type(symbol)
                is_tradfi = False
                if asset_type == 'forex' or asset_type == 'trad': is_tradfi = True
                elif '-' in symbol or '^' in symbol or '=' in symbol: is_tradfi = True
                
                if not is_tradfi: continue
                
                # Fetch MTF Data
                df_htf = fetch_data(symbol, asset_type='trad', timeframe='1d', limit=500)
                df_ltf = fetch_data(symbol, asset_type='trad', timeframe='1h', limit=400)
                
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

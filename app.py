import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import concurrent.futures
import threading
import warnings
from st_copy_to_clipboard import st_copy_to_clipboard
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ResourceWarning)
pd.set_option('future.no_silent_downcasting', True)

from src.strategies import manage_trades
import subprocess
import sys
import os

# --- AUTO-START BACKGROUND MONITOR ---
def ensure_monitor_running():
    """Starts the separate monitor_signals.py script if not already running."""
    pid_file = os.path.join("data", "monitor.pid")
    
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            # Check if process is running (Windows/Unix compatible check)
            os.kill(pid, 0) 
            # If we get here, it is running
            return 
        except (OSError, ValueError):
            # Process dead or file corrupt
            pass
    
    # Start the Monitor
    try:
        # Windows: CREATE_NO_WINDOW = 0x08000000 to avoid popup
        # This allows it to run silently in the background
        creation_flags = 0x08000000 if os.name == 'nt' else 0
        
        process = subprocess.Popen(
            [sys.executable, os.path.join("src", "core", "monitor_signals.py")],
            cwd=os.getcwd(),
            creationflags=creation_flags
        )
        
        with open(pid_file, "w") as f:
            f.write(str(process.pid))
            
        print(f"Started Background Monitor (PID: {process.pid})")
        
    except Exception as e:
        print(f"Failed to auto-start monitor: {e}")

# Run the check immediately on app load
ensure_monitor_running()

# --- Thread Manager for Background Fetching ---
class GlobalRunManager:
    def __init__(self):
        self.is_running = False
        self.progress = 0
        self.lock = threading.Lock()
        self.result_container = {} # keys: 'data', 'ready', 'message'

    def start_run(self):
        with self.lock:
            if self.is_running:
                return False
            self.is_running = True
            self.progress = 0
            self.result_container = {'ready': False, 'running': True, 'progress': 0}
            return True

    def update_progress(self, val):
        with self.lock:
            self.progress = val
            self.result_container['progress'] = val

    def finish_run(self, result_data):
        with self.lock:
            self.is_running = False
            self.progress = 100
            self.result_container = {
                'ready': True, 
                'running': False, 
                'progress': 100,
                'data': result_data
            }
            
    def get_status(self):
        with self.lock:
            return self.result_container.copy()

    def ack_result(self):
        with self.lock:
            self.result_container['ready'] = False

# Initialize Global Manager
if 'THREAD_MANAGER' not in st.session_state:
    st.session_state.THREAD_MANAGER = GlobalRunManager()
# Also use a module-level global for persistency if session state fails across thread boundary (it shouldn't if defined here but let's stick to session state if possible or global)
# Streamlit re-runs module, so global resets? 
# Correct: Globals reset on re-run unless st.cache_resource is used.
@st.cache_resource
def get_thread_manager():
    return GlobalRunManager()

thread_manager = get_thread_manager()

def run_runic_analysis():
    """Background worker function"""
    try:
        # Run All Timeframes
        # We pass silent=True to avoid UI calls
        r15m, a15m, h15m = analyze_timeframe("15 Minutes", silent=True)
        thread_manager.update_progress(16)
        r1h, a1h, h1h = analyze_timeframe("1 Hour", silent=True)
        thread_manager.update_progress(32)
        r4h, a4h, h4h = analyze_timeframe("4 Hours", silent=True)
        thread_manager.update_progress(48)
        r12h, a12h, h12h = analyze_timeframe("12 Hours", silent=True)
        thread_manager.update_progress(64)
        r1d, a1d, h1d = analyze_timeframe("1 Day", silent=True)
        thread_manager.update_progress(80)
        r4d, a4d, h4d = analyze_timeframe("4 Days", silent=True)
        thread_manager.update_progress(95)
        
        # --- CLS Strategy Scan ---
        _, a_cls, h_cls = analyze_cls_strategy(silent=True)
        thread_manager.update_progress(98)
        
        # --- Ichimoku Strategy Scan ---
        _, a_ichi, h_ichi = analyze_ichimoku_strategy(silent=True)
        thread_manager.update_progress(99)
        
        # Aggregate History
        all_history = []
        if h15m: all_history.extend(h15m)
        if h1h: all_history.extend(h1h)
        if h4h: all_history.extend(h4h)
        if h12h: all_history.extend(h12h)
        if h1d: all_history.extend(h1d)
        if h4d: all_history.extend(h4d)
        if h_cls: all_history.extend(h_cls)
        if h_ichi: all_history.extend(h_ichi)
        
        history_df = pd.DataFrame()
        if all_history:
             history_df = pd.DataFrame(all_history)
        
        # Aggregate Active
        active_dfs = [df for df in [a15m, a1h, a4h, a12h, a1d, a4d, a_cls, a_ichi] if df is not None and not df.empty]
        combined_active = pd.DataFrame()
        if active_dfs:
            combined_active = pd.concat(active_dfs).sort_values(by='_sort_key', ascending=False)
            
        # Process Discord (Side Effect - OK in thread? Yes, usually I/O)
        # DISABLE IN APP: Monitor Script handles discordant alerts to avoid duplicates
        # if not combined_active.empty:
        #    process_discord_alerts(combined_active)
            
        # Metrics Calculation
        calc_24h = 0.0
        calc_12h = 0.0
        try:
           if not history_df.empty and '_sort_key' in history_df.columns:
               history_df['_sort_key'] = pd.to_datetime(history_df['_sort_key'], utc=True)
               now_utc = pd.Timestamp.now(tz='UTC')
               recent_sigs_24 = history_df[history_df['_sort_key'] >= (now_utc - pd.Timedelta(hours=24))]
               recent_sigs_12 = history_df[history_df['_sort_key'] >= (now_utc - pd.Timedelta(hours=12))]
               calc_24h = recent_sigs_24['Return_Pct'].sum()
               calc_12h = recent_sigs_12['Return_Pct'].sum()
        except: pass

        # Package Result
        result = {
            'history': history_df,
            'active': combined_active,
            'metrics': (calc_24h, calc_12h),
            'timestamp': time.time()
        }
        
        # Finish
        thread_manager.finish_run(result)
        
    except Exception as e:
        print(f"Background Fetch Failed: {e}")
        import traceback
        traceback.print_exc()
        thread_manager.finish_run(None)

import joblib
from src.core.data_fetcher import fetch_data
from src.core.feature_engine import calculate_ml_features, calculate_ichi_features, calculate_cls_features
from src.strategies.strategy import WizardWaveStrategy
from src.strategies.strategy_scalp import WizardScalpStrategy
from src.strategies.strategy_cls import CLSRangeStrategy
from src.strategies.strategy_ichimoku import IchimokuStrategy
from src.utils.paths import get_model_path

# Load ML Models for New Strats
try:
    ICHI_MODEL = joblib.load(get_model_path("model_ichimoku.pkl"))
    # (Assuming features_ichimoku.json is also in a data/config path if we wanted, 
    # but let's keep it simple for now or check paths)
    with open("features_ichimoku.json", "r") as f:
        ICHI_FEATS = json.load(f)
except:
    ICHI_MODEL = None
    ICHI_FEATS = []
    
try:
    CLS_MODEL = joblib.load(get_model_path("model_cls.pkl"))
    with open("features_cls.json", "r") as f:
        CLS_FEATS = json.load(f)
except:
    CLS_MODEL = None
    CLS_FEATS = []
import streamlit.components.v1 as components
import json
import urllib.request
import os
from datetime import datetime, date


# --- Persistence Logic ---
STATE_FILE = os.path.join("data", "user_grimoire.json")

# --- Cloud Bootstrap (One-time) ---
@st.cache_resource
def bootstrap_system():
    import subprocess
    import sys
    import os
    
    print("[*] Performing System Bootstrap...")
    
    # 1. Check for Playwright
    try:
        import playwright
        # Check if browser binaries likely exist in common locations
        cache_dirs = [
            os.path.expanduser("~/.cache/ms-playwright"),
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium"
        ]
        
        if not any(os.path.exists(d) for d in cache_dirs):
             print("[!] Installing browser binaries for Playwright...")
             # Redirect output to /dev/null if it causes issues, or keep it for logs
             subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], capture_output=True)
    except ImportError:
        print("[!] Playwright library missing from environment.")
    except Exception as e:
        print(f"[!] Bootstrap warning: {e}")

    # 2. Auto-start Scraper if not running
    # Use a more reliable background process check
    lock_file = os.path.join(os.getcwd(), "data", "scraper.lock")
    scraper_script = "scrape_tv_indicators.py"
    
    if os.path.exists(scraper_script) and not os.path.exists(lock_file):
        try:
            # Ensure data dir exists
            os.makedirs("data", exist_ok=True)
            
            with open(lock_file, "w") as f:
                f.write(str(os.getpid()))
            
            print(f"[*] Starting Background Scraper...")
            # Use proper background logic for both OS
            if os.name == 'nt':
                 subprocess.Popen([sys.executable, scraper_script], creationflags=0x08000000)
            else:
                 # Standard Unix background
                 subprocess.Popen([sys.executable, scraper_script], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[!] Scraper start failed: {e}")
            if os.path.exists(lock_file): os.remove(lock_file)
            
    return True

# Call bootstrap later or here? Let's keep it here but with more safety.
try:
    bootstrap_system()
except:
    pass

# Load Strategy Config
config_file = os.path.join('config', 'strategy_config.json')
if os.path.exists(os.path.join('config', 'strategy_config_experimental.json')):
    config_file = os.path.join('config', 'strategy_config_experimental.json')

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except Exception as e:
    print(f"Error loading {config_file}: {e}")
    config = {
        "models": {
            "1d": {"triple_barrier": {"time_limit_bars": 21, "crypto_pt": 0.09, "crypto_sl": 0.033, "trad_pt": 0.04, "trad_sl": 0.02, "forex_pt": 0.03, "forex_sl": 0.015}},
            "4h": {"triple_barrier": {"time_limit_bars": 12, "crypto_pt": 0.04, "crypto_sl": 0.02, "trad_pt": 0.02, "trad_sl": 0.01, "forex_pt": 0.01, "forex_sl": 0.005}},
            "1h": {"triple_barrier": {"time_limit_bars": 24, "crypto_pt": 0.015, "crypto_sl": 0.01, "trad_pt": 0.01, "trad_sl": 0.005, "forex_pt": 0.005, "forex_sl": 0.0025}},
            "15m": {"triple_barrier": {"time_limit_bars": 12, "crypto_pt": 0.015, "crypto_sl": 0.0075, "trad_pt": 0.005, "trad_sl": 0.0025, "forex_pt": 0.0025, "forex_sl": 0.0015}},
        }
    } # Fallback

def load_grimoire():
    today = date.today()
    current_week = today.isocalendar()[1]
    
    # Defaults
    default_state = {
        'mana': 425,
        'spells_day': 2,
        'spells_week': 5,
        'last_date': str(today),
        'last_week': current_week
    }
    
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                saved_data = json.load(f)
                
            # Check Daily Reset
            last_date_str = saved_data.get('last_date', '')
            last_week_saved = saved_data.get('last_week', current_week)
            
            today_str = str(today)
            
            # 1. Daily Reset Check
            if last_date_str != today_str:
                saved_data['mana'] = 425 # Daily Reset
                saved_data['spells_day'] = 2 # Daily Reset
                saved_data['last_date'] = today_str
                
            # 2. Weekly Reset Check (New Week)
            if last_week_saved != current_week:
                saved_data['spells_week'] = 5 # Weekly Reset
                saved_data['last_week'] = current_week
                
            # Save if any resets happened
            if last_date_str != today_str or last_week_saved != current_week:
                save_grimoire(saved_data)
                
            return saved_data
        except Exception as e:
            print(f"Error loading grimoire: {e}")
            return default_state
            
    return default_state

def save_grimoire(data):
    try:
        data['last_date'] = str(date.today())
        data['last_week'] = date.today().isocalendar()[1]
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving grimoire: {e}")

# Page Config
st.set_page_config(
    page_title="Arcane Portal",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TRADING PLAN SIDEBAR ---
with st.sidebar:
    st.markdown("### 📜 My Trading Plan")
    st.info("Define your rules to streamline your process.")
    
    # 1. Filters
    st.caption("Auto-Filtering")
    plan_active = st.checkbox("Apply Plan Filters", value=False, help="Automatically filter Runic Alerts & History based on rules.")
    
    min_conf_plan = st.slider("Min Confidence", 0, 100, 55, step=5, key="plan_slider_conf")
    
    # Map common TFs
    tf_opts = ["15m", "1H", "4H", "12H", "1D", "4D"]
    allowed_tfs_plan = st.multiselect(
        "Allowed Timeframes", 
        tf_opts,
        default=["4H", "1D", "4D"],
        help="Select timeframes you want to focus on.",
        key="plan_multiselect_tf"
    )

    # Store in Session State for Global Access
    st.session_state['plan_active'] = plan_active
    st.session_state['plan_min_conf'] = min_conf_plan
    st.session_state['plan_allowed_tfs'] = allowed_tfs_plan

# Custom CSS for styling
st.markdown("""
<style>
    .stDataFrame {
        font-size: 1.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)
# Application Header
# st.title("🧙‍♂️ Wizard Wave Signals")

# Strategy Settings (Hardcoded)
lookback = 29
sensitivity = 1.06
cloud_spread = 0.64
zone_pad = 1.5

# Auto Refresh Enabled by Default
# auto_refresh = True

# Assets List
ASSETS = [
    {"symbol": "BTC/USDT", "type": "crypto", "name": "Bitcoin"},
    {"symbol": "ETH/USDT", "type": "crypto", "name": "Ethereum"},
    {"symbol": "SOL/USDT", "type": "crypto", "name": "Solana"},
    {"symbol": "DOGE/USDT", "type": "crypto", "name": "Dogecoin"},
    {"symbol": "XRP/USDT", "type": "crypto", "name": "XRP"},
    {"symbol": "BNB/USDT", "type": "crypto", "name": "BNB"},
    {"symbol": "LINK/USDT", "type": "crypto", "name": "Chainlink"},
    {"symbol": "^NDX", "type": "trad", "name": "Nasdaq 100"},
    {"symbol": "^GSPC", "type": "trad", "name": "S&P 500"},
    {"symbol": "^AXJO", "type": "trad", "name": "AUS 200"},
    {"symbol": "DX-Y.NYB", "type": "trad", "name": "DXY Index"},
    {"symbol": "GC=F", "type": "trad", "name": "Gold Futures"},
    {"symbol": "CL=F", "type": "trad", "name": "US Oil"},
    {"symbol": "EURUSD=X", "type": "forex", "name": "EUR/USD"},
    {"symbol": "GBPUSD=X", "type": "forex", "name": "GBP/USD"},
    {"symbol": "AUDUSD=X", "type": "forex", "name": "AUD/USD"},
    {"symbol": "SI=F", "type": "trad", "name": "Silver Futures"},
    {"symbol": "ARB/USDT", "type": "crypto", "name": "Arbitrum"},
    {"symbol": "AVAX/USDT", "type": "crypto", "name": "Avalanche"},
    {"symbol": "ADA/USDT", "type": "crypto", "name": "Cardano"},
]

# Initialize Session State
if 'processed_signals' not in st.session_state:
    st.session_state['processed_signals'] = set()

# Arcane Portal State
# Arcane Portal State
if 'mana' not in st.session_state:
    # Load from persistent storage
    grimoire = load_grimoire()
    st.session_state['mana'] = grimoire.get('mana', 425)
    st.session_state['spells_day'] = grimoire.get('spells_day', 2)
    st.session_state['spells_week'] = grimoire.get('spells_week', 5)
    st.session_state['last_date'] = grimoire.get('last_date', str(date.today()))
    st.session_state['last_week'] = grimoire.get('last_week', date.today().isocalendar()[1])
    st.session_state['last_reset_week'] = pd.Timestamp.now().to_period('W').start_time

# --- Live Reset Check (For active sessions crossing midnight) ---
today = date.today()
today_str = str(today)
current_week = today.isocalendar()[1]
state_needs_update = False

# Ensure keys exist (migration for existing sessions)
if 'last_date' not in st.session_state:
    st.session_state['last_date'] = today_str
if 'last_week' not in st.session_state:
    st.session_state['last_week'] = current_week

# Check Day
if st.session_state['last_date'] != today_str:
    st.session_state['mana'] = 425
    st.session_state['spells_day'] = 2
    st.session_state['last_date'] = today_str
    state_needs_update = True

# Check Week
if st.session_state['last_week'] != current_week:
    st.session_state['spells_week'] = 5
    st.session_state['last_week'] = current_week
    state_needs_update = True

if state_needs_update:
    # Load current file content to preserve other keys if any
    save_data = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                save_data = json.load(f)
        except:
             pass
             
    # Update with new values
    save_data.update({
        'mana': st.session_state['mana'],
        'spells_day': st.session_state['spells_day'],
        'spells_week': st.session_state['spells_week']
    })
    
    save_grimoire(save_data)
    
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# --- ML Model Integration ---
@st.cache_resource(ttl=3600) # Add TTL to prevent stale models
def load_ml_models_v2():
    """Load all 6 specific timeframe models using filenames from config"""
    models = {}
    model_keys = ["4d", "1d", "12h", "4h", "1h", "15m"]
    
    for key in model_keys:
        try:
            filename = config.get('models', {}).get(key, {}).get('model_file', f"model_{key}.pkl")
            if os.path.exists(filename):
                loaded_obj = joblib.load(filename)
                # Handle wrapped models
                if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
                    models[key] = loaded_obj['model']
                else:
                    models[key] = loaded_obj
                print(f"Successfully loaded model for {key} from {filename}")
            else:
                print(f"Model file {filename} not found.")
                models[key] = None
        except Exception as e:
            print(f"Error loading {key} model: {e}")
            models[key] = None
        
    return models

# --- Utility: TradingView Symbol Mapping ---
def get_tv_symbol(asset_entry):
    """Maps internal symbol to TradingView symbol"""
    s = asset_entry.get('symbol', '')
    
    # Crypto (Binance is a safe default for USDT pairs)
    if "USDT" in s:
        clean = s.replace("/", "")
        return f"BINANCE:{clean}"
    
    # Forex
    if "=X" in s:
        clean = s.replace("=X", "")
        return f"FX:{clean}"
    
    # Futures / Indices (Heuristics)
    # Use OANDA/CAPITALCOM CFDs for maximum widget compatibility
    if s == "^NDX": return "OANDA:NAS100USD"
    if s == "^GSPC": return "OANDA:SPX500USD"
    if s == "^AXJO": return "OANDA:AU200AUD"
    if s == "DX-Y.NYB": return "CAPITALCOM:DXY" # DXY CFD
    if s == "GC=F": return "OANDA:XAUUSD" # Gold Spot (More reliable than TVC:GOLD)
    if s == "CL=F": return "OANDA:WTICOUSD" # WTI Oil
    if s == "SI=F": return "OANDA:XAGUSD" # Silver Spot
    
    return f"COINBASE:BTCUSD" # Fallback

# Helper for Timeframe Mapping
def get_tv_interval(tf_label):
    # TV uses minutes or 'D', 'W'
    # "15 Minutes", "30 Minutes", "1 Hour", "4 Hours", "1 Day", "4 Days"
    # Also handles short codes: 15m, 1H, 4H, 12H, 1D, 4D
    if "15" in tf_label: return "15"
    if "1 Hour" in tf_label or "1H" in tf_label: return "60"
    if "4 Hour" in tf_label or "4H" in tf_label: return "240"
    if "12 Hour" in tf_label or "12H" in tf_label: return "720"
    if "1 Day" in tf_label or "1D" in tf_label: return "D"
    if "4 Day" in tf_label or "4D" in tf_label: return "240" # Fallback
    return "60"

# Initialize Active Symbol/Interval State
if 'active_tv_symbol' not in st.session_state:
    st.session_state.active_tv_symbol = "COINBASE:BTCUSD"
if 'active_tv_interval' not in st.session_state:
    st.session_state.active_tv_interval = "60"

# Initialize Active Symbol State
if 'active_tv_symbol' not in st.session_state:
    st.session_state.active_tv_symbol = "COINBASE:BTCUSD"


models = load_ml_models_v2()

def analyze_timeframe(timeframe_label, silent=False):
    results = []
    active_trades = []
    historical_signals = [] # Store all historical signals found
    
    # Map label to model code
    tf_map_code = {
        "15 Minutes": "15m",
        "15m": "15m",
        "1 Hour": "1h",
        "1H": "1h",
        "4 Hours": "4h",
        "4H": "4h",
        "12 Hours": "12h",
        "12H": "12h",
        "1 Day": "1d",
        "1D": "1d",
        "4 Days": "4d",
        "4D": "4d"
    }
    tf_code = tf_map_code.get(timeframe_label, "1d")

    # Shorten Timeframe Label for display
    tf_display_map = {
        "15 Minutes": "15m",
        "1 Hour": "1H",
        "4 Hours": "4H",
        "12 Hours": "12H",
        "1 Day": "1D",
        "4 Days": "4D"
    }
    short_tf = tf_display_map.get(timeframe_label, timeframe_label)

    # Get Config for this model
    model_config = config.get('models', {}).get(tf_code, {})
    if not model_config:
        if not silent: st.error(f"Configuration missing for {tf_code}")
        return None, None, None

    # Determine Strategy
    strat_name = model_config.get('strategy', 'WizardWave')
    
    if strat_name == 'WizardScalp':
        strat = WizardScalpStrategy(lookback=8, sensitivity=1.0)
    else:
        strat = WizardWaveStrategy(
            lookback=lookback, 
            sensitivity=sensitivity, 
            cloud_spread=cloud_spread, 
            zone_pad_pct=zone_pad 
        )
        
    model = models.get(tf_code)
    
    # Triple Barrier Config
    tb = model_config.get('triple_barrier', {})
    
    tp_crypto = tb.get('crypto_pt', 0.02)
    sl_crypto = tb.get('crypto_sl', 0.01)
    tp_trad = tb.get('trad_pt', 0.01)
    sl_trad = tb.get('trad_sl', 0.005)
    tp_forex = tb.get('forex_pt', 0.005)
    sl_forex = tb.get('forex_sl', 0.005)

    crypto_use_dynamic = tb.get('crypto_use_dynamic', False)
    crypto_dyn_pt_k = tb.get('crypto_dyn_pt_k', 0.5)
    crypto_dyn_sl_k = tb.get('crypto_dyn_sl_k', 0.5)
    
    threshold = model_config.get('confidence_threshold', 0.50)

    progress_bar = st.progress(0) if not silent else None
    status_text = st.empty() if not silent else None
    
    if not silent and status_text:
        status_text.text(f"[{timeframe_label}] Fetching data for {len(ASSETS)} assets...")

    # --- Macro Integration (DXY & BTC) ---
    macro_df = None
    crypto_macro_df = None
    try:
        macro_df = fetch_data('DX-Y.NYB', 'trad', timeframe=tf_code, limit=300)
        crypto_macro_df = fetch_data('BTC/USDT', 'crypto', timeframe=tf_code, limit=300)
    except: pass

    def log_debug(msg):
        try:
            with open("debug_signal_log.txt", "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()} - {msg}\n")
        except: pass

    def process_asset(asset):
        try:
            # log_debug(f"Processing {asset['symbol']} for {tf_code}")
            
            # Dynamic Limit
            current_limit = 1000
            if "Day" in timeframe_label:
                current_limit = 300 

            # --- FILTER: SKIP FOREX ON SWING TIMEFRAMES ---
            # User Request: Remove Forex swing signals (poor performance)
            if asset['type'] == 'forex' and tf_code in ['4h', '12h', '1d', '4d']:
                 log_debug(f"Skipping Forex on Swing TF: {asset['symbol']} {tf_code}")
                 return None, None, None
            
            # Fetch Data
            df = fetch_data(asset['symbol'], asset['type'], timeframe=tf_code, limit=current_limit)
            
            if df.empty:
                log_debug(f"EMPTY DATA for {asset['symbol']} {tf_code}")
                return None, None, None
            
            # Apply Strategy
            df_strat = strat.apply(df)
            
            # ML Features & Prediction
            df_strat = calculate_ml_features(df_strat, macro_df=macro_df, crypto_macro_df=crypto_macro_df)
            
            # --- Calculate Sigma for Dynamic Barriers ---
            if crypto_use_dynamic and asset['type'] == 'crypto':
                df_strat['sigma'] = df_strat['close'].pct_change().ewm(span=36, adjust=False).std()
                df_strat['sigma'] = df_strat['sigma'].bfill().fillna(0.01)
            
            prob = 0.0
            if model:
                # Standard Features List
                # Robustness: Use model's expected features if available to prevent crashes (rvol mismatch)
                if hasattr(model, 'feature_names_in_'):
                    features_list = list(model.feature_names_in_)
                else:
                    # Fallback for older sklearn versions or models without metadata
                    features_list = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'rvol', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi']
                
                # Check for feature columns presence
                missing_feats = [f for f in features_list if f not in df_strat.columns]
                if missing_feats:
                    log_debug(f"Missing features {asset['symbol']}: {missing_feats}")
                    for f in missing_feats: df_strat[f] = 0

                # Predict for CURRENT candle
                last_features = df_strat.iloc[[-1]][features_list]
                prob = model.predict_proba(last_features)[0][1] # Prob of Class 1 (Good)
                
                # Predict for ALL rows (for history)
                # Initialize column with 0.0 first to handle dropped NaNs
                df_strat['model_prob'] = 0.0
                
                df_clean = df_strat.dropna()
                if not df_clean.empty:
                    all_probs = model.predict_proba(df_clean[features_list])[:, 1]
                    df_strat.loc[df_clean.index, 'model_prob'] = all_probs
            
                # Explicitly set probability for the last row
                if not df_strat.empty:
                     df_strat.loc[df_strat.index[-1], 'model_prob'] = prob
                     # log_debug(f"{asset['symbol']} {tf_code} PROB: {prob:.4f} (Thresh: {threshold})")
            else:
                prob = 0.0
                df_strat['model_prob'] = 0.0
                log_debug(f"NO MODEL LOADED for {tf_code}")

            # --- Collect Historical Signals & Simulate PnL ---
            
            # Helper to run stateful simulation on the history
            def simulate_history_stateful(df, asset_type, threshold_val=0.40):
               trades = []
               position = None
               entry_price = 0.0
               entry_time = None
               entry_conf = 0.0
               sl_price = 0.0
               
               if asset_type == 'crypto':
                   if crypto_use_dynamic:
                       curr_tp_pct = tp_crypto 
                       curr_sl_pct = sl_crypto
                   else:
                       curr_tp_pct = tp_crypto
                       curr_sl_pct = sl_crypto
               elif asset_type == 'forex':
                   curr_tp_pct = tp_forex
                   curr_sl_pct = sl_forex
               else:
                   curr_tp_pct = tp_trad
                   curr_sl_pct = sl_trad
               
               # Iterate through all bars
               for idx, row in df.iterrows():
                   close = row['close']
                   high = row['high']
                   low = row['low']
                   signal = row['signal_type']
                   model_prob = row.get('model_prob', 0.0)
                   int_idx = df.index.get_loc(idx)
                   
                   # --- EXIT LOGIC ---
                   exit_trade = False
                   pnl = 0.0
                   status = ""
                   
                   if position == 'LONG':
                       # Check TP
                       if high >= entry_price * (1 + curr_tp_pct):
                           pnl = curr_tp_pct
                           status = "HIT TP 🟢"
                           exit_trade = True
                       # Check SL
                       elif low <= entry_price * (1 - curr_sl_pct):
                           pnl = -curr_sl_pct
                           status = "HIT SL 🔴"
                           exit_trade = True

                   elif position == 'SHORT':
                       # Check TP
                       if low <= entry_price * (1 - curr_tp_pct):
                           pnl = curr_tp_pct
                           status = "HIT TP 🟢"
                           exit_trade = True
                       # Check SL
                       elif high >= entry_price * (1 + curr_sl_pct):
                           pnl = -curr_sl_pct
                           status = "HIT SL 🔴"
                           exit_trade = True
                           
                   if exit_trade:
                       trades.append({
                            "_sort_key": entry_time,
                            "Asset": asset['name'],
                            "Timeframe": short_tf,
                            "Time": format_time(entry_time),
                            "Exit Time": format_time(idx),
                            "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                            "Price": entry_price,
                            "Confidence": f"{entry_conf:.0%}",
                            "Model": "✅",
                            "Return_Pct": pnl - 0.002, # Deduct 0.2% fee
                            "SL_Pct": curr_sl_pct,
                            "Status": status
                       })
                       position = None

                   # --- ENTRY & REVERSAL LOGIC ---
                   # Only take filtered signals
                   if model_prob > threshold_val:
                       new_pos = None
                       if 'LONG_ZONE' in signal or 'LONG_REV' in signal or 'SCALP_LONG' in signal:
                           new_pos = 'LONG'
                       elif 'SHORT_ZONE' in signal or 'SHORT_REV' in signal or 'SCALP_SHORT' in signal:
                           new_pos = 'SHORT'
                           
                       if new_pos:
                           if position is not None and position != new_pos:
                               last_close_pnl = 0.0
                               if position == 'LONG':
                                   last_close_pnl = (close - entry_price) / entry_price
                               else:
                                   last_close_pnl = (entry_price - close) / entry_price
                                   
                               status_label = "FLIP 🔄"
                               if last_close_pnl < -curr_sl_pct:
                                   last_close_pnl = -curr_sl_pct
                                   status_label = "HIT SL 🔴"
                                   
                               trades.append({
                                    "_sort_key": entry_time,
                                    "Asset": asset['name'],
                                    "Timeframe": short_tf,
                                    "Time": format_time(entry_time),
                                    "Exit Time": format_time(idx),
                                    "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                                    "Price": entry_price,
                                    "Confidence": f"{entry_conf:.0%}",
                                    "Model": "✅",
                                    "Return_Pct": last_close_pnl - 0.002, # Deduct 0.2% fee
                                    "SL_Pct": curr_sl_pct,
                                    "Status": status_label
                               })
                               position = None 
                           
                           if position is None:
                               position = new_pos
                               entry_price = close
                               entry_time = idx
                               entry_int_idx = int_idx
                               entry_conf = model_prob

                   # --- TIME LIMIT CHECK ---
                   if position is not None:
                       limit = tb.get('time_limit_bars', 24)
                       bars_held = int_idx - entry_int_idx
                       if bars_held >= limit:
                           tl_pnl = 0.0
                           if position == 'LONG':
                               tl_pnl = (close - entry_price) / entry_price
                           else:
                               tl_pnl = (entry_price - close) / entry_price
                               
                           trades.append({
                                "_sort_key": entry_time,
                                "Asset": asset['name'],
                                "Timeframe": short_tf,
                                "Time": format_time(entry_time),
                                "Exit Time": format_time(idx),
                                "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                                "Price": entry_price,
                                "Confidence": f"{entry_conf:.0%}",
                                "Model": "✅",
                                "Return_Pct": tl_pnl - 0.002, # Deduct 0.2% fee
                                "SL_Pct": curr_sl_pct,
                                "Status": "TIME LIMIT ⌛"
                           })
                           position = None
                               
               # --- END OF LOOP ---
               if position is not None:
                    last_price = df.iloc[-1]['close']
                    if position == 'LONG':
                        pnl = (last_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - last_price) / entry_price
                        
                    trades.append({
                        "_sort_key": entry_time,
                        "Asset": asset['name'],
                        "Timeframe": short_tf,
                        "Time": format_time(entry_time),
                        "Exit Time": "-",
                        "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                        "Price": entry_price,
                        "Confidence": f"{entry_conf:.0%}",
                        "Model": "✅",
                        "Return_Pct": pnl, 
                        "SL_Pct": curr_sl_pct,
                        "Status": "OPEN"
                    })

               return trades

            # Run simulation FIRST on full history
            asset_history = simulate_history_stateful(df_strat, asset['type'], threshold_val=threshold)
            
            # --- Check Active Trade Strategy (Prioritize OPEN history) ---
            active_trade_data = None
            
            # Logic: If last trade in history is "OPEN", that is our active trade.
            if asset_history and asset_history[-1]['Status'] == 'OPEN':
                last_open = asset_history[-1]
                ts = last_open['_sort_key']
                
                # Format TS relative to EST (Fix Future Time Issue)
                if ts.tzinfo:
                     ts_est = ts.tz_convert('America/New_York')
                else:
                     ts_est = ts.tz_localize('UTC').tz_convert('America/New_York')
                     
                ts_str = ts_est.strftime('%Y-%m-%d %H:%M:%S')

                # Determine Decimal Precision
                decimals = 2
                s_lower = asset['symbol'].lower()
                n_lower = asset['name'].lower()
                
                high_prec_keywords = ['doge', 'ada', 'xrp', 'link', 'arb', 'algo', 'matic', 'ftm', 'aud']
                if any(k in s_lower or k in n_lower for k in high_prec_keywords):
                    decimals = 4
                
                if '=x' in s_lower:
                    if 'jpy' in s_lower:
                        decimals = 2
                    else:
                        decimals = 5

                # Reconstruct Active Trade Data Object
                # Need TP/SL prices. They are derived from entry price in simulation logic but not stored in dict unless I change simulate_history.
                # I can recalculate them or assume they are implicit.
                # For display, we want explicit TP/SL.
                
                # Recalculate TP/SL for display (using same logic as loop)
                # Need to know if crypto dynamic was used.
                
                # ... Retrieve params ...
                a_type = asset['type']
                curr_tp = 0.0
                curr_sl = 0.0
                
                if a_type == 'crypto':
                    if crypto_use_dynamic:
                        # Find sigma at ENTRY time
                        try:
                            # ts is exact index
                            entry_idx = df_strat.index.get_loc(ts)
                            sigma_val = df_strat['sigma'].iloc[entry_idx]
                        except: sigma_val = 0.01 
                        
                        curr_tp = sigma_val * crypto_dyn_pt_k
                        curr_sl = sigma_val * crypto_dyn_sl_k
                    else:
                        curr_tp = tp_crypto
                        curr_sl = sl_crypto
                elif a_type == 'forex':
                    curr_tp = tp_forex
                    curr_sl = sl_forex
                else: # trad
                    curr_tp = tp_trad
                    curr_sl = sl_trad
                    
                ep = last_open['Price']
                
                pos_raw = 'LONG' if 'LONG' in last_open['Type'] else 'SHORT'
                
                if pos_raw == 'LONG':
                    tp_price = ep * (1 + curr_tp)
                    sl_price = ep * (1 - curr_sl)
                else:
                    tp_price = ep * (1 - curr_tp)
                    sl_price = ep * (1 + curr_sl)
                    
                rec_action = f"✅ TAKE" # By definition if it's open in history it passed threshold

                active_trade_data = {
                    "_sort_key": ts, # Keep raw for sorting
                    "Asset": last_open['Asset'],
                    "Symbol": asset['symbol'],
                    "Type": last_open['Type'],
                    "Timeframe": last_open['Timeframe'],
                    "Entry_Time": ts_str, # EST String
                    "Signal_Time": ts_str,
                    "Entry_Price": f"{ep:.{decimals}f}",
                    "Take_Profit": f"{tp_price:.{decimals}f}",
                    "Stop_Loss": f"{sl_price:.{decimals}f}",
                    "RR": f"{(abs(tp_price - ep) / abs(ep - sl_price) if abs(ep - sl_price) > 0 else 0):.2f}R",
                    "Current_Price": f"{df_strat.iloc[-1]['close']:.{decimals}f}",
                    "PnL (%)": f"{last_open['Return_Pct']:.2%}",
                    "Confidence": last_open['Confidence'],
                    "Action": rec_action,
                    "Signal": pos_raw,
                    "Strategy": strat_name
                }

            # --- Latest Candle Status ---
            current = df_strat.iloc[-1]
            signal = current['signal_type']
            ts_str = format_time(current.name)
            
            result_data = {
                "Asset": asset['name'],
                "Price": float(current['close']),
                "Trend": "BULLISH" if current['is_bullish'] else ("BEARISH" if current['is_bearish'] else "NEUTRAL"),
                "Confidence": f"{prob:.0%}",
                "Signal": signal,
                "Time": ts_str
            }
            
            # --- Collect Historical Signals & Simulate PnL ---
            
            # Helper to run stateful simulation on the history
            def simulate_history_stateful(df, asset_type, threshold_val=0.40):
               trades = []
               position = None
               entry_price = 0.0
               entry_time = None
               entry_conf = 0.0
               sl_price = 0.0
               
               if asset_type == 'crypto':
                   if crypto_use_dynamic:
                       # Estimate dynamic mean for history (simplified)
                       # Or just use the current config logic for approximate history
                       curr_tp_pct = tp_crypto 
                       curr_sl_pct = sl_crypto
                   else:
                       curr_tp_pct = tp_crypto
                       curr_sl_pct = sl_crypto
               elif asset_type == 'forex':
                   curr_tp_pct = tp_forex
                   curr_sl_pct = sl_forex
               else:
                   curr_tp_pct = tp_trad
                   curr_sl_pct = sl_trad
               
               # Iterate through all bars
               for idx, row in df.iterrows():
                   close = row['close']
                   high = row['high']
                   low = row['low']
                   signal = row['signal_type']
                   model_prob = row.get('model_prob', 0.0)
                   int_idx = df.index.get_loc(idx)
                   
                   # --- EXIT LOGIC ---
                   exit_trade = False
                   pnl = 0.0
                   status = ""
                   
                   if position == 'LONG':
                       # Check TP
                       if high >= entry_price * (1 + curr_tp_pct):
                           pnl = curr_tp_pct
                           status = "HIT TP 🟢"
                           exit_trade = True
                       # Check SL
                       elif low <= entry_price * (1 - curr_sl_pct):
                           pnl = -curr_sl_pct
                           status = "HIT SL 🔴"
                           exit_trade = True

                   elif position == 'SHORT':
                       # Check TP
                       if low <= entry_price * (1 - curr_tp_pct):
                           pnl = curr_tp_pct
                           status = "HIT TP 🟢"
                           exit_trade = True
                       # Check SL
                       elif high >= entry_price * (1 + curr_sl_pct):
                           pnl = -curr_sl_pct
                           status = "HIT SL 🔴"
                           exit_trade = True
                           
                   if exit_trade:
                       raw_tp_price = entry_price * (1 + curr_tp_pct) if position == 'LONG' else entry_price * (1 - curr_tp_pct)
                       raw_sl_price = entry_price * (1 - curr_sl_pct) if position == 'LONG' else entry_price * (1 + curr_sl_pct)

                       trades.append({
                            "_sort_key": entry_time,
                            "Asset": asset['name'],
                            "Timeframe": short_tf,
                            "Time": format_time(entry_time),
                            "Exit Time": format_time(idx),
                            "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                            "Price": entry_price,
                            "Confidence": f"{entry_conf:.0%}",
                            "Model": "✅",
                            "Return_Pct": pnl, 
                            "SL_Pct": curr_sl_pct,
                            "Status": status,
                            "Strategy": strat_name,
                            "Raw_TP": raw_tp_price,
                            "Raw_SL": raw_sl_price
                       })
                       position = None

                   # --- ENTRY & REVERSAL LOGIC ---
                   # Only take filtered signals
                   if model_prob > threshold_val:
                       new_pos = None
                       # Check Strategy Output strings
                       if 'LONG_ZONE' in signal or 'LONG_REV' in signal or 'SCALP_LONG' in signal:
                           new_pos = 'LONG'
                       elif 'SHORT_ZONE' in signal or 'SHORT_REV' in signal or 'SCALP_SHORT' in signal:
                           new_pos = 'SHORT'
                           
                       if new_pos:
                           # If reversal (flipping position)
                           if position is not None and position != new_pos:
                               # Close current position at market price (reversal)
                               last_close_pnl = 0.0
                               if position == 'LONG':
                                   last_close_pnl = (close - entry_price) / entry_price
                               else:
                                   last_close_pnl = (entry_price - close) / entry_price
                                   
                               # Safety: If PnL exceeds SL, cap it (assume SL hit during move)
                               status_label = "FLIP 🔄"
                               if last_close_pnl < -curr_sl_pct:
                                   last_close_pnl = -curr_sl_pct
                                   status_label = "HIT SL 🔴"
                                   
                               raw_tp_price = entry_price * (1 + curr_tp_pct) if position == 'LONG' else entry_price * (1 - curr_tp_pct)
                               raw_sl_price = entry_price * (1 - curr_sl_pct) if position == 'LONG' else entry_price * (1 + curr_sl_pct)

                               trades.append({
                                    "_sort_key": entry_time,
                                    "Asset": asset['name'],
                                    "Timeframe": short_tf,
                                    "Time": format_time(entry_time),
                                    "Exit Time": format_time(idx),
                                    "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                                    "Price": entry_price,
                                    "Confidence": f"{entry_conf:.0%}",
                                    "Model": "✅",
                                    "Return_Pct": last_close_pnl, 
                                    "SL_Pct": curr_sl_pct,
                                    "Status": status_label,
                                    "Strategy": strat_name,
                                    "Raw_TP": raw_tp_price,
                                    "Raw_SL": raw_sl_price
                               })
                               # Prepare for new entry
                               position = None 
                           
                           # Enter new position (if not already in one or just flipped)
                           if position is None:
                               position = new_pos
                               entry_price = close
                               entry_time = idx
                               entry_int_idx = int_idx # Store integer index for time limit check
                               entry_conf = model_prob
                               entry_tp_pct = curr_tp_pct
                               entry_sl_pct = curr_sl_pct

                   # --- TIME LIMIT CHECK ---
                   if position is not None:
                       # Determine Limit based on TF
                       # Use config values if possible
                       limit = tb.get('time_limit_bars', 24)
                       
                       bars_held = int_idx - entry_int_idx
                       if bars_held >= limit:
                           # Close at current close
                           tl_pnl = 0.0
                           if position == 'LONG':
                               tl_pnl = (close - entry_price) / entry_price
                           else:
                               tl_pnl = (entry_price - close) / entry_price
                               
                           raw_tp_price = entry_price * (1 + curr_tp_pct) if position == 'LONG' else entry_price * (1 - curr_tp_pct)
                           raw_sl_price = entry_price * (1 - curr_sl_pct) if position == 'LONG' else entry_price * (1 + curr_sl_pct)

                           trades.append({
                                "_sort_key": entry_time,
                                "Asset": asset['name'],
                                "Timeframe": short_tf,
                                "Time": format_time(entry_time),
                                "Exit Time": format_time(idx),
                                "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                                "Price": entry_price,
                                "Confidence": f"{entry_conf:.0%}",
                                "Model": "✅",
                                "Return_Pct": tl_pnl, 
                                "SL_Pct": curr_sl_pct,
                                "Status": "TIME LIMIT ⌛",
                                "Strategy": strat_name,
                                "Raw_TP": raw_tp_price,
                                "Raw_SL": raw_sl_price
                           })
                           position = None
                               
               # --- END OF LOOP ---
                               
               # --- END OF LOOP ---
               # If position is still open, calculate floating PnL
               if position is not None:
                    last_price = df.iloc[-1]['close']
                    if position == 'LONG':
                        pnl = (last_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - last_price) / entry_price
                        
                    if position == 'LONG':
                        raw_tp_price = entry_price * (1 + entry_tp_pct)
                        raw_sl_price = entry_price * (1 - entry_sl_pct)
                    else:
                        raw_tp_price = entry_price * (1 - entry_tp_pct)
                        raw_sl_price = entry_price * (1 + entry_sl_pct)

                    trades.append({
                        "_sort_key": entry_time,
                        "Asset": asset['name'],
                        "Timeframe": short_tf,
                        "Time": format_time(entry_time),
                        "Exit Time": "-",
                        "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                        "Price": entry_price,
                        "Confidence": f"{entry_conf:.0%}",
                        "Model": "✅",
                        "Return_Pct": pnl, 
                        "SL_Pct": curr_sl_pct,
                        "Status": "OPEN",
                        "Strategy": strat_name,
                        "Raw_TP": raw_tp_price,
                        "Raw_SL": raw_sl_price
                    })


               return trades

            # Run simulation on FULL fetched history (not just tail)
            asset_history = simulate_history_stateful(df_strat, asset['type'], threshold_val=threshold)
            
            return result_data, active_trade_data, asset_history
            
        except Exception as e:
            print(f"DEBUG: Error processing {asset['symbol']}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    # Parallel Execution
    # Increased workers to 8 to speed up initial fetch (I/O bound)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_asset = {executor.submit(process_asset, asset): asset for asset in ASSETS}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_asset)):
            if not silent and progress_bar:
                progress_bar.progress((i + 1) / len(ASSETS))
            try:
                # Add timeout to prevent hanging
                res, trade, hist = future.result(timeout=10)
                if res: results.append(res)
                if trade: active_trades.append(trade)
                if hist: historical_signals.extend(hist)
            except concurrent.futures.TimeoutError:
                print(f"Timeout processing asset")
            except Exception:
                pass

    if not silent and progress_bar: progress_bar.empty()
    if not silent and status_text: status_text.empty()
    
    # Sort
    if active_trades:
        active_trades.sort(key=lambda x: x['_sort_key'], reverse=True)
        
    return pd.DataFrame(results), pd.DataFrame(active_trades), historical_signals

def analyze_ichimoku_strategy(silent=False):
    """
    Runs Ichimoku Strategy (Cloud) for All Assets on 4H/1D.
    Hybrid Settings: Crypto=Turbo, TradFi=Slow.
    """
    status = None
    if not silent: 
        status = st.empty()
        status.text("Scanning Ichimoku Cloud...")
        
    active_trades = []
    historical_signals = []
    
    # Configs
    TRADFI_CFG = {"tenkan": 20, "kijun": 60, "span_b": 120, "displacement": 30}
    CRYPTO_CFG = {"tenkan": 7, "kijun": 21, "span_b": 42, "displacement": 21}
    
    def process_ichi_asset(asset):
        try:
            is_crypto = (asset['type'] == 'crypto')
            cfg = CRYPTO_CFG if is_crypto else TRADFI_CFG
            
            # Smart Timeframe Selection
            # Crypto: 1D Only (Turbo)
            # TradFi: 4H + 1D (Slow)
            target_tfs = ["1d"]
            if not is_crypto:
                target_tfs = ["4h", "1d"]
            
            ichi = IchimokuStrategy(**cfg)
            
            asset_res_trades = []
            asset_hist = []
            
            for tf in target_tfs:
                # Need enough history for Lookback (120) + Displacement (30) + Simulation
                df = fetch_data(asset['symbol'], asset['type'], tf, 500)
                
                if df is None or df.empty or len(df) < 160: continue
                
                df = ichi.apply_strategy(df, tf)
                
                # ML Feature Calc
                df = calculate_ichi_features(df)
                
                # Inference
                if ICHI_MODEL and ICHI_FEATS:
                    try:
                        # Ensure cols exist
                        # Fill missing just in case
                        for c in ICHI_FEATS:
                            if c not in df.columns: df[c] = 0
                            
                        probs = ICHI_MODEL.predict_proba(df[ICHI_FEATS])[:, 1]
                        df['ml_conf'] = probs
                    except:
                        df['ml_conf'] = 0.99
                else:
                    df['ml_conf'] = 1.0 # Default if no model
                
                signals = df[df['signal_type'].notna()].copy()
                
                if signals.empty: continue
                
                for entry_time, row in signals.iterrows():
                    signal_type = row['signal_type']
                    entry_price = row['close']
                    pos_type = "LONG" if signal_type == "LONG" else "SHORT"
                     
                    # Exit Check (Kijun Trail)
                    future_df = df.loc[entry_time:].iloc[1:]
                    
                    outcome_status = "OPEN"
                    pnl = 0.0
                    exit_time_str = "-"
                    exit_price = entry_price
                    curr_price = entry_price
                    
                    if not future_df.empty:
                        curr_price = df.iloc[-1]['close']
                        trade_exit_time = future_df.index.max()
                        
                        for t, f_row in future_df.iterrows():
                            if signal_type == "LONG" and f_row['close'] < f_row['kijun']:
                                outcome_status = "HIT SL 🔴"
                                exit_price = f_row['close']
                                exit_time_str = str(t)
                                trade_exit_time = t
                                break
                            elif signal_type == "SHORT" and f_row['close'] > f_row['kijun']:
                                outcome_status = "HIT SL 🔴"
                                exit_price = f_row['close']
                                exit_time_str = str(t)
                                trade_exit_time = t
                                break
                        
                        if outcome_status == "OPEN":
                             # Mark to market
                             if signal_type == "LONG": pnl = (curr_price - entry_price) / entry_price
                             else: pnl = (entry_price - curr_price) / entry_price
                        else:
                             # Closed
                             if signal_type == "LONG": pnl = (exit_price - entry_price) / entry_price
                             else: pnl = (entry_price - exit_price) / entry_price
                    else:
                        trade_exit_time = entry_time
                        curr_price = entry_price
                    
                    pnl -= 0.001 # 0.1% Fee estimate
                    kijun_sl = row['kijun']
                    
                    conf_val = row.get('ml_conf', 1.0)
                    
                    trade_obj = {
                        "_sort_key": entry_time,
                        "Asset": asset['name'],
                        "Timeframe": "4H" if tf=='4h' else "1D",
                        "Time": str(entry_time),
                        "Exit Time": exit_time_str,
                        "Type": f"{pos_type} {'🟢' if pos_type == 'LONG' else '🔴'}",
                        "Price": entry_price,
                        "Current_Price": curr_price,
                        "Confidence": f"{conf_val:.0%}",
                        "Model": "✅" if conf_val > 0.5 else "⚠️",
                        "Return_Pct": pnl,
                        "Status": outcome_status,
                        "Strategy": "Ichimoku Cloud",
                        "Raw_TP": 0.0,
                        "Raw_SL": kijun_sl,
                        "Entry_Price": entry_price,
                        "Entry_Time": str(entry_time),
                        "Stop_Loss": float(f"{kijun_sl:.5f}"),
                        "Take_Profit": 0.0,
                        "Stop Loss": float(f"{kijun_sl:.5f}"),
                        "Take Profit": 0.0,
                        "Action": "✅ TAKE",
                        "PnL (%)": f"{pnl*100:.2f}%"
                    }
                    
                    asset_hist.append(trade_obj)
                    if outcome_status == "OPEN":
                        asset_res_trades.append(trade_obj)
            
            return asset_res_trades, asset_hist

        except Exception:
            return [], []

    # Parallel Execution (5 workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ichi_asset, asset): asset for asset in ASSETS}
        for future in concurrent.futures.as_completed(futures):
            act, hist = future.result()
            if act: active_trades.extend(act)
            if hist: historical_signals.extend(hist)
            
    if not silent and status: status.empty()
    return None, pd.DataFrame(active_trades), historical_signals

def analyze_cls_strategy(silent=False):
    """
    Runs CLS Strategy for TradFi Assets on Daily/Hourly MTF.
    Includes Historical Simulation.
    """
    status = None
    if not silent: 
        status = st.empty()
        status.text("Scanning Daily CLS Ranges...")
        
    active_trades = []
    historical_signals = []
    
    # Filter TradFi Assets (Forex + Trad)
    tradfi_assets = [a for a in ASSETS if a['type'] == 'trad' or a['type'] == 'forex']
    
    cls_strat = CLSRangeStrategy() # Config B (1.5x/10) Default
    
    # Threading for speed
    def process_cls_asset(asset):
        try:
             # Fetch MTF
            df_htf = fetch_data(asset['symbol'], 'trad', '1d', 500)
            df_ltf = fetch_data(asset['symbol'], 'trad', '1h', 400)
            
            if df_htf.empty or df_ltf.empty: return None, []
            
            df = cls_strat.apply_mtf(df_htf, df_ltf)
            if df.empty or 'signal_type' not in df.columns: return None, []
            
            # ML Logic
            df = calculate_cls_features(df)
            
            if CLS_MODEL and CLS_FEATS:
                try:
                    for c in CLS_FEATS:
                        if c not in df.columns: df[c] = 0
                    probs = CLS_MODEL.predict_proba(df[CLS_FEATS])[:, 1]
                    df['ml_conf'] = probs
                except:
                    df['ml_conf'] = 0.99
            else:
                df['ml_conf'] = 1.0
            
            # --- Simulate History (ordered) ---
            sig_rows = df[df['signal_type'].astype(str).str.contains("CLS", na=False)]
            sig_rows = sig_rows.sort_index()

            hist_trades = []
            
            # Initialize last_exit using same tz
            tz_info = sig_rows.index.tz
            last_exit_time = pd.Timestamp.min
            if tz_info:
                last_exit_time = last_exit_time.tz_localize(tz_info)
            
            active_trade = None

            for entry_time, row in sig_rows.iterrows():
                # 1. Overlap Check
                if entry_time <= last_exit_time:
                    continue

                entry_price = row['close']
                s_type = row['signal_type']
                tp = row['target_price']
                sl = row['stop_loss']
                
                if pd.isna(tp) or pd.isna(sl): continue
                
                pos_type = "LONG" if "LONG" in s_type else "SHORT"
                
                # Check Outcome
                outcome_status = "OPEN"
                pnl = 0.0
                exit_time_str = "-"
                trade_exit_time = None
                
                # Look forward
                future_df = df.loc[entry_time:].iloc[1:] # strictly after
                
                if not future_df.empty:
                    for _, f_row in future_df.iterrows():
                        h = f_row['high']
                        l = f_row['low']
                        
                        if pos_type == "LONG":
                            if l <= sl:
                                outcome_status = "HIT SL 🔴"
                                pnl = (sl - entry_price) / entry_price
                                exit_time_str = str(f_row.name)
                                trade_exit_time = f_row.name
                                break
                            if h >= tp:
                                outcome_status = "HIT TP 🟢"
                                pnl = (tp - entry_price) / entry_price
                                exit_time_str = str(f_row.name)
                                trade_exit_time = f_row.name
                                break
                        else: # SHORT
                            if h >= sl:
                                outcome_status = "HIT SL 🔴"
                                pnl = (entry_price - sl) / entry_price
                                exit_time_str = str(f_row.name)
                                trade_exit_time = f_row.name
                                break
                            if l <= tp:
                                outcome_status = "HIT TP 🟢"
                                pnl = (entry_price - tp) / entry_price
                                exit_time_str = str(f_row.name)
                                trade_exit_time = f_row.name
                                break
                    
                    if outcome_status == "OPEN":
                        curr_price = df.iloc[-1]['close']
                        if pos_type == "LONG": pnl = (curr_price - entry_price) / entry_price
                        else: pnl = (entry_price - curr_price) / entry_price
                        trade_exit_time = future_df.index.max()
                else:
                    # No future data = Current Bar Signal
                    trade_exit_time = entry_time # Block until this bar passes
                
                # Update State
                if trade_exit_time:
                    last_exit_time = trade_exit_time
                
                # Deduct fee 0.2%
                pnl -= 0.002
                
                conf_val = row.get('ml_conf', 1.0)
                
                # Store
                trade_obj = {
                    "_sort_key": entry_time,
                    "Asset": asset['name'],
                    "Timeframe": "1H",
                    "Time": str(entry_time),
                    "Exit Time": exit_time_str,
                    "Type": f"{pos_type} {'🟢' if pos_type == 'LONG' else '🔴'}",
                    "Price": entry_price,
                    "Confidence": f"{conf_val:.0%}",
                    "Model": "✅" if conf_val > 0.5 else "⚠️",
                    "Return_Pct": pnl,
                    "SL_Pct": abs((entry_price - sl)/entry_price) if entry_price else 0,
                    "Status": outcome_status,
                    "Strategy": "Daily CLS Range",
                    "Raw_TP": tp,
                    "Raw_SL": sl
                }
                
                hist_trades.append(trade_obj)
            
            # --- Separate Active vs History ---
            if hist_trades:
                last_t = hist_trades[-1]
                last_ts = pd.Timestamp(last_t['_sort_key'])
                current_ts = df.iloc[-1].name
                
                # Show in Active if Fresh OR Open
                if last_ts == current_ts or last_t['Status'] == 'OPEN':
                    # Move to Active
                    pos_str = last_t['Type'] # "LONG 🟢"
                    
                    price = last_t['Price']
                    sl = last_t.get('Raw_SL', 0)
                    tp = last_t.get('Raw_TP', 0)
                    curr_p = df.iloc[-1]['close']
                    pnl_val = last_t.get('Return_Pct', 0.0)
                    
                    active_trade = {
                        "_sort_key": last_t['_sort_key'],
                        "Asset": asset['name'],
                        "Timeframe": "1H",
                        "Time": last_t['Time'], 
                        "Entry_Time": last_t['Time'],
                        "Type": pos_str,
                        "Price": price,
                        "Entry_Price": price,
                        "Current_Price": curr_p,
                        "PnL (%)": f"{pnl_val*100:.2f}%",
                        "Confidence": last_t.get('Confidence', "100%"),
                        "Action": "✅ TAKE",
                        "Stop_Loss": round(sl, 4),
                        "Take_Profit": round(tp, 4),
                        "Stop Loss": round(sl, 4), # Check compatibility
                        "Take Profit": round(tp, 4), # Check compatibility
                        "Strategy": "Daily CLS Range"
                    }
                    
                    # Do NOT remove from history list, so it appears in both (User Visibility)
                    # hist_trades.pop()
            
            return active_trade, hist_trades
            
        except Exception:
            pass
        return None, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_cls_asset, asset): asset for asset in tradfi_assets}
        for future in concurrent.futures.as_completed(futures):
            trade, hist = future.result()
            if trade:
                active_trades.append(trade)
            if hist:
                historical_signals.extend(hist)
                
    if not silent and status: status.empty()
    
    return None, pd.DataFrame(active_trades), historical_signals

def run_simulation(df, i, signal_type, asset_type, config):
    # Retrieve Triple Barrier Params
    if asset_type == 'crypto':
        pt = 0.08
        sl = 0.05
    else:
        pt = 0.03
        sl = 0.04
        
    entry_price = df.iloc[i]['close']
    
    # Determine direction
    direction = 1 if 'LONG' in signal_type else -1
    
    # Time Limit (40 Days) -> Convert to bars (approx)
    # df index is datetime
    start_time = df.index[i]
    # Look forward
    future_window = df.iloc[i+1:]
    
    # Limit to approx 40 days if possible? 
    # Or just iterate until exit.
    # Simple loop for accuracy
    
    outcome_pct = 0.0
    
    for j in range(len(future_window)):
        row = future_window.iloc[j]
        curr_time = row.name
        
        # Check Time Limit (approx 40 days)
        if (curr_time - start_time).days >= 40:
             # Timeout exit
            exit_price = row['close']
            if direction == 1:
                outcome_pct = (exit_price - entry_price) / entry_price
            else:
                outcome_pct = (entry_price - exit_price) / entry_price
            return outcome_pct, sl

        # Check Prices
        high = row['high']
        low = row['low']
        
        if direction == 1:
            # Check PT (High)
            if high >= entry_price * (1 + pt):
                return pt, sl # Hit PT
            # Check SL (Low)
            if low <= entry_price * (1 - sl):
                return -sl, sl # Hit SL
        else:
            # Check PT (Low)
            if low <= entry_price * (1 - pt):
                return pt, sl # Hit PT
            # Check SL (High)
            if high >= entry_price * (1 + sl):
                return -sl, sl # Hit SL
                
    # End of Data (Open Position)
    # Mark to market
    last_price = future_window.iloc[-1]['close'] if not future_window.empty else entry_price
    if direction == 1:
        outcome_pct = (last_price - entry_price) / entry_price
    else:
        outcome_pct = (entry_price - last_price) / entry_price
        
    return outcome_pct, sl


def format_time(ts):
    if pd.isna(ts): return "N/A"
    try:
        if ts.tz is None: ts = ts.tz_localize('UTC')
        ts_est = ts.tz_convert('America/New_York')
        return ts_est.strftime('%Y-%m-%d %H:%M:%S')
    except: return str(ts)

def highlight_confidence(row):
    try:
        # Determine Status based on available columns
        status = ""
        if 'Action' in row:
            status = str(row['Action']).upper()
        elif 'Model' in row:
            status = str(row['Model']).upper()
            
        if "TAKE" in status or "✅" in status:
            return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
        elif "SKIP" in status or "❌" in status:
            return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
        else:
            return [''] * len(row)
    except:
        return [''] * len(row)

# --- Render UI ---

# --- PROP RISK LOGIC (Ported from React) ---
# --- PROP RISK LOGIC ---
PROP_FIRM_CONFIGS = {
    # Configurations are now embedded in the account objects for flexibility
}

def init_prop_accounts():
    if 'user_accounts' not in st.session_state:
        # Load from Grimoire first
        grimoire = load_grimoire()
        saved_accounts = grimoire.get('prop_accounts', [])
        
        if saved_accounts:
             st.session_state.user_accounts = saved_accounts
        else:
            # Initial State based on User Request
            st.session_state.user_accounts = [
                {
                    "id": 1, 
                    "name": "Hyrotrader Swing 1 step 50K", 
                    "size": 50000, 
                    "profit_target_amt": 7317, 
                    "drawdown_limit_amt": 2000,
                    "currentBalance": 50000.0, 
                },
                {
                    "id": 2, 
                    "name": "MCF 2 step 50K", 
                    "profit_target_amt": 4000, 
                    "drawdown_limit_amt": 5000,
                    "currentBalance": 48788.18, 
                },
                {
                    "id": 3, 
                    "name": "Hyrotrader Swing 1 step 25K", 
                    "size": 25000, 
                    "profit_target_amt": 4000, 
                    "drawdown_limit_amt": 1000,
                    "currentBalance": 24875.0, 
                },
                 {
                    "id": 4, 
                    "name": "MCF 2 step 25K", 
                    "size": 25000, 
                    "profit_target_amt": 2000, 
                    "drawdown_limit_amt": 2500,
                    "currentBalance": 24554.26, 
                },
                {
                    "id": 5, 
                    "name": "Breakout 1 step 10K", 
                    "size": 10000, 
                    "profit_target_amt": 1000, 
                    "drawdown_limit_amt": 600,
                    "currentBalance": 9716.24, 
                }
            ]

def render_prop_risk():
    init_prop_accounts()

    # --- Position Size Calculator ---
    with st.container(border=True):
        st.markdown("### 🧮 Position Size Calculator")
        
        # Get list of asset names or symbols
        try:
            asset_options = [a['symbol'] for a in ASSETS]
        except:
            asset_options = ["BTC/USDT", "ETH/USDT"] # Fallback

        # Row 1: Inputs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            calc_asset = st.selectbox("Asset", options=asset_options, key="calc_asset")
        with c2:
            calc_entry = st.number_input("Entry Price", min_value=0.0, format="%.5f", step=0.0001, key="calc_entry")
        with c3:
            calc_sl = st.number_input("Stop Loss", min_value=0.0, format="%.5f", step=0.0001, key="calc_sl")
        with c4:
            calc_risk = st.number_input("Risk ($)", min_value=0.0, value=100.0, step=10.0, key="calc_risk")

        # Row 2: Results (Calculated only if valid)
        if calc_entry > 0 and calc_sl > 0 and calc_risk > 0 and calc_entry != calc_sl:
            st.markdown("---")
            
            diff = calc_entry - calc_sl
            direction = "LONG" if diff > 0 else "SHORT"
            risk_per_unit = abs(diff)
            
            # Position Size (Units) = Risk / |Entry - SL|
            pos_size = calc_risk / risk_per_unit
            
            # Notional Value = Units * Entry
            notional = pos_size * calc_entry
            
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                st.metric("Direction", f"{direction} {'🟢' if direction == 'LONG' else '🔴'}")
            with rc2:
                st.metric("Position Size", f"{pos_size:,.4f}")
            with rc3:
                st.metric("Total Position Value", f"${notional:,.2f}")

    # Grid Layout
    cols = st.columns(2)
    
    accounts_changed = False
    
    for i, account in enumerate(st.session_state.user_accounts):
        size = account.get('size', 50000)
        pt_amt = account['profit_target_amt']
        dd_limit_amt = account['drawdown_limit_amt']
        
        # Balance Target / Limit Levels
        target_bal = size + pt_amt
        min_bal = size - dd_limit_amt
        
        # Use col 0 or 1
        with cols[i % 2]:
            with st.container(border=True):
                # Header
                st.markdown(f"**{account['name']}**")
                
                # Input: Current Balance (Editable)
                # Use a callback or check for change manually
                cur_bal = st.number_input(
                    "Equity", 
                    value=float(account['currentBalance']), 
                    step=10.0, 
                    key=f"bal_{account['id']}"
                )
                
                if cur_bal != account['currentBalance']:
                    account['currentBalance'] = cur_bal
                    accounts_changed = True
                
                # --- Progress Bars ---
                
                # 1. Profit Target Progress
                if cur_bal >= size:
                    pt_progress = min(1.0, (cur_bal - size) / pt_amt)
                    pt_progress = max(0.0, pt_progress)
                else:
                    pt_progress = 0.0
                
                st.caption(f"Profit Target (${target_bal:,.0f})")
                st.progress(pt_progress)
                if cur_bal >= target_bal:
                    st.success("🎉 TARGET HIT")
                
                # 2. Drawdown Proximity
                dist_fail = cur_bal - min_bal
                health_pct = min(1.0, dist_fail / dd_limit_amt)
                health_pct = max(0.0, health_pct)
                
                st.caption(f"Drawdown Limit (${min_bal:,.0f}) - Health: {health_pct:.0%}")
                st.progress(health_pct)
                
                if dist_fail <= 0:
                    st.error("💀 ACCOUNT BREACHED")
                else:
                    st.markdown(f"**Risk Available: :red[${dist_fail:,.2f}]**")

                # --- Risk Calculator ---
                st.markdown("---")
                st.markdown("**Position Sizing (Risk Amount)**")
                c_r1, c_r2, c_r3 = st.columns(3)
                
                risk_1 = cur_bal * 0.01
                risk_05 = cur_bal * 0.005
                risk_025 = cur_bal * 0.0025
                
                c_r1.metric("1.0%", f"${risk_1:.0f}")
                c_r2.metric("0.5%", f"${risk_05:.0f}")
                c_r3.metric("0.25%", f"${risk_025:.0f}")

    if accounts_changed:
        # Load valid existing state (preserve mana/spells)
        current_state = load_grimoire()
        current_state['prop_accounts'] = st.session_state.user_accounts
        save_grimoire(current_state)

# --- Render UI ---

# Tabs removed for cleaner UI
# [tab_dash] = st.tabs(["⚡ Active Signals Dashboard"])

# --- Main Dashboard Logic ---
show_take_only = True 



# --- Runic Alerts Fragment ---
# --- Discord Alerting Logic ---
def send_discord_alert(webhook_url, signal_data):
    """Sends a formatted trade signal to Discord."""
    try:
        # 1. Parse Data (Handle Strings)
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
             if confidence < 1.0: confidence *= 100 # Convert decimal to pct if needed


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
            "footer": {"text": "WizardWave v1.0 • Automated Signal"}
        }

        data = {
            "username": "WizardWave Oracle",
            "embeds": [embed]
        }

        headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(webhook_url, data=json.dumps(data).encode('utf-8'), headers=headers)
        
        # Add Timeout
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status not in [200, 204]:
                pass # Silent fail
                
    except Exception as e:
        print(f"Error sending Discord alert: {e}")

def process_discord_alerts(df):
    """
    Checks for new 'TAKE' signals and sends Discord alerts.
    Uses a robust file-locking simulation strategy to prevent duplicate alerts
    when multiple Streamlit instances are running (common in multipage/multitab usage).
    """
    try:
        # Load Config
        if not os.path.exists(os.path.join('config', 'discord_config.json')):
            return

        with open(os.path.join('config', 'discord_config.json'), 'r') as f:
            disc_config = json.load(f)
            webhook_url = disc_config.get('webhook_url')
            
        if not webhook_url:
            return

        processed_file = os.path.join('data', 'processed_signals.json')
        
        # Max Age for Alert (e.g. 1.5 hours). 
        # Prevents flooding old alerts if app restarts.
        MAX_ALERT_AGE_HOURS = 1.5
        now_est = pd.Timestamp.now(tz='America/New_York')

        for _, row in df.iterrows():
            try:
                # 1. Check "TAKE" Criteria First (Optimization)
                action_str = str(row.get('Action', '')).upper()
                is_take = "TAKE" in action_str or "✅" in action_str
                
                if not is_take:
                    continue

                # 2. Create Unique ID (Asset + Timeframe + EntryTime)
                entry_time_str = str(row.get('Entry_Time', ''))
                asset = str(row.get('Asset', ''))
                tf = str(row.get('Timeframe', ''))
                
                # Robust ID: remove spaces/special chars from ID string itself just in case
                sig_id = f"{asset}_{tf}_{entry_time_str}".replace(" ", "_")
                
                if sig_id == "__": continue

                # 3. Freshness Check
                try:
                    # Handle already timezone-aware logic if needed
                    if isinstance(row.get('Entry_Time'), pd.Timestamp) and row.get('Entry_Time').tzinfo:
                         entry_dt = row.get('Entry_Time').tz_convert('America/New_York')
                    else:
                         entry_dt = pd.to_datetime(entry_time_str).tz_localize('America/New_York')
                    
                    if (now_est - entry_dt).total_seconds() > (MAX_ALERT_AGE_HOURS * 3600):
                        continue # Too old
                except:
                    continue # Skip if date parsing fails to be safe

                # 4. ATOMIC CHECK-AND-SEND
                # Add random jitter to desynchronize multiple tabs checking at exact same millisecond
                time.sleep(random.uniform(0.1, 1.5))
                
                # Read latest IDs from disk immediately before decision
                current_ids = set()
                if os.path.exists(processed_file):
                    try:
                        with open(processed_file, 'r') as f:
                            content = json.load(f)
                            if isinstance(content, list):
                                current_ids = set(content)
                    except:
                        pass # Start empty if corrupt

                # If NOT in file, then we send
                if sig_id not in current_ids:
                    
                    # SEND ALERT
                    send_discord_alert(webhook_url, row)
                    
                    # WRITE IMMEDIATELY to lock it for others
                    current_ids.add(sig_id)
                    try:
                        with open(processed_file, 'w') as f:
                            json.dump(list(current_ids), f)
                    except Exception as e:
                        print(f"Error saving processed ID {sig_id}: {e}")
                    
            except Exception as inner_e:
                print(f"Skipping alert row: {inner_e}")
                continue
                
    except Exception as e:
        print(f"Error processing Discord alerts: {e}")

@st.fragment(run_every=120)
def show_runic_alerts():
    # --- Execution Settings Removed ---

    # Header Row with Refresh Button
    with st.container(border=True):
        # Adjusted layout for Mobile Optimization
        # Row 1: Title (Centered)
        st.markdown('<div class="runic-header" style="font-size: 1rem; border: none !important; margin-bottom: 5px; padding: 0; margin-top: -5px; background: transparent; text-align: center;">RUNIC ALERTS</div>', unsafe_allow_html=True)
        
        # Row 2: Controls [History | Metrics | Refresh]
        # Using columns to center the metrics and place buttons on sides
        c_hist, c_metric, c_btn = st.columns([0.15, 0.70, 0.15], gap="small")
        
        with c_hist:
            # History Button
            if st.button("📜", key="hist_side", help="Signal History", use_container_width=True):
                st.session_state.active_tab = 'HISTORY'
                st.rerun()

        with c_metric:
             # Metrics
             # Display 24H Return (Will be populated after calculation or from state)
             # Use a placeholder for dynamic updates
             return_placeholder = st.empty()
        
        def render_return_value(val_24, val_12):
            # 24H Logic
            r24_color = "#00ff88" if val_24 >= 0 else "#ff3344"
            r24_sign = "+" if val_24 >= 0 else ""
            
            # 12H Logic
            r12_color = "#00ff88" if val_12 >= 0 else "#ff3344"
            r12_sign = "+" if val_12 >= 0 else ""
            
            return_placeholder.markdown(f"""
                <div style="text-align: center; white-space: nowrap; margin-top: 2px;">
                    <div style="display: inline-block; margin-right: 10px;">
                        <span style="font-size: 0.8rem; color: #888; font-weight: bold;">24H: </span>
                        <span style="font-size: 0.8rem; font-weight: bold; color: {r24_color}; text-shadow: 0 0 10px {r24_color}40;">
                            {r24_sign}{val_24:.2%}
                        </span>
                    </div>
                    <div style="display: inline-block;">
                        <span style="font-size: 0.8rem; color: #888; font-weight: bold;">12H: </span>
                        <span style="font-size: 0.8rem; font-weight: bold; color: {r12_color}; text-shadow: 0 0 10px {r12_color}40;">
                            {r12_sign}{val_12:.2%}
                        </span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        # Initial Render
        current_return_24 = st.session_state.get('runic_24h_return', 0.0)
        current_return_12 = st.session_state.get('runic_12h_return', 0.0)
        render_return_value(current_return_24, current_return_12)

        with c_btn:
            refresh_click = st.button("↻", key="refresh_top", help="Refresh", use_container_width=True)
            if refresh_click:
                st.rerun()


    # --- Render Active List Helper ---
    def render_active_list(combined_active):
        if not combined_active.empty:
            tf_order = {"15m": 0, "15 Minutes": 0, "1H": 2, "1 Hour": 2, "4H": 3, "4 Hours": 3, "12H": 4, "12 Hours": 4, "1D": 5, "1 Day": 5, "4D": 6, "4 Days": 6}
            unique_tfs = combined_active['Timeframe'].unique().tolist()
            sorted_tfs = sorted(unique_tfs, key=lambda x: tf_order.get(x, 99))
            
            # --- Active List Render Fix ---
            if "runic_active_tf_selector" not in st.session_state:
                st.session_state.runic_active_tf_selector = sorted_tfs
            
            st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)
            selected_short = st.multiselect("Timeframes", options=sorted_tfs, label_visibility="collapsed", key="runic_active_tf_selector")
            
            df_display = combined_active.copy()
            if show_take_only and 'Action' in df_display.columns:
                df_display = df_display[df_display['Action'].str.contains("TAKE")]

            # --- TRADING PLAN FILTERING ---
            if st.session_state.get('plan_active', False):
                # 1. Confidence
                plan_conf = st.session_state.get('plan_min_conf', 55)
                # Parse conf if string
                def _parse_conf_plan(x):
                    try: return float(str(x).replace('%',''))
                    except: return 0.0
                if 'Confidence' in df_display.columns:
                    df_display = df_display[df_display['Confidence'].apply(_parse_conf_plan) >= plan_conf]
                
                # 2. Timeframes
                plan_tfs = st.session_state.get('plan_allowed_tfs', [])
                if plan_tfs:
                    # Map standard TFs to what might be in DF (e.g. '1H' -> '1 Hour' matching?)
                    # Data uses "15 Minutes", "1 Hour", "4 Hours", "1 Day"?
                    # Or "15m", "1H", "4H", "1D"?
                    # DF usually has mixed or standard. "15 Minutes" (Monitor) vs "1D" (Ichimoku).
                    # We create a loose match.
                    # normalize plan_tfs to simplistic list
                    # "4 Hours" should match "4H"
                    
                    df_display = df_display[df_display['Timeframe'].apply(lambda tf: any(p_tf in str(tf) or (p_tf == "1H" and "1 Hour" in str(tf)) or (p_tf == "4H" and "4 Hours" in str(tf)) or (p_tf == "12H" and "12 Hours" in str(tf)) or (p_tf == "1D" and "1 Day" in str(tf)) for p_tf in plan_tfs))]

            # --- Manual Mode Filters ---
            if st.session_state.get('manual_mode', False):
                # 1. Filter out 15m
                df_display = df_display[~df_display['Timeframe'].isin(['15m', '15 Minutes'])]
                
                # 2. Filter Low Confidence (< 60%)
                def parse_conf(x):
                    try: return float(str(x).replace('%',''))
                    except: return 0.0
                
                if 'Confidence' in df_display.columns:
                    df_display = df_display[df_display['Confidence'].apply(parse_conf) >= 60.0]
            
            if selected_short:
                df_display = df_display[df_display['Timeframe'].isin(selected_short)]
            else:
                st.warning("Select Timeframe")
                df_display = pd.DataFrame(columns=df_display.columns)

            if df_display.empty:
                st.info("No active signals.")
            else:
                ITEMS_PER_PAGE = 5 
                if 'page_number' not in st.session_state: st.session_state.page_number = 0
                total_pages = max(1, (len(df_display) - 1) // ITEMS_PER_PAGE + 1)
                
                if st.session_state.page_number >= total_pages: st.session_state.page_number = total_pages - 1
                if st.session_state.page_number < 0: st.session_state.page_number = 0
                    
                start_idx = st.session_state.page_number * ITEMS_PER_PAGE
                end_idx = start_idx + ITEMS_PER_PAGE
                current_batch = df_display.iloc[start_idx:end_idx]
                
                for index, row in current_batch.iterrows():
                    with st.container():
                        # --- NEW "DATAPAD" RUNIC CARD DESIGN (Full Width) ---
                        with st.container():
                            is_long = "LONG" in row.get('Type', '')
                            direction_color = "#00ff88" if is_long else "#ff3344"
                            asset_name = row['Asset']
                            icon_char = "⚡"
                            if "BTC" in asset_name: icon_char = "₿"
                            elif "ETH" in asset_name: icon_char = "Ξ"
                            elif "SOL" in asset_name: icon_char = "◎"
                            elif "XAU" in asset_name or "Gold" in asset_name: icon_char = "🥇"
                            elif "XAG" in asset_name or "Silver" in asset_name: icon_char = "🥈"
                            
                            action_text = "BULL" if is_long else "BEAR"
                            
                            # Net PnL
                            raw_pnl_str = str(row.get('PnL (%)', '0.00%'))
                            try: raw_pnl_val = float(raw_pnl_str.replace('%',''))
                            except: raw_pnl_val = 0.0
                            fee_cost = st.session_state.get('est_fee_pct', 0.2)
                            net_pnl_val = raw_pnl_val - fee_cost
                            pnl_display_str = f"{net_pnl_val:.2f}%"
                            pnl_color = "#00ff88" if net_pnl_val >= 0 else "#ff3344"
                            
                            # Formatting entry time
                            try:
                                et_str = str(row.get('Entry_Time', ''))
                                if len(et_str) > 10:
                                    et_disp = f"{et_str[5:10]} {et_str[11:16]}"
                                else:
                                    et_disp = et_str
                            except: et_disp = ""

                            # --- Action Bar Prep & IDs ---
                            unique_id = f"{row['Asset']}_{row.get('Timeframe','')}_{row.get('Entry_Time','')}"
                            unique_id = "".join(c for c in unique_id if c.isalnum() or c in ['_','-'])

                            # Custom CSS
                            st.markdown("""
                            <style>
                            div[data-testid="stHorizontalBlock"] {
                                gap: 0rem;
                            }
                            button[kind="secondary"] {
                                border-radius: 0 !important;
                                border: 1px solid rgba(255,255,255,0.1) !important;
                                border-top: none !important;
                                background-color: rgba(0,0,0,0.3) !important;
                            }
                            button[kind="secondary"]:hover {
                                background-color: rgba(255,255,255,0.1) !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            import streamlit.components.v1 as components

                            # Calculate R:R
                            try:
                                ep_val = float(str(row.get('Entry_Price',0)).replace(',',''))
                                tp_val = float(str(row.get('Take_Profit',0)).replace(',',''))
                                sl_val = float(str(row.get('Stop_Loss',0)).replace(',',''))
                                
                                if ep_val > 0 and tp_val > 0 and sl_val > 0:
                                    dist_tp = abs(tp_val - ep_val)
                                    dist_sl = abs(ep_val - sl_val)
                                    if dist_sl > 0:
                                        rr_val = dist_tp / dist_sl
                                        rr_str = f"{rr_val:.2f}R"
                                    else: rr_str = "N/A"
                                else: rr_str = "-"
                            except: rr_str = "-"

                            # HTML Card Content (Reduced Spacing)
                            st.markdown(f"""
                            <div style="font-family: 'Lato', sans-serif; background: rgba(0,0,0,0.2); border-radius: 8px 8px 0 0; border: 1px solid rgba(255,255,255,0.05); padding: 8px 10px; margin-bottom: 0px; display: flex; flex-direction: column;">
                                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px; margin-bottom: 6px;">
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span style="font-size: 1.0rem; color: #f0f0f0;">{icon_char}</span>
                                        <span style="font-weight: 800; font-size: 0.9rem; color: #fff;">{asset_name}</span>
                                        <span style="font-size: 0.6rem; font-weight: bold; padding: 1px 4px; border-radius: 4px; background: {direction_color}25; color: {direction_color}; border: 1px solid {direction_color}30;">{action_text}</span>
                                        <span style="font-size: 0.7rem; font-weight: bold; color: {pnl_color}; margin-left: 5px;">{pnl_display_str}</span>
                                    </div>
                                    <div style="font-size: 0.75rem; font-weight: bold; color: #ff3344;">{row.get('Timeframe')}</div>
                                </div>
                                <div style="font-size: 0.7rem; color: #ccc; line-height: 1.4;">
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                                        <div><span style="color:#777">Sig:</span> <span style="font-weight:bold; color:#eee">{row.get('Action')}</span> <span style="color:#FFB74D">{row.get('Confidence')}</span></div>
                                        <div style="text-align: right;"><span style="color:#777">Ent:</span> <span style="color:#00ff88; font-family:monospace">{row.get('Entry_Price')}</span> <span style="color:#555">|</span> <span style="color:#777">R:R</span> <span style="color:#ffd700; font-family:monospace">{rr_str}</span></div>
                                        <div><span style="color:#777">TP:</span> <span style="color:#eee">{row.get('Take_Profit')}</span> <span style="color:#777">SL:</span> <span style="color:#d8b4fe">{row.get('Stop_Loss')}</span></div>
                                        <div style="text-align: right;"><span style="font-size:0.6rem; color:#00eaff; font-weight:bold; margin-right:5px;">{row.get('Strategy','WizardWave')}</span> <span style="color:#888; font-size: 0.65rem;">🕒 {et_disp}</span></div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            ac1, ac2, ac3 = st.columns(3, gap="small")
                            
                            # Standardized Button Style
                            btn_style = "border: 1px solid rgba(255,255,255,0.1); background-color: rgba(0,0,0,0.3); color: white; padding: 0.25rem 0.5rem; font-size: 0.8rem; cursor: pointer; width: 100%; border-radius: 4px; text-align: center;"
                            
                            with ac1:
                                if st.button("👁️ View", key=f"btn_v_{unique_id}", use_container_width=True, type="secondary"):
                                    st.session_state['active_signal'] = row.to_dict()
                                    st.session_state['active_view_mode'] = 'details' 
                                    tv_sym = get_tv_symbol({'symbol': row.get('Symbol', row.get('Asset'))})
                                    st.session_state['active_tv_symbol'] = tv_sym
                                    st.session_state['active_tab'] = 'PORTAL' 
                                    st.rerun()
                            with ac2:
                                if st.button("🧮 Calc", key=f"btn_c_{unique_id}", use_container_width=True, type="secondary"):
                                    st.session_state['active_signal'] = row.to_dict()
                                    st.session_state['active_view_mode'] = 'calculator'
                                    st.session_state['active_tab'] = 'RISK'
                                    try:
                                        ep = float(str(row['Entry_Price']).replace(',',''))
                                        st.session_state.calc_entry_input = ep
                                    except: st.session_state.calc_entry_input = 0.0
                                    st.rerun()
                            with ac3:
                                # Custom HTML Component for Copy to Clipboard to match style
                                copy_text_pl = f"{asset_name} {action_text} @ {row['Entry_Price']} | TP: {row['Take_Profit']} | SL: {row['Stop_Loss']}"
                                # Escaping quotes for JS
                                safe_copy = copy_text_pl.replace("'", "\\'")
                                
                                # HTML Button that clicks copy
                                # We use height=37 to match Streamlit buttons approx
                                components.html(f"""
                                <html>
                                <head>
                                <style>
                                    body {{ margin: 0; padding: 0; background: transparent; }}
                                    .btn {{
                                        {btn_style}
                                        display: flex; align-items: center; justify-content: center;
                                        font-family: "Source Sans Pro", sans-serif;
                                        height: 38px;
                                        box-sizing: border-box;
                                        transition: background-color 0.2s;
                                    }}
                                    .btn:hover {{ background-color: rgba(255,255,255,0.1); }}
                                    .btn:active {{ background-color: rgba(255,255,255,0.2); transform: translateY(1px); }}
                                </style>
                                </head>
                                <body>
                                    <button class="btn" onclick="copyToClipboard()">
                                        <span id="lbl">📋 Copy</span>
                                    </button>
                                    <script>
                                        function copyToClipboard() {{
                                            navigator.clipboard.writeText('{safe_copy}').then(function() {{
                                                document.getElementById('lbl').innerText = '✅ Copied!';
                                                setTimeout(() => {{ document.getElementById('lbl').innerText = '📋 Copy'; }}, 2000);
                                            }}, function(err) {{
                                                document.getElementById('lbl').innerText = '❌ Error';
                                            }});
                                        }}
                                    </script>
                                </body>
                                </html>
                                """, height=40)
                                
                            st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True) # Reduced Spacer



                # Pagination (Compact)
                st.markdown(f"<div style='text-align: center; color: #666; font-size: 0.75rem; margin-top: 5px; margin-bottom: 2px;'>page {st.session_state.page_number + 1} / {total_pages}</div>", unsafe_allow_html=True)
                p1, p2, p3, p4 = st.columns(4, gap="small")
                p1.button("⏮", key="p_f", disabled=(st.session_state.page_number==0), use_container_width=True, on_click=lambda: setattr(st.session_state, 'page_number', 0))
                p2.button("◀", key="p_p", disabled=(st.session_state.page_number==0), use_container_width=True, on_click=lambda: setattr(st.session_state, 'page_number', st.session_state.page_number - 1))
                p3.button("▶", key="p_n", disabled=(st.session_state.page_number>=total_pages-1), use_container_width=True, on_click=lambda: setattr(st.session_state, 'page_number', st.session_state.page_number + 1))
                p4.button("⏭", key="p_l", disabled=(st.session_state.page_number>=total_pages-1), use_container_width=True, on_click=lambda: setattr(st.session_state, 'page_number', total_pages - 1))
        else:
            st.info("No active signals.")

    # 1. RENDER CACHED UI
    if 'combined_active_df' not in st.session_state:
         st.session_state.combined_active_df = pd.DataFrame()
    render_active_list(st.session_state.combined_active_df)

    # 2. CHECK BACKGROUND STATUS
    status = thread_manager.get_status()
    
    # Check if thread finished
    if status.get('ready', False):
        # Ingest Data
        # We process result
        data = status.get('data')
        if data:
            st.session_state['runic_history_df'] = data['history']
            st.session_state['combined_active_df'] = data['active']
            st.session_state['runic_24h_return'] = data['metrics'][0]
            st.session_state['runic_12h_return'] = data['metrics'][1]
            st.session_state['last_runic_fetch'] = data['timestamp']
            
            thread_manager.ack_result()
            st.rerun()
    
    # Check if running
    # Check if running
    is_running = status.get('running', False)
    if is_running:
        progress_val = status.get('progress', 0)
        st.progress(progress_val)
        st.markdown(f"<div style='text-align: center; color: #ffd700; font-size: 0.8rem; margin-top: -15px;'>🔮 Consulting the Oracle... ({progress_val}%)</div>", unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()

    # 3. TRIGGER NEW RUN IF NEEDED
    now = time.time()
    should_start = False
    
    # Manual Trigger
    if refresh_click and not is_running:
        should_start = True
        
    # Auto Trigger
    elif not is_running:
        if 'last_runic_fetch' not in st.session_state:
            should_start = True
        elif now - st.session_state.get('last_runic_fetch', 0) > 115:
            should_start = True
            
    if should_start:
        started = thread_manager.start_run()
        if started:
            # Launch Thread
            t = threading.Thread(target=run_runic_analysis)
            t.start()
            st.rerun()
# --- Main Dashboard Logic (Simplified) ---
show_take_only = True # Default behavior

# Move Logic Inside Main Container
# --- CSS for Runic Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lato:wght@400;700&display=swap');

    /* Grand Portal Styling */
    /* Remove default top padding */
    .block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* SUPER ROBUST MAGICAL BORDER SELECTOR */
    /* Target only explicit border wrappers */
    div[data-testid="stVerticalBlockBorderWrapper"],
    .stVerticalBlockBorderWrapper {
        background-color: #0b0c15;
        /* Ultra-Thick Magical Frame */
        border: 2px solid #ffd700 !important;
        border-radius: 8px !important;
        
        /* Maximum Glow Power */
        box-shadow: 
            0 0 25px rgba(255, 215, 0, 0.6) !important, /* Outer Gold Halo */
            0 0 50px rgba(197, 160, 89, 0.4) !important, /* Distant Haze */
            inset 0 0 30px #000000 !important;          /* Deep Void Inner */
            
        padding: 10px !important;
        margin-bottom: 8px !important;
        position: relative;
        z-index: 1;
    }

    /* Epic Invoke Button (Primary Only) */
    div.stButton > button[kind="primary"] {
        width: 100%;
        height: 60px;
        background: linear-gradient(135deg, #c5a059 0%, #8a6e3c 100%);
        color: #000;
        font-family: 'Cinzel', serif;
        font-size: 2.4rem;
        font-weight: bold;
        border: 2px solid #ffd700;
        box-shadow: 0 0 20px rgba(197, 160, 89, 0.6);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(197, 160, 89, 0.9);
        color: #fff;
    }
    
    /* Transparent Secondary Buttons (Nav & Alerts) */
    div.stButton > button[kind="secondary"] {
        height: auto !important;
        min-height: 15px;
        padding: 2px 3px !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: #888 !important;
        box-shadow: none !important;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stButton > button[kind="secondary"]:active,
    div.stButton > button[kind="secondary"]:focus {
        color: #ffd700 !important;
        text-shadow: 0 0 5px rgba(255, 215, 0, 0.5);
        border: 1px solid transparent !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }


    .runic-header {
        text-align: center;
        color: #c5a059;
        font-size: 1rem;
        font-weight: bold;
        border-bottom: 1px solid #4a3b22;
        padding-bottom: 2px;
        margin-bottom: 4px;
        margin-top: -15px;
        text-shadow: 0 0 5px #c5a059;
    }

    .runic-item {
        display: flex;
        align-items: center;
        background: linear-gradient(90deg, rgba(20, 20, 30, 0.8) 0%, rgba(35, 35, 50, 0.8) 100%);
        border: 1px solid #4a4a60;
        border-left: 4px solid #444;
        margin-bottom: 2.2px;
        padding: 3px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .runic-item:hover {
        box-shadow: 0 0 10px rgba(197, 160, 89, 0.2);
        border-color: #c5a059;
    }

    .runic-icon {
        font-size: 25px;
        margin-right: 12px;
        width: 30px;
        text-align: center;
    }

    .runic-content {
        flex-grow: 1;
    }

    .runic-title {
        font-family: 'Lato', sans-serif;
        font-weight: bold;
        font-size: 1.0rem;
        color: #e0e0e0;
        margin: 0;
    }

    .runic-subtitle {
        font-family: 'Lato', sans-serif;
        font-size: 0.85rem;
        color: #888;
        margin: 1px 0 0 0;
    }

    .bullish { border-left-color: #00ff88; }
    .bearish { border-left-color: #ff3344; }
    

    .bullish .runic-icon { color: #00ff88; text-shadow: 0 0 8px rgba(0, 255, 136, 0.5); }
    .bearish .runic-icon { color: #ff3344; text-shadow: 0 0 8px rgba(255, 51, 68, 0.5); }

    /* Spell Cards Styling */
    .spell-card-container {
        display: flex;
        gap: 5px;
        justify-content: center;
        margin-bottom: 8px;
        margin-top: 5px;
    }
    .spell-card {
        flex: 1;
        background-color: #0b0c15; /* Dark background */
        border: 2px solid; 
        border-radius: 6px; /* Slightly rounded corners */
        padding: 2px 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Cinzel', serif;
        font-weight: bold;
        font-size: 0.8rem;
        position: relative;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: inset 0 0 15px rgba(0,0,0,0.8); /* Inner depth */
    }
    
    .spell-card-day {
        border-color: #c5a059;
        color: #c5a059;
        box-shadow: 
            inset 0 0 10px rgba(197, 160, 89, 0.2),
            0 0 8px rgba(197, 160, 89, 0.5); /* Outer Glow */
    }
    
    .spell-card-week {
        border-color: #00eaff;
        color: #00eaff;
        box-shadow: 
            inset 0 0 10px rgba(0, 234, 255, 0.2),
            0 0 8px rgba(0, 234, 255, 0.5); /* Outer Glow */
    }
    
    .spell-icon {
        margin-right: 6px;
        font-size: 0.95rem;
    }
    
    .spell-value {
        color: #fff;
        margin-left: 5px;
        text-shadow: 0 0 5px currentColor;
    }

    /* Arcane Portal Header */
    .arcane-header-container {
        text-align: center;
        margin-bottom: 4px;
        position: relative;
        padding: 3px 0;
        background: radial-gradient(circle at center, rgba(11, 12, 21, 0) 0%, rgba(11, 12, 21, 0.8) 100%);
        border-bottom: 2px solid transparent; 
        border-image: linear-gradient(90deg, transparent, #c5a059, transparent) 1;
    }

    .arcane-title {
        font-family: 'Cinzel', serif;
        font-size: 2.0rem;
        font-weight: 700;
        color: #f0e6d2; /* Light gold/parchment base */
        text-transform: uppercase;
        letter-spacing: 7px;
        
        /* Metallic Gradient Text */
        background: linear-gradient(to bottom, #fff 0%, #ffd700 50%, #c5a059 51%, #8a6e3c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        
        /* Glow and Shadow */
        filter: drop-shadow(0 0 2px rgba(0,0,0,0.8)) drop-shadow(0 0 10px rgba(197, 160, 89, 0.6));
        
        position: relative;
        display: inline-block;
    }
    
    /* Decorative side wings (pseudo-elements difficult in inline CSS, using separate spans or borders) */
    .arcane-decoration {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00eaff, transparent);
        margin-top: 5px;
        box-shadow: 0 0 10px #00eaff;
        opacity: 0.7;
    }

    /* Compact Multiselect Tags - LIGHT BLUE UPDATE */
    span[data-baseweb="tag"] {
        background-color: #00f0ff !important;
        color: #000000 !important;
        font-size: 0.8rem !important;
        font-weight: bold !important;
        padding: 0px 4px !important;
        height: 22px !important;
        margin-top: 2px !important;
        margin-bottom: 2px !important;
        border: 1px solid #00f0ff;
        box-shadow: 0 0 5px #00f0ff;
    }
    
    /* Global Compactness Overrides */
    .stVerticalBlockBorderWrapper {
        margin-bottom: 4px !important; 
        padding: 2px !important; 
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }
    
    /* Responsive Ticker Tape Hack */
    /* Mobile First: Default is handled by Python height=80 */
    
    /* Desktop Override: Shrink to 60px */
    @media (min-width: 900px) {
        div[style*="height: 81px"],
        div[style*="height:81px"] {
            height: 60px !important;
            min-height: 60px !important;
        }
        iframe[height="81"],
        iframe[style*="height: 81px"] {
            height: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- ARCANE PORTAL HEADER ---
st.markdown("""
    <div class="arcane-header-container">
        <div class="arcane-title">Arcane Portal</div>
        <div class="arcane-decoration"></div>
    </div>
""", unsafe_allow_html=True)

# --- MARKET TICKER TAPE ---
st.components.v1.html("""
<div class="tradingview-widget-container" style="width: 100%; height: 100%;">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
  "symbols": [
    {
      "proName": "BINANCE:BTCUSDT",
      "title": "Bitcoin"
    },
    {
      "proName": "BINANCE:ETHUSDT",
      "title": "Ethereum"
    },
    {
      "proName": "BINANCE:SOLUSDT",
      "title": "Solana"
    },
    {
      "proName": "BINANCE:DOGEUSDT",
      "title": "Dogecoin"
    },
    {
      "proName": "BINANCE:ARBUSDT",
      "title": "Arbitrum"
    },
    {
      "proName": "BINANCE:AVAXUSDT",
      "title": "Avalanche"
    },
    {
      "proName": "BINANCE:LINKUSDT",
      "title": "Chainlink"
    },
    {
      "proName": "CAPITALCOM:DXY",
      "title": "DXY"
    },
    {
      "proName": "CRYPTOCAP:BTC.D",
      "title": "BTC.D"
    },
    {
      "proName": "OANDA:SPX500USD",
      "title": "S&P 500"
    },
    {
      "proName": "OANDA:XAUUSD",
      "title": "Gold"
    },
    {
      "proName": "OANDA:XAGUSD",
      "title": "Silver"
    },
    {
      "proName": "TVC:USOIL",
      "title": "US Oil"
    },
    {
      "proName": "BINANCE:BNBUSDT",
      "title": "BNB"
    },
    {
      "proName": "FX:EURUSD",
      "title": "EUR/USD"
    },
    {
      "proName": "FX:AUDUSD",
      "title": "AUD/USD"
    }
  ],
  "showSymbolLogo": true,
  "colorTheme": "dark",
  "isTransparent": true,
  "displayMode": "regular",
  "locale": "en"
}
  </script>
</div>
""", height=81, scrolling=False)

# --- Debug Sidebar (Safe Location) --- 
with st.sidebar.expander("Debug Info", expanded=False):
    st.write(f"CWD: {os.getcwd()}")
    st.write("Model Files:")
    st.write([f for f in os.listdir('.') if f.endswith('.pkl')])
    st.write("Loaded Models:")
    if 'models' in globals():
        st.write(list(models.keys()))
    st.write(f"Assets: {len(ASSETS)}")
    if st.button("Run Pipeline Manually"):
        import pipeline
        pipeline.run_pipeline()
        st.success("Pipeline Run Initiated!")
        
    st.divider()
    st.write("Playwright Scraper (Mango)")
    if st.button("Invoke Scraper"):
        with st.spinner("Invoking Playwright Scraper..."):
            try:
                import subprocess
                subprocess.Popen(["python", "scrape_tv_indicators.py"])
                st.info("Scraper started in background. Refresh in 1-2 mins.")
            except Exception as e:
                st.error(f"Failed to start scraper: {e}")

# Layout Columns
col_left, col_center, col_right = st.columns([0.25, 0.5, 0.25], gap="small")

# --- CENTER COLUMN: MAIN PORTAL ---
with col_center:
    with st.container(border=True):
        # --- TOP HEADER: MANA & SPELLS ---
        # Resets Logic (Integrated here)
        try:
            now = pd.Timestamp.now()
            # Daily Reset
            if now.floor('D') > st.session_state.last_reset_day:
                st.session_state.spells_day = 2
                st.session_state.last_reset_day = now.floor('D')
            # Weekly Reset
            current_week = now.to_period('W').start_time
            if current_week > st.session_state.last_reset_week:
                st.session_state.spells_week = 5
                st.session_state.last_reset_week = current_week
        except: pass

        # Layout: [Mana Bar (70%)] [Spells (30%)]
        c_mana, c_spells = st.columns([0.7, 0.3], gap="medium")
        
        with c_mana:
             st.markdown('<div class="runic-header" style="text-align: left; margin-top: 0;">MANA POOL</div>', unsafe_allow_html=True)
             mana_pct = max(0, min(100, (st.session_state.mana / 425) * 100))
             st.markdown(f"""
                <div style="background-color: #0b0c15; border: 1px solid #444; border-radius: 6px; height: 22px; margin-bottom: 2px; position: relative; box-shadow: inset 0 0 10px #000; margin-top: 5px;">
                    <div style="
                        background: linear-gradient(90deg, #00eaff 0%, #00ff88 100%);
                        width: {mana_pct}%; 
                        height: 100%; 
                        border-radius: 5px; 
                        box-shadow: 0 0 15px rgba(0, 255, 136, 0.6); 
                        transition: width 0.5s ease-out;
                    "></div>
                    <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; text-shadow: 0 1px 4px black; letter-spacing: 1px; font-size: 0.8rem;">
                        {st.session_state.mana} / 425
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with c_spells:
            st.markdown('<div class="runic-header" style="text-align: right; margin-top: 0;">SPELLS</div>', unsafe_allow_html=True)
            s_day = st.session_state.spells_day
            s_week = st.session_state.spells_week
            st.markdown(f"""
                <div class="spell-card-container" style="margin-top: 5px; margin-bottom: 0;">
                    <div class="spell-card spell-card-day" style="padding: 1px 4px; font-size: 0.75rem;">
                        Day: <span class="spell-value">{s_day}</span>
                    </div>
                    <div class="spell-card spell-card-week" style="padding: 1px 4px; font-size: 0.75rem;">
                        Week: <span class="spell-value">{s_week}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---") # Divider before Navbar
        # Interactive Navbar
        if 'active_tab' not in st.session_state: st.session_state.active_tab = 'PORTAL'
        
        def set_tab(t):
            st.session_state.active_tab = t
            
        # Adjusted columns for remaining tabs (History button removed)
        c2, c3, c4, c5 = st.columns([1, 1, 1, 1])
        
        # Determine labels with indicator
        lbl_portal = "✨ PORTAL" if st.session_state.active_tab=='PORTAL' else "PORTAL"
        lbl_shield = "🛡️ SHIELD" if st.session_state.active_tab=='RISK' else "SHIELD"
        lbl_rules = "📜 SCRIPTURE" if st.session_state.active_tab=='SCRIPTURE' else "SCRIPTURE"
        lbl_spell = "📘 SPELLBOOK" if st.session_state.active_tab=='SPELLBOOK' else "SPELLBOOK"

        c2.button(lbl_portal, use_container_width=True, type="secondary", on_click=set_tab, args=('PORTAL',))
        
        c3.button(lbl_shield, use_container_width=True, type="secondary", on_click=set_tab, args=('RISK',))
        
        c4.button(lbl_rules, use_container_width=True, type="secondary", on_click=set_tab, args=('SCRIPTURE',))
        
        c5.button(lbl_spell, use_container_width=True, type="secondary", on_click=set_tab, args=('SPELLBOOK',))
        
        # Remove standard divider and use negative margin wrapper for tighter fit
        st.markdown('<div style="margin-top: -10px; margin-bottom: -10px;"><hr style="margin: 5px 0; border-color: #333;"></div>', unsafe_allow_html=True)
        
        if st.session_state.active_tab == 'HISTORY':
             st.markdown("### 📜 Signal History")
             
             hist_df = st.session_state.get('runic_history_df', pd.DataFrame())

             
             if not hist_df.empty and '_sort_key' in hist_df.columns:
                 # Standardize
                 try:
                    hist_df['_sort_key'] = pd.to_datetime(hist_df['_sort_key'], utc=True)
                 except: pass
                 
                 # Options Layout
                 opt_col1, opt_col2 = st.columns([0.4, 0.6])
                 with opt_col1:
                     # Toggle for 24H Only
                     show_24h_only = st.checkbox("Show last 24 Hours Only", value=False)
                     # Toggle for 7 Days Only
                     show_7d_only = st.checkbox("Show last 7 Days Only", value=False)
                     # Toggle for 30 Days Only
                     show_30d_only = st.checkbox("Show last 30 Days Only", value=False)
                     # Toggle for Open Trades Only (New)
                     show_open_only = st.checkbox("Show Open Trades Only", value=False)
                 
                 # 1. Filter Time
                 if show_24h_only:
                     cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)
                     filtered_df = hist_df[hist_df['_sort_key'] >= cutoff].copy()
                 elif show_7d_only:
                     cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
                     filtered_df = hist_df[hist_df['_sort_key'] >= cutoff].copy()
                 elif show_30d_only:
                     cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30)
                     filtered_df = hist_df[hist_df['_sort_key'] >= cutoff].copy()
                 else:
                     filtered_df = hist_df.copy()
                     
                 # 2. Filter Open Trades (New)
                 if show_open_only:
                     filtered_df = filtered_df[filtered_df['Status'] == 'OPEN']

                 # --- TRADING PLAN FILTERING ---
                 if st.session_state.get('plan_active', False):
                     # 1. Confidence
                     plan_conf = st.session_state.get('plan_min_conf', 55)
                     def _parse_conf_plan_hist(x):
                         try: return float(str(x).replace('%',''))
                         except: return 0.0
                     if 'Confidence' in filtered_df.columns:
                         filtered_df = filtered_df[filtered_df['Confidence'].apply(_parse_conf_plan_hist) >= plan_conf]
                     
                     # 2. Timeframes
                     plan_tfs = st.session_state.get('plan_allowed_tfs', [])
                     if plan_tfs:
                         filtered_df = filtered_df[filtered_df['Timeframe'].apply(lambda tf: any(p_tf in str(tf) or (p_tf == "1H" and "1 Hour" in str(tf)) or (p_tf == "4H" and "4 Hours" in str(tf)) or (p_tf == "12H" and "12 Hours" in str(tf)) or (p_tf == "1D" and "1 Day" in str(tf)) for p_tf in plan_tfs))]
                 if show_open_only:
                     if 'Status' in filtered_df.columns:
                         filtered_df = filtered_df[filtered_df['Status'] == 'OPEN']
                     
                 # 3. Timeframe & Strategy Filter
                 if 'Timeframe' in hist_df.columns:
                     # Ensure Strategy Column Exists for Filtering
                     if 'Strategy' not in hist_df.columns:
                         hist_df['Strategy'] = 'WizardWave'
                     else:
                         hist_df['Strategy'] = hist_df['Strategy'].fillna('WizardWave')
                         
                     # Timeframes
                     tf_order = {
                        "15m": 0, "15 Minutes": 0,
                        "1H": 2, "1 Hour": 2,
                        "4H": 3, "4 Hours": 3,
                        "12H": 4, "12 Hours": 4,
                        "1D": 5, "1 Day": 5,
                        "4D": 6, "4 Days": 6
                     }
                     
                     unique_tfs = hist_df['Timeframe'].unique().tolist()
                     sorted_tfs = sorted(unique_tfs, key=lambda x: tf_order.get(x, 99))
                     
                     # Strategies
                     unique_strats = sorted(hist_df['Strategy'].unique().tolist())
                     
                     with opt_col2:
                         # 1. Timeframe Select
                         if "history_tf_filter" not in st.session_state:
                             st.session_state.history_tf_filter = sorted_tfs
                         
                         selected_short = st.multiselect("Timeframes", options=sorted_tfs, key="history_tf_filter")
                         
                         # 2. Strategy Select
                         # Initialize default to all if key missing
                         if "history_strat_filter" not in st.session_state:
                              st.session_state.history_strat_filter = unique_strats
                         selected_strats = st.multiselect("Strategies", options=unique_strats, key="history_strat_filter")
                         
                     # Apply Filters
                     # 1. TF
                     if selected_short:
                         filtered_df = filtered_df[filtered_df['Timeframe'].isin(selected_short)]
                     else:
                         filtered_df = pd.DataFrame(columns=filtered_df.columns)
                         
                     # 2. Strategy
                     if selected_strats:
                          # Ensure target df has the col
                          if 'Strategy' not in filtered_df.columns: filtered_df['Strategy'] = 'WizardWave'
                          else: filtered_df['Strategy'] = filtered_df['Strategy'].fillna('WizardWave')
                          
                          filtered_df = filtered_df[filtered_df['Strategy'].isin(selected_strats)]
                     else:
                          filtered_df = pd.DataFrame(columns=filtered_df.columns)
                         
                 # Sort newest first
                 filtered_df = filtered_df.sort_values(by='_sort_key', ascending=False)

                 # Freshness Check
                 if not filtered_df.empty:
                     latest_ts = filtered_df['_sort_key'].max()
                     # Format if possible
                     try:
                         latest_str = latest_ts.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M:%S EST')
                     except:
                         latest_str = str(latest_ts)
                     st.caption(f"Last Signal: {latest_str}")
                 
                 # Summary Stats
                 total_trades = len(filtered_df)
                 total_ret = filtered_df['Return_Pct'].sum()
                 winners = len(filtered_df[filtered_df['Return_Pct'] > 0])
                 win_rate = winners / total_trades if total_trades > 0 else 0
                 
                 # Metrics
                 m1, m2, m3 = st.columns(3)
                 
                 pnl_label = "PnL Sum (All Time)"
                 if show_24h_only: pnl_label = "PnL Sum (24hrs)"
                 elif show_7d_only: pnl_label = "PnL Sum (7 Days)"
                 elif show_30d_only: pnl_label = "PnL Sum (30 Days)"
                 
                 m1.metric(pnl_label, f"{total_ret:.2%}")
                 m2.metric("Trades", total_trades)
                 m3.metric("Win Rate", f"{win_rate:.0%}")
                 
                 st.divider()
                 
                 # --- Reference Parameters (TP/SL) ---
                 with st.expander("🛡️ Current Strategy Parameters", expanded=True):
                     sys_c1, sys_c2 = st.columns(2)
                     
                     # HTF (Swing - 1D)
                     with sys_c1:
                        htf_c = config['models']['1d']['triple_barrier']
                        st.caption(f"**Swing (1D)** | Timeout: {htf_c.get('time_limit_bars', 21)} Bars")
                        # Format for display
                        st.markdown(f"""
                        - **Crypto:** TP `{htf_c['crypto_pt']:.1%}` / SL `{htf_c['crypto_sl']:.1%}`
                        - **TradFi:** TP `{htf_c['trad_pt']:.1%}` / SL `{htf_c['trad_sl']:.1%}`
                        - **Forex:**  TP `{htf_c['forex_pt']:.1%}` / SL `{htf_c['forex_sl']:.1%}`
                        """)
                     
                     # LTF (Scalp - 15m)
                     with sys_c2:
                        ltf_c = config['models']['15m']['triple_barrier']
                        st.caption(f"**Scalp (15m)** | Timeout: {ltf_c.get('time_limit_bars', 12)} Bars")
                        
                        if ltf_c.get('crypto_use_dynamic'):
                            c_str = f"⚡ *Dynamic* (σ x {ltf_c.get('crypto_dyn_pt_k')})"
                        else:
                            c_str = f"TP `{ltf_c['crypto_pt']:.1%}` / SL `{ltf_c['crypto_sl']:.1%}`"
                            
                        st.markdown(f"""
                        - **Crypto:** {c_str}
                        - **TradFi:** TP `{ltf_c['trad_pt']:.1%}` / SL `{ltf_c['trad_sl']:.1%}`
                        - **Forex:**  TP `{ltf_c['forex_pt']:.1%}` / SL `{ltf_c['forex_sl']:.1%}`
                        """)

                 # --- MY TRADES SECTION ---
                 with st.expander("⭐ My Trades", expanded=False):
                     my_trades_list = manage_trades.load_trades()
                     if my_trades_list:
                         mt_df = pd.DataFrame(my_trades_list)
                         st.dataframe(mt_df, use_container_width=True)
                         
                         if st.button("Clear All"):
                             manage_trades.save_trades_list([])
                             st.rerun()
                     else:
                         st.info("No trades saved yet. Check the box in the history table to save a trade.")

                 # Simplify cols
                 if not filtered_df.empty:
                     # Backward compatibility
                     if 'Strategy' not in filtered_df.columns:
                         filtered_df['Strategy'] = 'WizardWave'
                     else:
                         filtered_df['Strategy'] = filtered_df['Strategy'].fillna('WizardWave')

                     # Format TP/SL for Display
                     if 'Raw_TP' in filtered_df.columns:
                         filtered_df['TP'] = filtered_df['Raw_TP'].fillna(0.0).apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                     else:
                         filtered_df['TP'] = "0.0000"
                         
                     if 'Raw_SL' in filtered_df.columns:
                         filtered_df['SL'] = filtered_df['Raw_SL'].fillna(0.0).apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
                     else:
                         filtered_df['SL'] = "0.0000"

                     display_cols = ['Time', 'Asset', 'Timeframe', 'Type', 'Confidence', 'Price', 'TP', 'SL', 'Exit Time', 'Return_Pct', 'Status', 'Strategy']
                     # Fill Status if missing
                     if 'Status' not in filtered_df.columns:
                         filtered_df['Status'] = 'CLOSED'
                         
                     # Format Return
                     filtered_df['Return'] = filtered_df['Return_Pct'].apply(lambda x: f"{x:.2%}")
                     
                     # --- SAVED STATUS LOGIC ---
                     saved_keys = manage_trades.get_saved_keys()
                     filtered_df['Saved'] = filtered_df.apply(lambda x: (x.get('Asset'), x.get('Time')) in saved_keys, axis=1)
                     
                     # Reorder: Saved first
                     final_cols = ['Saved'] + display_cols + ['Return']
                     
                     # Prepare for Editor
                     # We must reset index to ensure 0..N alignment for edited_rows
                     editor_df = filtered_df[final_cols].reset_index(drop=True)
                     
                     # Render Editor
                     edited_df = st.data_editor(
                         editor_df,
                         column_config={
                             "Saved": st.column_config.CheckboxColumn("Save", help="Check to save this trade to 'My Trades'", default=False),
                             "Return_Pct": None, 
                             "Return": st.column_config.TextColumn("Return"),
                             "Type": st.column_config.TextColumn("Signal Type"),
                             "Strategy": st.column_config.TextColumn("Strategy"),
                             "Timeframe": st.column_config.TextColumn("TF"),
                         },
                         use_container_width=True,
                         hide_index=True,
                         key="history_editor",
                         disabled=[c for c in final_cols if c != 'Saved'] # Disable editing other cols
                     )
                     
                     # Handle Changes
                     if "history_editor" in st.session_state:
                         changes = st.session_state["history_editor"].get("edited_rows", {})
                         if changes:
                             any_change = False
                             for idx, change_dict in changes.items():
                                 if 'Saved' in change_dict:
                                     is_saved = change_dict['Saved']
                                     # Get data from source DF (snapshot)
                                     try:
                                         if int(idx) < len(editor_df):
                                             row_data = editor_df.iloc[int(idx)].to_dict()
                                             manage_trades.toggle_trade(row_data, is_saved)
                                             any_change = True
                                     except: pass
                             
                             if any_change:
                                 # Clean up state? Streamlit handles this on rerun if we don't persist key?
                                 # We just rerun.
                                 # To avoid infinite loop, we rely on the fact that next run 'Saved' col will match user input
                                 # So 'edited_rows' (diff) will be empty?
                                 # Yes, mostly.
                                 st.rerun()
                 else:
                     st.info("No trades in this period.")
                 
             else:
                 st.info("No history available. Please wait for the Runic Alerts to refresh.")

        elif st.session_state.active_tab == 'RISK':
            # Ensure accounts are loaded/initialized if not present
            if 'user_accounts' not in st.session_state or not st.session_state.user_accounts:
                 # Initialize default if empty, or try to load
                 init_prop_accounts()

            # Check for Calculator Mode
            view_mode = st.session_state.get('active_view_mode', 'default')
            active_sig = st.session_state.get('active_signal', {})

            if view_mode == 'calculator' and active_sig:
                 # --- POSITION SIZE CALCULATOR ---
                 c_back, c_title = st.columns([0.2, 0.8])
                 with c_back:
                     if st.button("⬅ Close"):
                         st.session_state.active_view_mode = 'default'
                         st.rerun()
                 with c_title:
                     st.markdown(f"### 🧮 Position Wizard: {active_sig.get('Asset', 'Unknown')}")
                 
                 st.divider()
                 
                 # 1. Select Account
                 accounts = st.session_state.get('user_accounts', [])
                 if not accounts: 
                     accounts = [{'name': 'Default 50k', 'currentBalance': 50000}]
                     
                 acc_names = [a['name'] for a in accounts]
                 selected_acc_name = st.selectbox("Select Grimoire (Account)", acc_names)
                 
                 # Find selected account object
                 selected_acc = next((a for a in accounts if a['name'] == selected_acc_name), accounts[0])
                 balance = float(selected_acc.get('currentBalance', 50000))
                 
                 # 2. Select Risk
                 risk_pct_map = {"0.25%": 0.0025, "0.50%": 0.005, "1.00%": 0.01}
                 risk_label = st.radio("Risk Enchantment", list(risk_pct_map.keys()), horizontal=True, index=1)
                 risk_val = risk_pct_map[risk_label]
                 
                 risk_amt = balance * risk_val
                 
                 st.info(f"**Risk Amount:** ${risk_amt:.2f} (on ${balance:,.0f} balance)")
                 
                 # 3. Inputs (Entry / SL)
                 def parse_price(v):
                     try: return float(str(v).replace(',', ''))
                     except: return 0.0
                 
                 default_entry = st.session_state.get('calc_entry_input', parse_price(active_sig.get('Entry_Price', 0)))
                 default_sl = parse_price(active_sig.get('Stop_Loss', 0))
                 
                 c_in1, c_in2 = st.columns(2)
                 with c_in1:
                     entry_in = st.number_input("Entry Price", value=default_entry, format="%.5f")
                 with c_in2:
                     sl_in = st.number_input("Stop Loss", value=default_sl, format="%.5f")
                     
                 # 4. Calculation
                 st.divider()
                 
                 if entry_in > 0 and sl_in > 0 and entry_in != sl_in:
                     dist_pct = abs(entry_in - sl_in) / entry_in
                     pos_size_value = risk_amt / dist_pct
                     units = pos_size_value / entry_in
                     leverage = pos_size_value / balance
                     
                     mc1, mc2, mc3 = st.columns(3)
                     mc1.metric("Position Size ($)", f"${pos_size_value:,.2f}")
                     mc2.metric("Units", f"{units:.4f}")
                     mc3.metric("Est. Leverage", f"{leverage:.2f}x")
                     
                     # Fixed Formatting
                     st.success(f"To risk ${risk_amt:.2f} ({risk_label}), open a position size of ${pos_size_value:,.2f}.")
                 else:
                     st.warning("Please enter valid Entry and Stop Loss prices to cast the calculation.")

            else:
                render_prop_risk()
            
        elif st.session_state.active_tab == 'SPELLBOOK':
             st.markdown("### 📘 Grimoire of Knowledge")
             st.markdown("Upload your trade screenshots here for NLP analysis and feedback.")
             st.file_uploader("Analyze Rune (Upload)", type=['png', 'jpg'], label_visibility="collapsed")
             
             # --- REALMS (Moved from Right Col) ---
             st.markdown("---")
             st.markdown('<div class="runic-header">REALMS</div>', unsafe_allow_html=True)
             
             # Session Data (EST)
             sessions = [
                 {'name': 'Sydney', 'start': 17, 'end': 26}, # 5PM - 2AM (Next day = +24 = 26)
                 {'name': 'Tokyo', 'start': 19, 'end': 28}, # 7PM - 4AM (Next day = +24 = 28)
                 {'name': 'London', 'start': 3, 'end': 11},  # 3AM - 11AM
                 {'name': 'New York', 'start': 8, 'end': 17} # 8AM - 5PM
             ]
             
             now_est = pd.Timestamp.now(tz='America/New_York')
             curr_hour = now_est.hour
             curr_min = now_est.minute
             
             # Calculate current time percentage for marker (00:00 to 24:00)
             current_time_pct = ((curr_hour * 60 + curr_min) / (24 * 60)) * 100
             
             session_html = ""
             
             for sess in sessions:
                 s_real = sess['start']
                 e_real = sess['end']
                 
                 is_active = False
                 s_mod = s_real % 24
                 e_mod = e_real % 24
                 if s_mod > e_mod: # Wraps midnight
                     if curr_hour >= s_mod or curr_hour < e_mod: is_active = True
                 else:
                     if s_mod <= curr_hour < e_mod: is_active = True
                 
                 # Text Logic
                 status_text = ""
                 if is_active:
                     target_h = e_mod
                     if target_h < curr_hour: target_h += 24
                     
                     diff_h = target_h - curr_hour
                     diff_m = 0 - curr_min
                     total_min = diff_h * 60 + diff_m
                     h_left = total_min // 60
                     m_left = total_min % 60
                     status_text = f"Ends in {h_left}hr {m_left}min"
                     bar_color = "#00ff88" # Green
                     text_color = "#fff"
                 else:
                     target_h = s_mod
                     if target_h < curr_hour: target_h += 24 
                     
                     diff_h = target_h - curr_hour
                     diff_m = 0 - curr_min
                     total_min = diff_h * 60 + diff_m
                     h_left = total_min // 60
                     m_left = total_min % 60
                     
                     status_text = f"Begins in {h_left}hr {m_left}min"
                     bar_color = "#4a4a60" # Grey
                     text_color = "#aaa"
                     
                 # Render Bars
                 bars_svg = ""
                 def draw_rect(s, e, col):
                     width = (e - s) / 24 * 100
                     left = (s / 24) * 100
                     return f'<div style="position: absolute; left: {left}%; top: 5px; height: 20px; width: {width}%; background-color: {col}; border-radius: 4px; display: flex; align-items: center; padding-left: 5px; white-space: nowrap; overflow: hidden;"></div>'
                 
                 if s_real >= 24: pass
                 elif e_real > 24:
                     bars_svg += draw_rect(s_real, 24, bar_color)
                     bars_svg += draw_rect(0, e_real-24, bar_color)
                 else:
                     bars_svg += draw_rect(s_real, e_real, bar_color)
                     
                 session_html += f"""
     <div class="realm-row" title="{status_text}" style="margin-bottom: 8px; position: relative; height: 30px; display: flex; align-items: center;">
         <div style="width: 70px; font-size: 0.75rem; font-weight: bold; color: {text_color if not is_active else '#fff'}; text-align: right; margin-right: 10px;">{sess['name']}</div>
         <div style="flex-grow: 1; position: relative; height: 100%; background: #1a1a2e; border-radius: 4px; overflow: hidden;">
             {bars_svg}
             <div style="position: absolute; top:0; left:5px; font-size: 0.7rem; color: {text_color if is_active else '#888'}; line-height: 30px; font-weight: bold; z-index: 2; text-shadow: 0 1px 3px rgba(0,0,0,0.9);">{status_text if is_active else ''}</div>
         </div>
     </div>
                 """
     
             st.markdown(f"""
     <div style="padding: 10px 0;">
     <div style="display: flex; margin-left: 80px; font-size: 0.6rem; color: #666; margin-bottom: 5px; justify-content: space-between;">
     <span>12AM</span><span>4AM</span><span>8AM</span><span>12PM</span><span>4PM</span><span>8PM</span><span>12AM</span>
     </div>
     {session_html}
     <div style="text-align: center; font-size: 0.7rem; color: #666; margin-top: 5px;">
     Current Time: {now_est.strftime('%H:%M')} EST
     </div>
     <div class="realm-overlay"></div>
     </div>
     <style>
     .realm-overlay {{
     position: absolute;
     left: calc(80px + (100% - 80px) * ({current_time_pct:.2f}/100));
     top: 40px; 
     bottom: 25px;
     width: 2px;
     background-color: #ffd700;
     box-shadow: 0 0 5px #ffd700;
     z-index: 10;
     pointer-events: none;
     }}
     </style>
             """, unsafe_allow_html=True)
             
             # --- ORACLE (Moved from Right Col) ---
             st.markdown("---")
             st.markdown('<div class="runic-header">ORACLE</div>', unsafe_allow_html=True)
             
             # Economic Calendar
             # Note: Dates are best estimates based on standard schedules (CPI ~13th, NFP ~1st Friday, FOMC ~Wed)
             economic_events = [
                 # late 2025
                 {"event": "PCE Price Index", "datetime": "2025-12-23 08:30:00"},
                 
                 # Jan 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-01-09 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-01-13 08:30:00"},
                 {"event": "PPI Inflation Data", "datetime": "2026-01-14 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-01-28 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-01-30 08:30:00"},
                 
                 # Feb 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-02-06 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-02-11 08:30:00"}, # Estimated
                 {"event": "PCE Price Index", "datetime": "2026-02-27 08:30:00"},

                 # Mar 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-03-06 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-03-12 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-03-18 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-03-27 08:30:00"},

                 # Apr 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-04-03 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-04-14 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-04-29 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-04-24 08:30:00"},

                 # May 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-05-08 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-05-13 08:30:00"},
                 {"event": "PCE Price Index", "datetime": "2026-05-29 08:30:00"},

                 # Jun 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-06-05 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-06-12 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-06-17 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-06-26 08:30:00"},
                 
                 # Jul 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-07-03 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-07-14 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-07-29 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-07-31 08:30:00"},

                 # Aug 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-08-07 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-08-13 08:30:00"},
                 {"event": "PCE Price Index", "datetime": "2026-08-28 08:30:00"},

                 # Sep 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-09-04 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-09-15 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-09-16 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-09-25 08:30:00"},

                 # Oct 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-10-02 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-10-13 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-10-28 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-10-30 08:30:00"},
                 
                 # Nov 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-11-06 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-11-13 08:30:00"},
                 {"event": "PCE Price Index", "datetime": "2026-11-25 08:30:00"},
                 
                 # Dec 2026
                 {"event": "Non-Farm Payrolls", "datetime": "2026-12-04 08:30:00"},
                 {"event": "CPI Inflation Data", "datetime": "2026-12-11 08:30:00"},
                 {"event": "FOMC Rate Decision", "datetime": "2026-12-09 14:00:00"},
                 {"event": "PCE Price Index", "datetime": "2026-12-23 08:30:00"},
             ]
             
             # Find Next Event
             now_est = pd.Timestamp.now(tz='America/New_York')
             next_event = None
             
             for e in economic_events:
                 dt = pd.Timestamp(e['datetime']).tz_localize('America/New_York')
                 if dt > now_est:
                     next_event = e
                     target_dt = dt
                     break
             
             if next_event:
                 # Calculate Countdown
                 diff = target_dt - now_est
                 days = diff.days
                 hours = diff.seconds // 3600
                 minutes = (diff.seconds % 3600) // 60
                 
                 # Format Date
                 date_str = target_dt.strftime("%b %d, %H:%M EST")
                 event_name = next_event['event'].upper()
                 
                 # Color Logic (Red for very close)
                 time_color = "white"
                 if days < 1: time_color = "#ff3344"
                 
                 # Load Background Image for Oracle
                 oracle_bg = ""
                 try:
                     import base64
                     with open("Crystall Ball.png", "rb") as img_file:
                         b64_ball = base64.b64encode(img_file.read()).decode()
                     oracle_bg = f"background-image: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.6)), url('data:image/png;base64,{b64_ball}'); background-size: cover; background-position: center;"
                 except Exception as e:
                     pass

                 st.markdown(f"""
                     <div style="
                         text-align: center; 
                         min-height: 200px; 
                         display: flex; 
                         flex-direction: column; 
                         justify-content: flex-start;
                         padding-top: 20px;
                         margin-bottom: 15px;
                         border-radius: 8px;
                         {oracle_bg}
                     ">
                         <div style="background: rgba(11, 12, 21, 0.85); padding: 15px; border-radius: 6px; border: 1px solid #4a4a60; margin: 0 15px; box-shadow: 0 0 15px rgba(0,0,0,0.8);">
                             <div style="font-size: 0.8rem; color: #a0c5e8; margin-bottom: 5px;">NEXT EVENT: <span style="color: #ffd700;">{event_name}</span></div>
                             <div style="font-size: 0.9rem; color: #ccc; margin-bottom: 5px;">{date_str}</div>
                             <div style="font-size: 2.2rem; font-weight: bold; color: {time_color}; text-shadow: 0 0 10px {time_color};">
                                 {days}d {hours}h {minutes}m
                             </div>
                         </div>
                     </div>
                 """, unsafe_allow_html=True)
             else:
                 st.markdown(f"""
                     <div style="text-align: center; min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
                         <div style="font-size: 0.8rem; color: #a0c5e8;">NO UPCOMING EVENTS</div>
                         <div style="font-size: 2.5rem; font-weight: bold; color: white; text-shadow: 0 0 10px #a0c5e8;">--:--:--</div>
                     </div>
                 """, unsafe_allow_html=True)
                 
             # --- GREAT SORCERER (Moved from Right Col) ---
             st.markdown("---")
             st.markdown('<div class="runic-header">GREAT SORCERER</div>', unsafe_allow_html=True)
             
             quotes = [
                 "The market is a mirror of the mind.",
                 "Clarity comes not from the chart, but from the discipline within.",
                 "Do not chase the dragon; let it come to you.",
                 "Patience is the wizard's greatest spell.",
                 "Risk is the mana you pay for the reward you seek.",
                 "A calm mind sees the trend; a chaotic mind sees only noise.",
                 "I am a risk manager. My edge is my patience. I don’t gamble; I execute a system.",
                 "I accept the outcome of any single trade because I am focused on the long-term survival of my capital.",
                 "Passion = Emotion | Commitment = Discipline",
                 "Reminder: You don’t have to trade everyday!",
                 "You are a robot executing code. You don’t “feel” or “hope” the market will move a certain way.",
                 "You are a sniper with 2 bullets. You are not a machine gunner spraying. You reject good trades to wait for great trades.",
                 "You are a risk manager, not a profit generator. Your job is to protect your capital. Profit is just a byproduct of good survival skills"
             ]
             import random
             selected_quote = random.choice(quotes)
             
             # Load Background Image
             bg_style = ""
             try:
                 import base64
                 with open("great_sorcerer.png", "rb") as img_file:
                     b64_img = base64.b64encode(img_file.read()).decode()
                 bg_style = f"background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.6)), url('data:image/png;base64,{b64_img}'); background-size: cover; background-position: center;"
             except Exception as e:
                 pass
             
             # Shuffle Button
             if st.button("🔮 Shuffle Wisdom", key="shuffle_wis"):
                 st.rerun()

             st.markdown(f"""
                 <div style="padding: 20px; border-radius: 10px; color: #f0f0f0; text-align: center; min-height: 250px; display: flex; align-items: flex-end; justify-content: center; {bg_style} box-shadow: 0 4px 15px rgba(0,0,0,0.5); border: 1px solid #4a4a60; margin-top: 10px;">
                     <div style="background: rgba(0,0,0,0.7); padding: 15px; border-radius: 6px; width: 100%; backdrop-filter: blur(2px);">
                         <div style="font-size: 1.1rem; font-style: italic; font-family: 'Georgia', serif;">“{selected_quote}”</div>
                     </div>
                 </div>
             """, unsafe_allow_html=True)

        elif st.session_state.active_tab == 'SCRIPTURE':
            st.markdown("""
            ### 📜 The Code of the Wizard
            
            **Goal**: To consistently get funded payouts.
            
            **My Identity**: "I am a risk manager. My edge is my patience. I don’t gamble; I execute a system. I accept the outcome of any single trade because I am focused on the long-term survival of my capital. If the setup is not perfect, I do not chase."
            <br>*Passion = Emotion | Commitment = Discipline*
            
            ---
            
            #### 🧠 Mental Frameworks
            
            **1. Systematic/Rule Based - Identity**: 
            You are a robot executing code. You don’t “feel” or “hope” the market will move a certain way. You see an If/Then statement. If A happens, Then I do B. If A does not happen, I do nothing. Your ego wants to be a genius who predicts the future. A system makes you a data entry clerk. You have to surrender your need to be "right" & replace it with "compliance.”
            
            **2. Emotionally Neutral - Identity**: 
            You are the House, not the Gambler. Professionals don’t cheer when they win, & they don't cry when they lose. They know the math plays out over 1,000 trades. You are biologically wired to feel pain when you lose money. Neutrality isn't the absence of emotion; it's the refusal to act on it. If your heart races when you enter a trade, - position size is too big. Before you click enter, visualize the trade hitting your stop loss immediately. Accept that loss mentally. If you can't accept the loss beforehand, you can’t take the trade.
            
            **3. Selective - Identity**: 
            You are a sniper with 2 bullets. You are not a machine gunner spraying. You reject "good" trades to wait for "great" trades. You are looking for reasons not to trade. Overtrading is the #1 account killer. It stems from the fear that "this is the last opportunity ever." Give yourself a hard cap on trades per day - "I have 2 bullets today". This forces you to treat every trade like a precious resource.
            
            **4. Focused on Survival First - Identity**: 
            You are a risk manager, not a profit generator. Your job is to protect your capital. Profit is just a byproduct of good survival skills. Amateur traders ask, "How much can I make?" Professional traders ask, "How much can I lose?" Defense-First Thinking. You survive to trade another day.
            
            ---
            
            #### ⏳ Timeframes Protocol
            
            *   **Primary Analysis (Swing) — 1D** → You think in daily structure, trends, levels, and bias.
            *   **Execution Timeframe — 4H** → You execute when conditions align with the daily trend. *(No forcing. No guessing. No noise.)*
            *   **Optional Day Trades — 15m** → Only when your higher timeframe bias + system criteria + mental state align. *(Not a default mode — a rare, intentional opportunity.)*
            """, unsafe_allow_html=True)
            

        
        elif st.session_state.active_tab == 'PORTAL':
            
            # --- CHART VIEW (Standard) ---
            # TradingView Widget
            # Use Active Symbol or Fallback
            tv_sym = st.session_state.get('active_tv_symbol', 'COINBASE:BTCUSD')
            tv_int = st.session_state.get('active_tv_interval', '60')
            
            # Helper to generate TV URL
            clean_sym = tv_sym.replace("BINANCE:", "").replace("COINBASE:", "").replace("OANDA:", "")
            tv_url = f"https://www.tradingview.com/chart?symbol={tv_sym}"
            
            # st.caption(f"**active:** {tv_sym} ({tv_int}m) | [Open in TradingView ↗]({tv_url})")
            
            tv_widget_code = f"""
            <div class="tradingview-widget-container" style="height:100%;width:100%">
              <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget(
              {{
              "width": "100%",
              "height": "485", 
              "autosize": false,
              "symbol": "{tv_sym}",
              "interval": "{tv_int}",
              "timezone": "America/New_York",
              "theme": "dark",
              "style": "1",
              "locale": "en",
              "toolbar_bg": "#f1f3f6",
              "enable_publishing": false,
              "allow_symbol_change": true,
              "container_id": "tradingview_chart",
              "hide_side_toolbar": false,
              "details": false,
              "hotlist": false,
              "calendar": true,
              "studies": [
                "MASimple@tv-basicstudies"
              ]
            }}
              );
              </script>
            </div>
            """
            
            components.html(tv_widget_code, height=500, scrolling=False)
            
            # st.markdown("<br>", unsafe_allow_html=True) # Spacer removed for tighter alignment
            
            # Invoke Button Logic
            
            @st.dialog("🔮 Cast Spell")
            def cast_spell_dialog():
                st.markdown("### Invocation Ritual")
                
                # Resources
                mana = st.session_state.mana
                s_day = st.session_state.spells_day
                s_week = st.session_state.spells_week
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Mana Pool", mana)
                c2.metric("Spells (Day)", s_day)
                c3.metric("Spells (Week)", s_week)
                
                st.markdown("---")
                
                # Inputs
                risk = st.number_input("Trade Risk (Mana Cost)", min_value=1, max_value=425, value=50, step=10)
                
                st.write("Confirmations:")
                check1 = st.checkbox("Two timeframe alignment?")
                check2 = st.checkbox("Within Bid Zone?")
                check3 = st.checkbox("Risk/Reward > 2.0?")
                check4 = st.checkbox("Mental State Neutral?")
                
                st.markdown("---")
                
                # Validation
                can_cast = True
                error_msg = ""
                
                if risk > mana:
                    can_cast = False
                    error_msg = "❌ Insufficient Mana!"
                elif s_day <= 0:
                    can_cast = False
                    error_msg = "❌ Daily Limit Reached!"
                elif s_week <= 0:
                    can_cast = False
                    error_msg = "❌ Weekly Limit Reached!"
                elif not (check1 and check2 and check3 and check4):
                    can_cast = False
                    error_msg = "❌ Complete Ritual Checklist"
                
                if not can_cast and error_msg:
                    st.error(error_msg)
                
                # Magical Button CSS
                st.markdown("""
                <style>
                div[data-testid="stDialog"] button[kind="primary"] {
                    background: linear-gradient(90deg, #6a11cb 0%, #2575fc 50%, #6a11cb 100%);
                    background-size: 200% auto;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    box-shadow: 0 0 15px rgba(100, 100, 255, 0.5);
                    animation: glowing 3s linear infinite;
                    transition: all 0.3s ease-in-out;
                }
                div[data-testid="stDialog"] button[kind="primary"]:hover {
                    background-position: right center;
                    transform: scale(1.02);
                    box-shadow: 0 0 25px rgba(100, 200, 255, 0.8);
                }
                @keyframes glowing {
                    0% { background-position: 0 0; }
                    50% { background-position: 100% 0; }
                    100% { background-position: 0 0; }
                }
                </style>
                """, unsafe_allow_html=True)
                
                if st.button("CAST SPELL ⚡", type="primary", use_container_width=True, disabled=not can_cast):
                    # Deduced Resources
                    st.session_state.mana -= risk
                    st.session_state.spells_day -= 1
                    st.session_state.spells_week -= 1
                    
                    # Save State
                    new_state = {
                        'mana': st.session_state.mana,
                        'spells_day': st.session_state.spells_day,
                        'spells_week': st.session_state.spells_week
                    }
                    save_grimoire(new_state)
                    
                    # Spectacular Animation - Custom CSS Overlay
                    # potential fix for "showing from right":
                    # Since st.dialog likely uses transforms, 'fixed' becomes relative to the dialog.
                    # We center it on the dialog (which is centered on screen) using 50%/50% + translate.
                    st.markdown("""
                    <div id="spell-overlay" style="
                        position: fixed;
                        top: 50%; left: 50%;
                        transform: translate(-50%, -50%);
                        width: 150vw; height: 150vh;
                        background: rgba(0,0,0,0.1); 
                        z-index: 999999;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        pointer-events: none;
                        backdrop-filter: blur(2px);
                    ">
                        <div style="
                            width: 10px; height: 10px;
                            border-radius: 50%;
                            background: radial-gradient(circle, #ffffff 0%, #00ffff 20%, #00ff88 40%, transparent 70%);
                            box-shadow: 0 0 50px #00ffff, 0 0 100px #00ff88;
                            animation: shockwave 1.2s cubic-bezier(0, 0, 0.2, 1) forwards;
                        "></div>
                    </div>
                    <style>
                        @keyframes shockwave {
                            0% { transform: scale(0.1); opacity: 1; }
                            40% { opacity: 0.9; }
                            100% { transform: scale(150); opacity: 0; }
                        }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(1.3) # Allow animation to play
                    st.rerun()

            _, c_inv, _ = st.columns([0.35, 0.3, 0.35])
            with c_inv:
                # Check Global Disable Conditions
                global_can_cast = (st.session_state.mana > 0) and (st.session_state.spells_day > 0) and (st.session_state.spells_week > 0)
                
                if st.button("\nINVOKE\n", use_container_width=True, type="primary", disabled=not global_can_cast):
                    cast_spell_dialog()


# --- RIGHT COLUMN: STATS, ORACLE, WIZARD ---
with col_right:
    # 1. Fractal Alignment (Replacing Realms)
    with st.container(border=True):
        st.markdown('<div class="runic-header">FRACTAL ALIGNMENT</div>', unsafe_allow_html=True)
        
        # Guide Header for Timeframes
        st.markdown("""
            <div style="display: flex; justify-content: space-between; padding: 0 10px; margin-bottom: 5px; border-bottom: 1px solid #c5a05930; padding-bottom: 2px;">
                <span style="font-size: 0.65rem; color: #888; font-weight: bold;">ASSET</span>
                <div style="display: flex; gap: 12px;">
                    <span style="font-size: 0.65rem; color: #888; width: 30px; text-align: center; font-weight: bold;">1D</span>
                    <span style="font-size: 0.65rem; color: #888; width: 30px; text-align: center; font-weight: bold;">4H</span>
                    <span style="font-size: 0.65rem; color: #888; width: 30px; text-align: center; font-weight: bold;">1H</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if os.path.exists(os.path.join("data", "mango_dynamic_data.json")):
            try:
                with open(os.path.join("data", "mango_dynamic_data.json"), "r") as f:
                    mango_data = json.load(f)
                
                if mango_data:
                    # Sort alphabetical
                    sorted_assets = sorted(mango_data.items())
                    
                    for asset, tfs in sorted_assets:
                        # Explicitly filter out Forex pairs if they linger in data
                        if asset in ["EURUSD", "GBPUSD", "AUDUSD"]:
                            continue

                        lights = ""
                        lights = ""
                        
                        # Bid Zone Check (Focus on 4H as requested)
                        bid_value = tfs.get("4h", {}).get("Bid Zone", "No") # Default to No if missing
                        
                        # Style based on Yes/No?
                        # User image shows roughly same white text. Let's keep it clean.
                        # We separate Asset and Bid Zone with some space.
                        bid_display = f'<span style="font-size: 0.75rem; color: #ccc; margin-left: 15px;">Bid Zone: {bid_value}</span>'

                        # Reordered: 1d -> 4h -> 1h
                        for tf in ["1d", "4h", "1h"]:
                            d = tfs.get(tf, {})
                            trend = d.get("Trend", "Unknown")
                            
                            # Accessibility Optimization: Color + Unique Symbol
                            color = "#444" 
                            symbol = "●" # Default
                            
                            if "Bullish" in trend: 
                                color = "#00ff88"
                                symbol = "▲"
                            elif "Bearish" in trend: 
                                color = "#ff3344"
                                symbol = "▼"
                            elif "Neutral" in trend: 
                                color = "#ffd700"
                                symbol = "◆"
                            
                            lights += f'<div title="{tf.upper()}: {trend}" style="color: {color}; text-shadow: 0 0 8px {color}60; width: 30px; text-align: center; font-size: 1.1rem; font-weight: bold;">{symbol}</div>'
                        
                        st.markdown(f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; background: rgba(255,255,255,0.02); border-radius: 4px; margin-bottom: 3px; border-left: 3px solid #c5a05940;"><div style="display: flex; align-items: center;"><span style="font-size: 0.85rem; font-weight: bold; color: #eee; width: 60px;">{asset}</span>{bid_display}</div><div style="display: flex; gap: 12px;">{lights}</div></div>', unsafe_allow_html=True)

                    # Legend for Color Blindness & Clarity
                    st.markdown("""
                        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 12px; padding: 6px; background: rgba(0,0,0,0.3); border-radius: 4px; border: 1px solid #c5a05920;">
                            <span style="font-size: 0.7rem; color: #00ff88; font-weight: bold;">▲ BULL</span>
                            <span style="font-size: 0.7rem; color: #ff3344; font-weight: bold;">▼ BEAR</span>
                            <span style="font-size: 0.7rem; color: #ffd700; font-weight: bold;">◆ NEUT</span>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No alignment data found.")
            except Exception as e:
                st.error(f"Error loading alignment: {e}")
        else:
            st.info("Scraper not yet run. Use 'Invoke Scraper' in sidebar.")


    

# --- LEFT COLUMN: MANA, SPELLS, ALERTS ---
with col_left:
    # 1. Runic Trade Alerts (Mana/Spells moved to Center)
    
    # 3. Runic Trade Alerts
    show_runic_alerts()



# --- Auto Refresh Logic ---


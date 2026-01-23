import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import concurrent.futures
import joblib
import sys
# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.core.data_fetcher import fetch_data
from src.strategies.strategy import WizardWaveStrategy
from src.strategies.strategy_scalp import WizardScalpStrategy
from src.utils.paths import get_data_path, get_model_path
import streamlit.components.v1 as components
import json
import urllib.request
import os
from datetime import datetime, date

# --- Persistence Logic ---
STATE_FILE = get_data_path("user_grimoire.json")

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
    initial_sidebar_state="collapsed"
)

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

# Strategy Settings (Defaults for HTF)
htf_lookback = 29
htf_sensitivity = 1.06
htf_cloud_spread = 0.64
htf_zone_pad = 1.5

# LTF Settings (Scalping)
ltf_lookback = 8
ltf_sensitivity = 1.0

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
    {"symbol": "EURUSD=X", "type": "trad", "name": "EUR/USD"},
    {"symbol": "GBPUSD=X", "type": "trad", "name": "GBP/USD"},
    {"symbol": "AUDUSD=X", "type": "trad", "name": "AUD/USD"},
    {"symbol": "SI=F", "type": "trad", "name": "Silver Futures"},
    {"symbol": "ARB/USDT", "type": "crypto", "name": "Arbitrum"},
    {"symbol": "AVAX/USDT", "type": "crypto", "name": "Avalanche"},
    {"symbol": "ADA/USDT", "type": "crypto", "name": "Cardano"},
]

# Initialize Session State
if 'processed_signals' not in st.session_state:
    st.session_state['processed_signals'] = set()

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
@st.cache_resource
def load_ml_models():
    """Load both HTF and LTF models"""
    models = {'htf': None, 'ltf': None}
    try:
        models['htf'] = joblib.load(get_model_path('model_htf.pkl'))
    except Exception as e:
        print(f"Error loading HTF model: {e}")
        
    try:
        models['ltf'] = joblib.load(get_model_path('model_ltf.pkl'))
    except Exception as e:
        print(f"Error loading LTF model: {e}")
        
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
    # "15 Minutes", "1 Hour", "4 Hours", "1 Day", "4 Days"
    if "15" in tf_label: return "15"
    if "1 Hour" in tf_label: return "60"
    if "4 Hour" in tf_label: return "240"
    if "1 Day" in tf_label: return "D"
    if "4 Day" in tf_label: return "240" # Fallback
    return "60"

# Initialize Active Symbol/Interval State
if 'active_tv_symbol' not in st.session_state:
    st.session_state.active_tv_symbol = "COINBASE:BTCUSD"
if 'active_tv_interval' not in st.session_state:
    st.session_state.active_tv_interval = "60"

# Initialize Active Symbol State
if 'active_tv_symbol' not in st.session_state:
    st.session_state.active_tv_symbol = "COINBASE:BTCUSD"

def calculate_ml_features(df):
    """
    Calculates features for ML (Must match pipeline.py logic)
    Features: volatility, rsi, ma_dist, adx, mom
    """
    df = df.copy()
    try:
        # 1. Volatility (20 period std dev of returns)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # 2. RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # 3. MA Distance (Price / SMA50 - 1)
        df['sma50'] = ta.sma(df['close'], length=50)
        df['ma_dist'] = (df['close'] / df['sma50']) - 1
        
        # 4. ADX
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if not adx_df.empty and 'ADX_14' in adx_df.columns:
            df['adx'] = adx_df['ADX_14']
        else:
            df['adx'] = 0

        # 5. Momentum (ROC)
        df['mom'] = ta.roc(df['close'], length=10)
        
        # Fill NaNs with 0 to prevent crashes
        df.fillna(0, inplace=True)
        return df
    except Exception as e:
        print(f"Feature Calc Error: {e}")
        return df

models = load_ml_models()

def analyze_timeframe(timeframe_label):
    results = []
    active_trades = []
    historical_signals = [] # Store all historical signals found
    
    tf_map = {
        "15 Minutes": "15m",
        "1 Hour": "1h",
        "4 Hours": "4h",
        "1 Day": "1d", 
        "4 Days": "4d"
    }
    tf_code = tf_map.get(timeframe_label, "1h")
    
    # Determine HTF or LTF
    is_ltf = tf_code in ['15m', '1h', '4h']
    
    # Initialize Strategy & Select Model
    if is_ltf:
        # Scalping
        strat = WizardScalpStrategy(lookback=ltf_lookback, sensitivity=ltf_sensitivity)
        model = models['ltf']
        tp_crypto = 0.02
        sl_crypto = 0.015
        tp_trad = 0.005
        sl_trad = 0.005
    else:
        # Trend
        strat = WizardWaveStrategy(
            lookback=htf_lookback,
            sensitivity=htf_sensitivity,
            cloud_spread=htf_cloud_spread,
            zone_pad_pct=htf_zone_pad
        )
        model = models['htf']
        tp_crypto = 0.08
        sl_crypto = 0.05
        tp_trad = 0.03
        sl_trad = 0.04

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"[{timeframe_label}] Fetching data for {len(ASSETS)} assets...")

    def process_asset(asset):
        try:
            # Dynamic Limit for performance
            # Increase limits to ensure ample history is captured
            current_limit = 1000  # Default to 1000 bars for all timeframes
            
            # Fetch Data
            df = fetch_data(asset['symbol'], asset['type'], timeframe=tf_code, limit=current_limit)
            
            if df.empty:
                return None, None, None
            
            # Apply Strategy
            df_strat = strat.apply(df)
            
            # ML Features & Prediction
            df_strat = calculate_ml_features(df_strat)
            
            if model:
                features = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom']
                # Predict for CURRENT candle
                last_features = df_strat.iloc[[-1]][features]
                prob = model.predict_proba(last_features)[0][1] # Prob of Class 1 (Good)
                
                # Predict for ALL rows (for history)
                # Ensure no NaNs
                df_clean = df_strat.dropna()
                if not df_clean.empty:
                    all_probs = model.predict_proba(df_clean[features])[:, 1]
                    # Map back to original index
                    df_strat.loc[df_clean.index, 'model_prob'] = all_probs
                else:
                    df_strat['model_prob'] = 0.0
                
                # Explicitly set probability for the last row (calculated separately above)
                # This ensures the active forming candle is included in history simulation
                if not df_strat.empty:
                     df_strat.loc[df_strat.index[-1], 'model_prob'] = prob
            else:
                prob = 0.0
                df_strat['model_prob'] = 0.0

            # --- Check Active Trade ---
            active_trade_data = None
            trade = strat.get_active_trade(df_strat)
            
            if trade:
                ts = trade['Entry Time']
                ts_str = format_time(ts)
                sort_ts = ts if not pd.isna(ts) else pd.Timestamp.min.tz_localize('UTC')

                # Get confidence at ENTRY time
                try:
                    # Find probability at entry time
                    entry_idx = df_strat.index.get_loc(ts)
                    entry_conf = df_strat.iloc[entry_idx]['model_prob']
                except:
                    entry_conf = prob # Fallback to current if entry index fail
                
                # Check Labels
                is_trad = asset['type'] == 'trad'
                threshold = 0.40 # Optimized Threshold
                rec_action = "✅ TAKE" if entry_conf > threshold else "❌ SKIP"
                
                type_display = f"⬆️ {trade['Position']}" if trade['Position'] == 'LONG' else f"⬇️ {trade['Position']}"
                
                # Determine Decimal Precision
                decimals = 2
                s_lower = asset['symbol'].lower()
                n_lower = asset['name'].lower()
                
                # High Precision Assets
                high_prec_keywords = ['doge', 'ada', 'xrp', 'link', 'arb', 'algo', 'matic', 'ftm']
                if any(k in s_lower or k in n_lower for k in high_prec_keywords):
                    decimals = 4
                
                # Forex 
                if '=x' in s_lower:
                    if 'jpy' in s_lower:
                        decimals = 2
                    else:
                        decimals = 5
                        
                # Calculate TP/SL Prices
                # Use LTF/HTF specific Values
                curr_tp = tp_trad if is_trad else tp_crypto
                curr_sl = sl_trad if is_trad else sl_crypto
                
                ep = trade['Entry Price']
                
                if trade['Position'] == 'LONG':
                    tp_price = ep * (1 + curr_tp)
                    sl_price = ep * (1 - curr_sl)
                else:
                    tp_price = ep * (1 - curr_tp)
                    sl_price = ep * (1 + curr_sl)

                active_trade_data = {
                    "_sort_key": sort_ts,
                    "Asset": asset['name'],
                    "Symbol": asset['symbol'], # Store raw symbol
                    "Type": type_display,
                    "Timeframe": timeframe_label,
                    "Entry_Time": ts_str,
                    "Signal_Time": ts_str, # Redundant but safe
                    "Entry_Price": f"{ep:.{decimals}f}",
                    "Take_Profit": f"{tp_price:.{decimals}f}",
                    "Stop_Loss": f"{sl_price:.{decimals}f}",
                    "Current_Price": f"{df_strat.iloc[-1]['close']:.{decimals}f}",
                    "PnL (%)": f"{trade['PnL (%)']:.2f}%",
                    "Confidence": f"{entry_conf:.0%}",
                    "Action": rec_action,
                    "Signal": trade['Position'] # Map Position to Signal column for frontend
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
            def simulate_history_stateful(df, asset_type):
               trades = []
               position = None
               entry_price = 0.0
               entry_time = None
               
               curr_tp_pct = tp_trad if asset_type == 'trad' else tp_crypto
               curr_sl_pct = sl_trad if asset_type == 'trad' else sl_crypto
               
               # Iterate through all bars
               for idx, row in df.iterrows():
                   close = row['close']
                   high = row['high']
                   low = row['low']
                   signal = row['signal_type']
                   model_prob = row.get('model_prob', 0.0)
                   
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
                            "Timeframe": timeframe_label,
                            "Time": format_time(entry_time),
                            "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                            "Price": entry_price,
                            "Confidence": "N/A",
                            "Model": "✅",
                            "Return_Pct": pnl, 
                            "SL_Pct": curr_sl_pct,
                            "Status": status
                       })
                       position = None

                   # --- ENTRY & REVERSAL LOGIC ---
                   # Only take filtered signals
                   if model_prob > 0.40:
                       new_pos = None
                       # Check Strategy Output strings
                       if 'LONG' in signal:
                           new_pos = 'LONG'
                       elif 'SHORT' in signal:
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
                                   
                               trades.append({
                                    "_sort_key": entry_time,
                                    "Asset": asset['name'],
                                    "Timeframe": timeframe_label,
                                    "Time": format_time(entry_time),
                                    "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                                    "Price": entry_price,
                                    "Confidence": "N/A",
                                    "Model": "✅",
                                    "Return_Pct": last_close_pnl, 
                                    "SL_Pct": curr_sl_pct,
                                    "Status": "FLIP 🔄"
                               })
                               # Prepare for new entry
                               position = None 
                           
                           # Enter new position (if not already in one or just flipped)
                           if position is None:
                               position = new_pos
                               entry_price = close
                               entry_time = idx
                               
               # --- END OF LOOP ---
               # If position is still open, calculate floating PnL
               if position is not None:
                    last_price = df.iloc[-1]['close']
                    if position == 'LONG':
                        pnl = (last_price - entry_price) / entry_price
                    else:
                        pnl = (entry_price - last_price) / entry_price
                        
                    trades.append({
                        "_sort_key": entry_time,
                        "Asset": asset['name'],
                        "Timeframe": timeframe_label,
                        "Time": format_time(entry_time),
                        "Type": f"{position} {'🟢' if position == 'LONG' else '🔴'}",
                        "Price": entry_price,
                        "Confidence": "N/A",
                        "Model": "✅",
                        "Return_Pct": pnl, 
                        "SL_Pct": curr_sl_pct,
                        "Status": "OPEN"
                    })

               return trades

            # Run simulation on FULL fetched history (not just tail)
            asset_history = simulate_history_stateful(df_strat, asset['type'])
            
            return result_data, active_trade_data, asset_history
            
        except Exception as e:
            print(f"DEBUG: Error processing {asset['symbol']}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    # Parallel Execution
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_asset = {executor.submit(process_asset, asset): asset for asset in ASSETS}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_asset)):
            progress_bar.progress((i + 1) / len(ASSETS))
            try:
                res, trade, hist = future.result()
                if res: results.append(res)
                if trade: active_trades.append(trade)
                if hist: historical_signals.extend(hist)
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    
    # Sort
    if active_trades:
        active_trades.sort(key=lambda x: x['_sort_key'], reverse=True)
        
    return pd.DataFrame(results), pd.DataFrame(active_trades), historical_signals

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
                    "size": 50000, 
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
                
                # Progress to Target
                profit_gained = cur_bal - size
                pct_to_target = max(0.0, min(1.0, profit_gained / pt_amt)) if pt_amt > 0 else 0
                
                # Drawdown limit (Trailing or Static? Assuming Static for simplicity/safety unless specified)
                # Usually DD is relative to High Water Mark or Static.
                # User didn't specify, so we visualize distance to static Min Balance.
                
                dist_to_dd = cur_bal - min_bal
                dd_pct = max(0.0, min(1.0, dist_to_dd / dd_limit_amt))
                
                # Visuals
                col_a, col_b = st.columns(2)
                col_a.metric("Target", f"${target_bal:,.0f}", delta=f"{pct_to_target:.1%}")
                col_b.metric("Breach Level", f"${min_bal:,.0f}", delta=f"${dist_to_dd:,.0f} Room", delta_color="normal")
                
                st.progress(pct_to_target, text="Progress to Payout")
                st.progress(dd_pct, text="Drawdown Room")

    if accounts_changed:
        # Save to Grimoire
        grimoire = load_grimoire()
        grimoire['prop_accounts'] = st.session_state.user_accounts
        save_grimoire(grimoire)

# --- Main Layout ---
with st.sidebar:
    st.header("🔮 Arcane Settings")
    
    # Mana System
    st.markdown(f"**Mana:** {st.session_state['mana']} / 500 🔵")
    st.progress(st.session_state['mana'] / 500)
    
    st.markdown("---")
    st.markdown("**Spells Available:**")
    st.markdown(f"Daily: {st.session_state['spells_day']} / 2 📜")
    st.markdown(f"Weekly: {st.session_state['spells_week']} / 5 📚")
    
    st.markdown("---")
    
    # Timeframe Selector - Reordered
    timeframe = st.selectbox("Select Timeframe", 
                             ["15 Minutes", "1 Hour", "4 Hours", "1 Day", "4 Days"],
                             index=4) # Default to 1D (HTF)
    
    active_tv_interval = get_tv_interval(timeframe)
    if active_tv_interval != st.session_state.active_tv_interval:
        st.session_state.active_tv_interval = active_tv_interval
        # st.rerun() # Re-render to update widget

    if st.button("Refresh Signals"):
        st.cache_data.clear()
        st.rerun()

# --- Tabs ---
# Tab Order: Portal (Dashboard) -> Runic Alerts -> Prop Risk -> Oracle
tab1, tab2, tab3, tab4 = st.tabs(["Arcane Portal", "Runic Alerts", "Prop Risk", "Oracle"])

with tab1:
    # TradingView Widget
    # TradingView Widget
    symbol = st.session_state.active_tv_symbol
    interval = st.session_state.active_tv_interval
    
    # Full Screen Widget Code
    tv_widget = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_12345"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {{
        "width": "100%",
        "height": 600,
        "symbol": "{symbol}",
        "interval": "{interval}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tradingview_12345"
      }}
      );
      </script>
    </div>
    """
    components.html(tv_widget, height=610)

with tab2:
    st.subheader("Runic Trade Alerts")
    
    active_signals, active_trades_df, historical_signals = analyze_timeframe(timeframe)
    
    # Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Active Assets", len(ASSETS))
    
    # Filter for 'TAKE' signals
    take_signals = active_trades_df[active_trades_df['Action'] == "✅ TAKE"] if not active_trades_df.empty else pd.DataFrame()
    c2.metric("Signals Found", len(active_trades_df) if not active_trades_df.empty else 0)
    c3.metric("Actionable", len(take_signals))
    
    st.markdown("### ⚡ Active Signals")
    if not active_trades_df.empty:
        # Reorder columns
        cols = ["Asset", "Type", "Entry_Price", "Take_Profit", "Stop_Loss", "PnL (%)", "Confidence", "Action"]
        
        # Apply Styling
        st.dataframe(active_trades_df[cols].style.apply(highlight_confidence, axis=1), use_container_width=True)
        
        # Update Chart if User clicks a row (Streamlit default interaction is limited, so we add buttons)
        st.markdown("#### Inspect Asset")
        cols = st.columns(5)
        for idx, row in active_trades_df.iterrows():
            if cols[idx % 5].button(f"{row['Asset']}", key=f"btn_{idx}"):
                # Find asset entry
                asset_entry = next((a for a in ASSETS if a['name'] == row['Asset']), None)
                if asset_entry:
                    st.session_state.active_tv_symbol = get_tv_symbol(asset_entry)
                    st.rerun()

    else:
        st.info("No active signals for this timeframe.")
        
    st.markdown("---")
    st.markdown("### 📜 Signal History & Verification")
    
    if historical_signals:
        hist_df = pd.DataFrame(historical_signals)
        # Show last 50
        hist_df.sort_values(by="_sort_key", ascending=False, inplace=True)
        
        # Calculate Stats
        total_trades = len(hist_df)
        wins = len(hist_df[hist_df['Return_Pct'] > 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        m1, m2 = st.columns(2)
        m1.metric("Historical Trades Captured", total_trades)
        m2.metric("Win Rate (Raw)", f"{win_rate:.1f}%")

        # Display History Table
        disp_cols = ["Time", "Asset", "Type", "Price", "Return_Pct", "Status"]
        
        # Format Return Pct
        hist_df['Return_Pct'] = hist_df['Return_Pct'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(hist_df[disp_cols], use_container_width=True)
    else:
        st.write("No history available.")

with tab3:
    st.subheader("Prop Firm Risk Manager")
    render_prop_risk()

with tab4:
    st.subheader("The Oracle")
    st.write("Coming Soon...")

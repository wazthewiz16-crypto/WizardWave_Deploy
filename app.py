import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import concurrent.futures
import joblib
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy
import streamlit.components.v1 as components
import json
import urllib.request
import os
from datetime import datetime, date


# --- Persistence Logic ---
STATE_FILE = "user_grimoire.json"

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
def load_ml_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

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
    if "15" in tf_label: return "15"
    if "30" in tf_label: return "30"
    if "1 Hour" in tf_label: return "60"
    if "4 Hour" in tf_label: return "240"
    if "1 Day" in tf_label: return "D"
    if "4 Day" in tf_label: return "240" # Fallback to 4H as 4D isn't standard in basic widget, or use 'D'
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

model = load_ml_model()

def analyze_timeframe(timeframe_label):
    results = []
    active_trades = []
    historical_signals = [] # Store all historical signals found
    
    tf_map = {
        "15 Minutes": "15m",
        "30 Minutes": "30m",
        "1 Hour": "1h",
        "4 Hours": "4h",
        "1 Day": "1d", 
        "4 Days": "4d"
    }
    tf_code = tf_map.get(timeframe_label, "1h")
    
    # Initialize Strategy
    strat = WizardWaveStrategy(
        lookback=lookback,
        sensitivity=sensitivity,
        cloud_spread=cloud_spread,
        zone_pad_pct=zone_pad
    )

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
                pt_pct = 0.03 if is_trad else 0.08
                sl_pct = 0.04 if is_trad else 0.05
                ep = trade['Entry Price']
                
                if trade['Position'] == 'LONG':
                    tp_price = ep * (1 + pt_pct)
                    sl_price = ep * (1 - sl_pct)
                else:
                    tp_price = ep * (1 - pt_pct)
                    sl_price = ep * (1 + sl_pct)

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
            # --- Collect Historical Signals & Simulate PnL ---
            
            # Helper to run stateful simulation on the history
            def simulate_history_stateful(df, asset_type):
               trades = []
               position = None
               entry_price = 0.0
               entry_time = None
               sl_price = 0.0
               
               sl_pct = 0.05 if asset_type == 'crypto' else 0.04
               tp_pct = 0.08 if asset_type == 'crypto' else 0.03
               
               # Iterate through all bars
               # Assumes df is sorted by time
               # print(f"DEBUGGING HIST: {asset['name']} - {len(df)} rows. Last: {df.index[-1]}")
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
                       if high >= entry_price * (1 + tp_pct):
                           pnl = tp_pct
                           status = "HIT TP 🟢"
                           exit_trade = True
                       # Check SL
                       elif low <= entry_price * (1 - sl_pct):
                           pnl = -sl_pct
                           status = "HIT SL 🔴"
                           exit_trade = True

                   elif position == 'SHORT':
                       # Check TP
                       if low <= entry_price * (1 - tp_pct):
                           pnl = tp_pct
                           status = "HIT TP 🟢"
                           exit_trade = True
                       # Check SL
                       elif high >= entry_price * (1 + sl_pct):
                           pnl = -sl_pct
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
                            "SL_Pct": sl_pct,
                            "Status": status
                       })
                       position = None

                   # --- ENTRY & REVERSAL LOGIC ---
                   # Only take filtered signals
                   if model_prob > 0.40:
                       new_pos = None
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
                                    "SL_Pct": sl_pct,
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
                        "SL_Pct": sl_pct,
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
        size = account['size']
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
    """Checks for new signals and sends Discord alerts."""
    try:
        # Load Config
        if not os.path.exists('discord_config.json'):
            return
            
        with open('discord_config.json', 'r') as f:
            config = json.load(f)
            webhook_url = config.get('webhook_url')
            
        if not webhook_url:
            return

        # Load Processed IDs
        processed_file = 'processed_signals.json'
        processed_ids = set()
        
        if os.path.exists(processed_file):
            try:
                with open(processed_file, 'r') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        processed_ids = set(content)
            except:
                processed_ids = set() # Reset if corrupt

        new_alerts_sent = False
        
        # Iterate and Check
        for _, row in df.iterrows():
            try:
                # Create Unique ID (Asset + Timeframe + EntryTime)
                # Use Entry_Time as primary differentiator
                entry_time = str(row.get('Entry_Time', ''))
                asset = str(row.get('Asset', ''))
                tf = str(row.get('Timeframe', ''))
                
                sig_id = f"{asset}_{tf}_{entry_time}"
                
                # Check if it is a TAKE signal
                action_str = str(row.get('Action', '')).upper()
                is_take = "TAKE" in action_str or "✅" in action_str
                
                if is_take and sig_id not in processed_ids and sig_id != "__":
                    send_discord_alert(webhook_url, row)
                    processed_ids.add(sig_id)
                    new_alerts_sent = True
            except Exception as inner_e:
                print(f"Skipping alert row: {inner_e}")
                continue
        
        # Save updated IDs
        if new_alerts_sent:
            try:
                with open(processed_file, 'w') as f:
                    json.dump(list(processed_ids), f)
            except Exception as e:
                print(f"Error saving processed IDs: {e}")
                
    except Exception as e:
        print(f"Error processing Discord alerts: {e}")

@st.fragment(run_every=60)
def show_runic_alerts():
    # Header Row with Refresh Button
    with st.container(border=True):
        # Adjusted columns: Title | History Btn | Metric | Refresh
        c_title, c_hist, c_metric, c_btn = st.columns([0.35, 0.15, 0.35, 0.15], gap="small")
        
        with c_title:
             st.markdown('<div class="runic-header" style="font-size: 1rem; border: none !important; margin-bottom: 0; padding: 0; margin-top: -10px; background: transparent; text-align: left;">RUNIC ALERTS</div>', unsafe_allow_html=True)
        
        with c_hist:
            # History Button in Sidebar
            st.markdown('<div style="margin-top: -5px;"></div>', unsafe_allow_html=True)
            if st.button("📜", key="hist_side", help="Signal History", use_container_width=True):
                st.session_state.active_tab = 'HISTORY'
                st.rerun()

        # Display 24H Return (Will be populated after calculation or from state)
        # Use a placeholder for dynamic updates
        return_placeholder = c_metric.empty()
        
        def render_return_value(val_24, val_12):
            # 24H Logic
            r24_color = "#00ff88" if val_24 >= 0 else "#ff3344"
            r24_sign = "+" if val_24 >= 0 else ""
            
            # 12H Logic
            r12_color = "#00ff88" if val_12 >= 0 else "#ff3344"
            r12_sign = "+" if val_12 >= 0 else ""
            
            return_placeholder.markdown(f"""
                <div style="text-align: center; margin-top: -12px; white-space: nowrap; line-height: 1.1;">
                    <div style="margin-bottom: 2px;">
                        <span style="font-size: 0.9rem; color: #888; font-weight: bold;">24H: </span>
                        <span style="font-size: 0.9rem; font-weight: bold; color: {r24_color}; text-shadow: 0 0 10px {r24_color}40;">
                            {r24_sign}{val_24:.2%}
                        </span>
                    </div>
                    <div>
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
            # Shift button up slightly to align
            st.markdown('<div style="margin-top: -5px;"></div>', unsafe_allow_html=True)
            refresh_click = st.button("↻", key="refresh_top", help="Refresh", use_container_width=True)
            
        # --- Data Fetching Logic ---
        # Determine if we need to fetch data
        now = time.time()
        should_fetch = False
        
        if refresh_click:
            should_fetch = True
        elif 'last_runic_fetch' not in st.session_state:
            should_fetch = True
        elif 'runic_12h_return' not in st.session_state:
             should_fetch = True
        elif now - st.session_state.get('last_runic_fetch', 0) > 55:
            should_fetch = True
            
        # Initialize Data Container
        if 'combined_active_df' not in st.session_state:
             st.session_state.combined_active_df = pd.DataFrame()

        if should_fetch:

        
            # Run All Timeframes
            # Note: analyze_timeframe uses st.progress which will display here
            r15m, a15m, h15m = analyze_timeframe("15 Minutes")
            r30m, a30m, h30m = analyze_timeframe("30 Minutes")
            r1h, a1h, h1h = analyze_timeframe("1 Hour")
            r4h, a4h, h4h = analyze_timeframe("4 Hours")
            r1d, a1d, h1d = analyze_timeframe("1 Day")
            r4d, a4d, h4d = analyze_timeframe("4 Days")
        
            # Clear status
            
            # --- Aggregated History for Returns ---
            # Combine all history
            all_history = []
            if h15m: all_history.extend(h15m)
            if h30m: all_history.extend(h30m)
            if h1h: all_history.extend(h1h)
            if h4h: all_history.extend(h4h)
            if h1d: all_history.extend(h1d)
            if h4d: all_history.extend(h4d)
            
            total_24h_return = 0.0
            total_12h_return = 0.0
            
            if all_history:
                hist_df = pd.DataFrame(all_history)
                # Ensure sort key is datetime and UTC for consistent filtering
                if not hist_df.empty and '_sort_key' in hist_df.columns:
                    try:
                       # Standardize to UTC
                       hist_df['_sort_key'] = pd.to_datetime(hist_df['_sort_key'], utc=True)
                       now_utc = pd.Timestamp.now(tz='UTC')
                       
                       # Filter for last 24h and 12h
                       cutoff_24 = now_utc - pd.Timedelta(hours=24)
                       cutoff_12 = now_utc - pd.Timedelta(hours=12)
                       
                       recent_sigs_24 = hist_df[hist_df['_sort_key'] >= cutoff_24]
                       recent_sigs_12 = hist_df[hist_df['_sort_key'] >= cutoff_12]
                       
                       total_24h_return = recent_sigs_24['Return_Pct'].sum()
                       total_12h_return = recent_sigs_12['Return_Pct'].sum()
                    except Exception as e:
                        print(f"Error calcing returns: {e}")
                        total_24h_return = 0.0
                        total_12h_return = 0.0

            # Update state AND update visual immediately
            st.session_state['runic_24h_return'] = total_24h_return
            st.session_state['runic_12h_return'] = total_12h_return
            
            # ALSO store the full history for the Verification Tab
            if all_history:
                 st.session_state['runic_history_df'] = pd.DataFrame(all_history)
            
            render_return_value(total_24h_return, total_12h_return)

        
            # Consolidate
            active_dfs = [df for df in [a15m, a30m, a1h, a4h, a1d, a4d] if df is not None and not df.empty]
            
            if active_dfs:
                combined_active = pd.concat(active_dfs).sort_values(by='_sort_key', ascending=False)
            else:
                combined_active = pd.DataFrame()
            
            # --- PROCESS DISCORD ALERTS ---
            if not combined_active.empty:
                process_discord_alerts(combined_active)

            # Save to Session State
            st.session_state['combined_active_df'] = combined_active
            st.session_state['last_runic_fetch'] = now
            
        # Get Data for Display
        combined_active = st.session_state.get('combined_active_df', pd.DataFrame())
        
        # --- Timeframe Filter (Inside Box) ---
        if not combined_active.empty:
            # Get unique timeframes and sort chronologically
            tf_order = {
                "15 Minutes": 0, "30 Minutes": 1, 
                "1 Hour": 2, "4 Hours": 3, 
                "1 Day": 4, "4 Days": 5
            }
            
            # Mapping to Compact format
            tf_map = {
                "15 Minutes": "15m", "30 Minutes": "30m", 
                "1 Hour": "1H", "4 Hours": "4H", 
                "1 Day": "1D", "4 Days": "4D"
            }
            tf_map_rev = {v: k for k, v in tf_map.items()}

            unique_tfs = combined_active['Timeframe'].unique().tolist()
            # Sort first based on original order
            sorted_tfs = sorted(unique_tfs, key=lambda x: tf_order.get(x, 99))
            
            # Convert to display options
            display_opts = [tf_map.get(x, x) for x in sorted_tfs]
            
            # Compact Multiselect
            st.markdown("<div style='margin-top: -15px;'></div>", unsafe_allow_html=True)
            selected_short = st.multiselect("Timeframes", options=display_opts, default=display_opts, label_visibility="collapsed")
            
            # Map back for filtering
            selected_tfs = [tf_map_rev.get(x, x) for x in selected_short]
            
            # --- Filter Data ---
            df_display = combined_active.copy()
            
            # Use global show_take_only variable (default True)
            if show_take_only:
                 if 'Action' in df_display.columns:
                    df_display = df_display[df_display['Action'].str.contains("TAKE")]
            
            if selected_tfs:
                df_display = df_display[df_display['Timeframe'].isin(selected_tfs)]
            else:
                st.warning("Select Timeframe")
                df_display = pd.DataFrame(columns=df_display.columns) # Empty

            # --- Render ---
            if df_display.empty:
                st.info("No active signals.")
            else:
                # Pagination Logic (Reduced per page for standard height)
                ITEMS_PER_PAGE = 5 
                if 'page_number' not in st.session_state:
                    st.session_state.page_number = 0
                    
                total_pages = max(1, (len(df_display) - 1) // ITEMS_PER_PAGE + 1)
                
                # Ensure page number is valid
                if st.session_state.page_number >= total_pages:
                    st.session_state.page_number = total_pages - 1
                if st.session_state.page_number < 0:
                    st.session_state.page_number = 0
                    
                start_idx = st.session_state.page_number * ITEMS_PER_PAGE
                end_idx = start_idx + ITEMS_PER_PAGE
                
                current_batch = df_display.iloc[start_idx:end_idx]
                
                for index, row in current_batch.iterrows():
                    # --- Card Container for "Inside" Look ---
                    # We use a container with a border to group Content + Button
                    with st.container(border=True):
                        
                        # Layout: [Content (0.8) | Button (0.2)]
                        c_content, c_btn = st.columns([0.75, 0.25])
                        
                        # --- 1. Content Section ---
                        with c_content:
                            is_long = "LONG" in row.get('Type', '')
                            direction_color = "#00ff88" if is_long else "#ff3344"
                            
                            # Icon Logic
                            asset_name = row['Asset']
                            asset_symbol = row.get('Symbol', '')
                            if not asset_symbol:
                                 for a in ASSETS:
                                     if a['name'] == asset_name:
                                         asset_symbol = a['symbol']
                                         break
                            
                            icon_char = "⚡"
                            if "BTC" in asset_name: icon_char = "₿"
                            elif "ETH" in asset_name: icon_char = "Ξ"
                            elif "SOL" in asset_name: icon_char = "◎"
                            
                            action_text = "BULL" if is_long else "BEAR"
                            pnl_val = row.get('PnL (%)', '0.00%')
                            pnl_color = "#00ff88" if not str(pnl_val).startswith("-") else "#ff3344"
                            
                            # Compact HTML representation
                            st.markdown(f"""
                                <div style="display: flex; align-items: flex-start; margin-top: -10px;">
                                    <div style="color: {direction_color}; font-size: 1.2rem; margin-right: 8px; margin-top: -2px;">{icon_char}</div>
                                    <div style="flex-grow: 1;">
                                        <div style="font-weight: bold; font-size: 0.9rem; color: #e0e0e0; display: flex; justify-content: space-between;">
                                            <span>{asset_name} <span style="color:{direction_color}; font-size: 0.8rem;">{action_text}</span></span>
                                            <span style="color: {pnl_color};">{pnl_val}</span>
                                        </div>
                                        <div style="font-size: 0.75rem; color: #aaa; margin-top: 2px;">
                                            {row.get('Action')} | Conf: {row.get('Confidence')} | {row.get('Timeframe')}
                                        </div>
                                        <div style="font-size: 0.7rem; color: #888; margin-top: 1px;">
                                            TP: {row.get('Take_Profit', 'N/A')} | SL: {row.get('Stop_Loss', 'N/A')}
                                        </div>
                                        <div style="font-size: 0.7rem; color: #666;">
                                            Entry: <span style="color: #00ff88;">{row.get('Entry_Price')}</span> | Now: <span style="color: #ffd700;">{row.get('Current_Price', 'N/A')}</span>
                                        </div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        # --- 2. Button Section (Inside the Card) ---
                        with c_btn:
                            # Center the button vertically relative to content
                            st.markdown('<div style="height: 2px;"></div>', unsafe_allow_html=True)
                            
                            # Unique Key
                            unique_id = f"{row['Asset']}_{row.get('Timeframe','')}_{row.get('Entry_Time','')}"
                            unique_id = "".join(c for c in unique_id if c.isalnum() or c in ['_','-'])
                            
                            # Visual "View" Button
                            if st.button("👁️", key=f"btn_card_{unique_id}", use_container_width=True):
                                # 1. Resolve Trading View Symbol
                                tv_sym = get_tv_symbol({'symbol': asset_symbol, 'name': asset_name})
                                # 2. Resolve Interval
                                tv_int = get_tv_interval(row.get('Timeframe', '1 Hour'))
                                
                                # 3. Update Session State
                                st.session_state.active_tv_symbol = tv_sym
                                st.session_state.active_tv_interval = tv_int
                                
                                # 4. Trigger Main App Rerun
                                st.rerun()
                                
                            # Time under button
                            time_val = row.get('Entry_Time', row.get('Signal_Time', 'N/A'))
                            # Try to make it shorter? e.g. 2025-12-15 17:30 -> 12-15 17:30
                            try:
                                if len(str(time_val)) > 10:
                                    short_time = str(time_val)[5:-3] # remove YYYY- and :SS
                                else:
                                    short_time = str(time_val)
                            except:
                                short_time = str(time_val)
                                
                            st.markdown(f"<div style='text-align: center; font-size: 0.65rem; color: #00eaff; margin-top: -2px;'>{short_time}</div>", unsafe_allow_html=True)
                            
                
                # --- Compact Numbered Pagination ---
                # Layout: [First] [Prev] [Page Text] [Next]
                p_first, p_prev, p_text, p_next = st.columns([0.15, 0.15, 0.55, 0.15])
                
                with p_first:
                    if st.button("⏪", key="first_main", disabled=(st.session_state.page_number == 0), help="First Page"):
                        st.session_state.page_number = 0
                        st.rerun()
                        
                with p_prev:
                    if st.button("◀", key="prev_main", disabled=(st.session_state.page_number == 0)):
                        st.session_state.page_number -= 1
                        st.rerun()
                        
                with p_next:
                    if st.button("▶", key="next_main", disabled=(st.session_state.page_number >= total_pages - 1)):
                        st.session_state.page_number += 1
                        st.rerun()
                        
                with p_text:
                    st.markdown(f"<div style='text-align: center; color: #888; font-size: 0.8rem; padding-top: 5px;'>Page {st.session_state.page_number + 1}/{total_pages}</div>", unsafe_allow_html=True)

        else:
            st.info("No active signals.")

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

    /* Epic Invoke Button */
    div.stButton > button[kind="primary"] {
        width: 100%;
        height: 65px;
        background: linear-gradient(135deg, #c5a059 0%, #8a6e3c 100%);
        color: #000;
        font-family: 'Cinzel', serif;
        font-size: 2.0rem;
        font-weight: bold;
        border: 2px solid #ffd700;
        box-shadow: 0 0 20px rgba(197, 160, 89, 0.6);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 4px;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(197, 160, 89, 0.9);
        color: #fff;
    }


    .runic-header {

        text-align: center;
        color: #c5a059;
        font-size: 0.9rem;
        font-weight: bold;
        border-bottom: 1px solid #4a3b22;
        padding-bottom: 2px;
        margin-bottom: 4px;
        margin-top: -17px;
        text-shadow: 0 0 5px #c5a059;
    }

    .runic-item {
        display: flex;
        align-items: center;
        background: linear-gradient(90deg, rgba(20, 20, 30, 0.8) 0%, rgba(35, 35, 50, 0.8) 100%);
        border: 1px solid #4a4a60;
        border-left: 4px solid #444;
        margin-bottom: 2px;
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
        gap: 6px;
        justify-content: center;
        margin-bottom: 2px;
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
        margin-bottom: 5px;
        position: relative;
        padding: 5px 0;
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
        font-size: 0.7rem !important;
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
        padding: 8px !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
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
  "displayMode": "adaptive",
  "locale": "en"
}
  </script>
</div>
""", height=60, scrolling=False)

# Layout Columns
col_left, col_center, col_right = st.columns([0.25, 0.5, 0.25], gap="small")

# --- CENTER COLUMN: MAIN PORTAL ---
with col_center:
    with st.container(border=True):
        # Interactive Navbar
        if 'active_tab' not in st.session_state: st.session_state.active_tab = 'PORTAL'
        
        def set_tab(t):
            st.session_state.active_tab = t
            
        # Adjusted columns for remaining tabs (History button removed)
        c2, c3, c4, c5 = st.columns([1, 1, 1, 1])
        
        # History Tab Hidden (Access via Sidebar)
        # But we still respect the active_tab logic below
        
        c2.button("PORTAL", use_container_width=True, type="primary" if st.session_state.active_tab=='PORTAL' else "secondary", on_click=set_tab, args=('PORTAL',))
        
        c3.button("SHIELD", use_container_width=True, type="primary" if st.session_state.active_tab=='RISK' else "secondary", on_click=set_tab, args=('RISK',))
        
        c4.button("RULES", use_container_width=True, type="primary" if st.session_state.active_tab=='RULES' else "secondary", on_click=set_tab, args=('RULES',))
        
        c5.button("SPELLBOOK", use_container_width=True, type="primary" if st.session_state.active_tab=='SPELLBOOK' else "secondary", on_click=set_tab, args=('SPELLBOOK',))
        
        # Remove standard divider and use negative margin wrapper for tighter fit
        st.markdown('<div style="margin-top: -10px; margin-bottom: -10px;"><hr style="margin: 5px 0; border-color: #333;"></div>', unsafe_allow_html=True)
        
        if st.session_state.active_tab == 'HISTORY':
             st.markdown("### 📜 Signal History & Verification")
             
             hist_df = st.session_state.get('runic_history_df', pd.DataFrame())

             
             if not hist_df.empty and '_sort_key' in hist_df.columns:
                 # Standardize
                 try:
                    hist_df['_sort_key'] = pd.to_datetime(hist_df['_sort_key'], utc=True)
                 except: pass
                 
                 # Toggle for 24H Only
                 show_24h_only = st.checkbox("Show last 24 Hours Only", value=True)
                 
                 # 1. Filter Time
                 if show_24h_only:
                     cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)
                     filtered_df = hist_df[hist_df['_sort_key'] >= cutoff].copy()
                 else:
                     filtered_df = hist_df.copy()
                     
                 # Sort newest first
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
                 m1.metric("PnL Sum (Verify)", f"{total_ret:.2%}")
                 m2.metric("Trades", total_trades)
                 m3.metric("Win Rate", f"{win_rate:.0%}")
                 
                 st.divider()
                 
                 # Display Table
                 # Simplify cols
                 if not filtered_df.empty:
                     display_cols = ['Time', 'Asset', 'Timeframe', 'Type', 'Price', 'Return_Pct', 'Status']
                     # Fill Status if missing
                     if 'Status' not in filtered_df.columns:
                         filtered_df['Status'] = 'CLOSED'
                         
                     # Format Return
                     filtered_df['Return'] = filtered_df['Return_Pct'].apply(lambda x: f"{x:.2%}")
                     
                     st.dataframe(
                         filtered_df[display_cols + ['Return']], 
                         column_config={
                             "Return_Pct": None, # Hide raw
                             "Return": st.column_config.TextColumn("Return"),
                             "Type": st.column_config.TextColumn("Signal Type"),
                             "Timeframe": st.column_config.TextColumn("TF"),
                         },
                         use_container_width=True,
                         hide_index=True
                     )
                 else:
                     st.info("No trades in this period.")
                 
             else:
                 st.info("No history available. Please wait for the Runic Alerts to refresh.")

        elif st.session_state.active_tab == 'RISK':
            render_prop_risk()
            
        elif st.session_state.active_tab == 'SPELLBOOK':
             st.markdown("### 📘 Grimoire of Knowledge")
             st.markdown("Upload your trade screenshots here for NLP analysis and feedback.")
             st.file_uploader("Analyze Rune (Upload)", type=['png', 'jpg'], label_visibility="collapsed")
             
             # Placeholder for future "Spellbook" features (Journal, logs, etc)
             st.info("The Grimoire is open. Future enchantments pending.")

        elif st.session_state.active_tab == 'RULES':
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
            # TradingView Widget
            # Use Active Symbol or Fallback
            tv_sym = st.session_state.get('active_tv_symbol', 'COINBASE:BTCUSD')
            tv_int = st.session_state.get('active_tv_interval', '60')
            
            # Helper to generate TV URL
            clean_sym = tv_sym.replace("BINANCE:", "").replace("COINBASE:", "").replace("OANDA:", "")
            tv_url = f"https://www.tradingview.com/chart?symbol={tv_sym}"
            
            st.caption(f"**active:** {tv_sym} ({tv_int}m) | [Open in TradingView ↗]({tv_url})")
            
            tv_widget_code = f"""
            <div class="tradingview-widget-container" style="height:100%;width:100%">
              <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget(
              {{
              "width": "100%",
              "height": "600", 
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
            components.html(tv_widget_code, height=620, scrolling=False)
            
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
    # 1. Realms (Market Sessions)
    with st.container(border=True):
        st.markdown('<div class="runic-header">REALMS</div>', unsafe_allow_html=True)
        
        # Current Time EST
        now_est = pd.Timestamp.now(tz='America/New_York')
        curr_hour = now_est.hour
        curr_min = now_est.minute
        
        # Sessions (Start Hour, End Hour, Name). Using 0-24 scale.
        # Sydney: 5pm - 2am (17 - 2)
        # Tokyo: 7pm - 4am (19 - 4)
        # London: 3am - 12pm (3 - 12)
        # New York: 8am - 5pm (8 - 17)
        
        sessions = [
            {"name": "Sydney", "start": 17, "end": 26}, # 26 = 2am next day
            {"name": "Tokyo", "start": 19, "end": 28}, # 28 = 4am next day
            {"name": "London", "start": 3, "end": 12},
            {"name": "New York", "start": 8, "end": 17}
        ]
        
        # HTML Gen
        session_html = ""
        current_time_pct = ((curr_hour + curr_min/60) / 24) * 100
        
        for sess in sessions:
            # Normalize for timeline 0-24
            # Handle wrapping: if start > end (e.g. 17 - 2), we treat it as 17 to 26 for calc, but display might need split?
            # Easiest: Canvas is 0-24. 
            # Sydney (17-26) -> 17-24 (Part 1) AND 0-2 (Part 2)
            
            s_real = sess['start']
            e_real = sess['end']
            
            # Check Active
            # Convert current hour to 'extended' if needed? 
            # Simpler: Check interval
            is_active = False
            
            # Adjusted current for check
            # if 17 <= curr < 24 OR 0 <= curr < 2
            
            # Logic for wrapping check
            s_mod = s_real % 24
            e_mod = e_real % 24
            if s_mod > e_mod: # Wraps midnight
                if curr_hour >= s_mod or curr_hour < e_mod: is_active = True
            else:
                if s_mod <= curr_hour < e_mod: is_active = True
            
            # Text Logic
            status_text = ""
            if is_active:
                # Calc time left
                # Target is End
                # handle wrap
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
                # Calc time to start
                target_h = s_mod
                if target_h < curr_hour: target_h += 24 # Begins tomorrow
                
                diff_h = target_h - curr_hour
                diff_m = 0 - curr_min
                total_min = diff_h * 60 + diff_m
                h_left = total_min // 60
                m_left = total_min % 60
                
                status_text = f"Begins in {h_left}hr {m_left}min"
                bar_color = "#4a4a60" # Grey
                text_color = "#aaa"
                
            # Render Bars (Handles wrapping by drawing two if needed)
            bars_svg = ""
            
            # Helper to draw rect
            def draw_rect(s, e, col):
                width = (e - s) / 24 * 100
                left = (s / 24) * 100
                return f'<div style="position: absolute; left: {left}%; top: 5px; height: 20px; width: {width}%; background-color: {col}; border-radius: 4px; display: flex; align-items: center; padding-left: 5px; white-space: nowrap; overflow: hidden;"></div>'
            
            if s_real >= 24: # Should not happen with initial definition
                pass
            elif e_real > 24:
                # Split: Start->24 using green/grey
                bars_svg += draw_rect(s_real, 24, bar_color)
                # Split: 0->End-24
                bars_svg += draw_rect(0, e_real-24, bar_color)
            else:
                bars_svg += draw_rect(s_real, e_real, bar_color)
                
            # Text Overlay (Centered relative to container or explicit?)
            # Just use a row layout similar to Forex Factory
            # [Name  Time]  [Bar area]
            
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
<!-- Timeline Header 0 - 24 -->
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

    
    # 2. Oracle (Countdown to next Economic Event)
    with st.container(border=True):
        st.markdown('<div class="runic-header">ORACLE</div>', unsafe_allow_html=True)
        
        # Economic Calendar (Hardcoded for 2025/2026)
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
                    min-height: 210px; 
                    display: flex; 
                    flex-direction: column; 
                    justify-content: flex-start;
                    padding-top: 10px;
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
    
    # 3. Great Wizard
    # 3. Great Sorcerer
    with st.container(border=True):
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
                b64_str = base64.b64encode(img_file.read()).decode()
            bg_style = f"background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.8)), url('data:image/png;base64,{b64_str}'); background-size: cover; background-position: center;"
        except Exception as e:
            pass # Fallback to default dark theme
            
        st.markdown(f"""
            <div style="
                font-family: 'Cinzel', serif; 
                color: #f0e6d2; 
                text-align: center; 
                font-style: italic; 
                line-height: 1.6; 
                min-height: 222px;
                display: flex; 
                align-items: center; 
                justify-content: center;
                padding: 15px;
                margin-top: 15px;
                margin-bottom: 5px;
                border-radius: 8px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.9);
                {bg_style}
            ">
                <div style="background: rgba(11, 12, 21, 0.7); padding: 20px; border: 1px solid #c5a059; border-radius: 2px; box-shadow: 0 0 20px rgba(0,0,0,0.8);">
                    "{selected_quote}"
                </div>
            </div>
        """, unsafe_allow_html=True)

# --- LEFT COLUMN: MANA, SPELLS, ALERTS ---
with col_left:
    # 1. Mana Pool
    with st.container(border=True):
        st.markdown('<div class="runic-header">MANA POOL</div>', unsafe_allow_html=True)
        
        # Calculate Percentage
        mana_pct = max(0, min(100, (st.session_state.mana / 425) * 100))
        
        st.markdown(f"""
            <div style="background-color: #0b0c15; border: 1px solid #444; border-radius: 6px; height: 22px; margin-bottom: 2px; position: relative; box-shadow: inset 0 0 10px #000;">
                <div style="
                    background: linear-gradient(90deg, #00eaff 0%, #00ff88 100%);
                    width: {mana_pct}%; 
                    height: 100%; 
                    border-radius: 5px; 
                    box-shadow: 0 0 15px rgba(0, 255, 136, 0.6); 
                    transition: width 0.5s ease-out;
                "></div>
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; text-shadow: 0 1px 4px black; letter-spacing: 1px;">
                    {st.session_state.mana} / 425
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Compact Buttons in One Line (CSS Injection for small buttons)
        # Compact Buttons in One Line (CSS Injection for small buttons AND Multiselect Tags)
        st.markdown("""
            <style>
            /* Compact Buttons */
            div[data-testid="stColumn"] button {
                padding: 0px 2px !important;
                min-height: 0px !important;
                height: 28px !important;
                font-size: 0.7rem !important;
                line-height: 1 !important;
            }
            /* Compact Multiselect Tags */
            span[data-baseweb="tag"] {
                font-size: 0.65rem !important;
                padding: 0px 4px !important;
                height: 20px !important;
                margin-top: 2px !important;
                margin-bottom: 2px !important;
            }
            </style>
        """, unsafe_allow_html=True)



    # 2. Spells Left & Reset Logic
    # Check Resets
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
    except:
        pass # Handle potential timestamp errors gracefully

    with st.container(border=True):
        st.markdown('<div class="runic-header">SPELLS LEFT</div>', unsafe_allow_html=True)
        
        # Custom HTML for Spells
        s_day = st.session_state.spells_day
        s_week = st.session_state.spells_week
        
        st.markdown(f"""
            <div class="spell-card-container">
                <div class="spell-card spell-card-day">
                    <span class="spell-icon">⚡</span>
                    Day: <span class="spell-value">{s_day}</span>
                </div>
                <div class="spell-card spell-card-week">
                    <span class="spell-icon">⚡</span>
                    Week: <span class="spell-value">{s_week}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # 3. Runic Trade Alerts
    show_runic_alerts()



# --- Auto Refresh Logic ---


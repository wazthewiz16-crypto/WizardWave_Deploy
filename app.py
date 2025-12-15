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

# Page Config
st.set_page_config(
    page_title="Wizard Wave V9 ML Dashboard",
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
if 'mana' not in st.session_state:
    st.session_state['mana'] = 500
if 'spells_day' not in st.session_state:
    st.session_state['spells_day'] = 2
if 'spells_week' not in st.session_state:
    st.session_state['spells_week'] = 5
if 'last_reset_day' not in st.session_state:
    st.session_state['last_reset_day'] =  pd.Timestamp.now().floor('D')
if 'last_reset_week' not in st.session_state:
    st.session_state['last_reset_week'] = pd.Timestamp.now().to_period('W').start_time

# --- ML Model Integration ---
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

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
            # 15m is often slow due to volume/parsing, reduced history is acceptable for live monitoring
            current_limit = 200 if tf_code == '15m' else 600
            
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
                    "Type": type_display,
                    "Timeframe": timeframe_label,
                    "Entry_Time": ts_str,
                    "Signal_Time": ts_str, # Redundant but safe
                    "Entry_Price": f"{ep:.2f}",
                    "Take_Profit": f"{tp_price:.2f}",
                    "Stop_Loss": f"{sl_price:.2f}",
                    "Current_Price": f"{df_strat.iloc[-1]['close']:.2f}",
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
            signals_df = df_strat[df_strat['signal_type'] != 'NONE'].tail(20).copy()
            asset_history = []
            
            # Limit history to reasonable amount for display, but simulate all available?
            # User wants equity curve of "generated in above table". That usually implies recent history. 
            # But recent 50 might not be enough for a curve. Let's take all fetched signals.
            
            for start_time in signals_df.index:
                row = signals_df.loc[start_time]
                signal_type = row['signal_type']
                model_prob = row['model_prob']
                
                # Simulation
                # Only if using ML filter
                if model_prob > 0.40:
                    pass
                    # ret_pct, sl_pct = run_simulation(df_strat, df_strat.index.get_loc(start_time), signal_type, asset['type'], None)
                    
                    # # Store Results
                    # asset_history.append({
                    #     "_sort_key": start_time,
                    #     "Asset": asset['name'],
                    #     "Timeframe": timeframe_label,
                    #     "Time": format_time(start_time),
                    #     "Type": signal_type,
                    #     "Price": row['close'],
                    #     "Confidence": f"{model_prob:.0%}",
                    #     "Model": "✅",
                    #     "Return_Pct": 0, # Placeholder
                    #     "SL_Pct": 0      # Placeholder
                    # })
            
            return result_data, active_trade_data, asset_history
            
        except Exception as e:
            # print(f"Error {asset['symbol']}: {e}")
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
PROP_FIRM_CONFIGS = {
    "APEX": {"type": "Futures", "maxDrawdown": 0.045, "dailyDrawdown": 0.025, "profitTarget": 0.06},
    "TOPSTEP": {"type": "Futures", "maxDrawdown": 0.04, "dailyDrawdown": 0.02, "profitTarget": 0.06},
    "FTMO": {"type": "Forex", "maxDrawdown": 0.10, "dailyDrawdown": 0.05, "profitTarget": 0.10}
}

def init_prop_accounts():
    if 'user_accounts' not in st.session_state:
        st.session_state.user_accounts = [
            {"id": 1, "name": "Apex 50k - #1", "firm": "APEX", "size": 50000, "currentBalance": 50000, "startOfDayBalance": 50000},
            {"id": 2, "name": "Topstep 50k", "firm": "TOPSTEP", "size": 50000, "currentBalance": 50000, "startOfDayBalance": 50000}
        ]

def render_prop_risk():
    st.markdown("### 🛡️ Prop Firm Manager")
    init_prop_accounts()
    
    # Grid Layout
    cols = st.columns(2)
    
    for i, account in enumerate(st.session_state.user_accounts):
        config = PROP_FIRM_CONFIGS[account['firm']]
        size = account['size']
        
        # Use col 0 or 1
        with cols[i % 2]:
            with st.container(border=True):
                # Header
                st.markdown(f"**{account['name']}** <span style='color:#888'>({account['firm']})</span>", unsafe_allow_html=True)
                st.markdown(f"Size: **${size:,}**")
                
                # Inputs (Balance Updates)
                c_bal, c_sod = st.columns(2)
                cur_bal = c_bal.number_input("Current Balance", value=float(account['currentBalance']), step=100.0, key=f"bal_{account['id']}")
                sod_bal = c_sod.number_input("Start Day", value=float(account['startOfDayBalance']), step=100.0, key=f"sod_{account['id']}")
                
                # Update State
                account['currentBalance'] = cur_bal
                account['startOfDayBalance'] = sod_bal
                
                # Calculations
                target_bal = size * (1 + config['profitTarget'])
                max_dd_amount = size * config['maxDrawdown']
                max_dd_level = size - max_dd_amount
                
                daily_dd_amount = size * config['dailyDrawdown'] 
                daily_dd_level = sod_bal - daily_dd_amount
                
                dist_target = target_bal - cur_bal
                dist_max_dd = cur_bal - max_dd_level
                dist_daily_dd = cur_bal - daily_dd_level
                
                max_loss = min(dist_max_dd, dist_daily_dd)
                
                # Progress to Target
                st.markdown("---")
                tgt_pct = max(0.0, min(1.0, (cur_bal - size) / (target_bal - size))) if target_bal > size else 0
                st.caption(f"Target: ${target_bal:,.0f}")
                st.progress(tgt_pct)
                if dist_target <= 0:
                    st.success("✅ PASSED")
                else:
                    st.caption(f"To Go: ${dist_target:,.0f}")
                
                # Drawdown Risk
                st.markdown(f"**Max Loss Available: :red[${max_loss:,.2f}]**")
                
                # Risk Calculator
                st.markdown("#### ⚖️ Position Sizing")
                r1, r2 = st.columns(2)
                risk_1pct = cur_bal * 0.01
                rec_lots = risk_1pct / 500
                r1.metric("1% Risk", f"${risk_1pct:.0f}")
                r2.metric("Rec. Lots", f"{rec_lots:.2f}")

# --- Render UI ---

# Tabs removed for cleaner UI
# [tab_dash] = st.tabs(["⚡ Active Signals Dashboard"])

# We collect data first (Aggregate 4 Timeframes)
# NOTE: To save time for this specific request, checking only 1D and 4H would be faster,
# but the user might want all. I'll stick to running all but optimizing.
# Actually, let's run them all.

# Check Session State for Data
if 'combined_active_df' not in st.session_state:
    status_msg = st.empty()
    status_msg.info("Running Analysis on 15m, 30m, 1H, 4H, 1D, 4D...")

    # Run All Timeframes
    r15m, a15m, h15m = analyze_timeframe("15 Minutes")
    r30m, a30m, h30m = analyze_timeframe("30 Minutes")
    r1h, a1h, h1h = analyze_timeframe("1 Hour")
    r4h, a4h, h4h = analyze_timeframe("4 Hours")
    r1d, a1d, h1d = analyze_timeframe("1 Day")
    r4d, a4d, h4d = analyze_timeframe("4 Days")

    # Clear the status message
    status_msg.empty()

    # Consolidate
    active_dfs = [df for df in [a15m, a30m, a1h, a4h, a1d, a4d] if not df.empty]
    combined_active = pd.concat(active_dfs).sort_values(by='_sort_key', ascending=False) if active_dfs else pd.DataFrame()
    
    all_history = h15m + h30m + h1h + h4h + h1d + h4d
    hist_df = pd.DataFrame(all_history)
    if not hist_df.empty:
        hist_df.sort_values(by='_sort_key', ascending=False, inplace=True)
        
    # Save to Session State
    st.session_state['combined_active_df'] = combined_active
    st.session_state['hist_df'] = hist_df

else:
    # Load from Session State
    combined_active = st.session_state['combined_active_df']
    hist_df = st.session_state['hist_df']
    
    # --- Equity Curve Calculation ---
    # Filter for trades with PnL data (simulated ones)
    if 'Return_Pct' in hist_df.columns:
        pass
        # # Sort Ascending for Curve Calculation
        # curve_df = hist_df.dropna(subset=['Return_Pct']).sort_values(by='_sort_key', ascending=True)
        
        # # Limit to last 200 trades
        # curve_df = curve_df.tail(200).copy()
        
        # initial_balance = 50000
        # balance = initial_balance
        # balances = [initial_balance]
        # dates = [] # Unused for x-axis now
        
        # for idx, row in curve_df.iterrows():
        #     # 1% Risk Strategy
        #     # Risk Amount = 1% of Current Balance (Compounding)
        #     risk_amount = balance * 0.01
        #     sl_pct = row['SL_Pct']
        #     return_pct = row['Return_Pct']
            
        #     # PnL = (Return / SL_Dist) * Risk_Amount
        #     if sl_pct > 0:
        #        pnl = (return_pct / sl_pct) * risk_amount
        #     else:
        #        pnl = 0
               
        #     balance += pnl
        #     balances.append(balance)
        #     dates.append(row['_sort_key'])
            
        # # Create Curve DF for Chart
        # # X-Axis: Trade Count (Implicit Index)
        # # Y-Axis: Equity
        # eq_df = pd.DataFrame({'Equity': balances})
        
        # st.markdown("### 📈 Simulated Equity Curve ($50k Start, 1% Risk)")
        # st.markdown(f"**Current Simulated Balance: ${balance:,.2f} (Trades: {len(curve_df)})**")
        # st.area_chart(eq_df, color="#00FF00")

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
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* SUPER ROBUST MAGICAL BORDER SELECTOR */
    /* Target the container by its content */
    div[data-testid="stVerticalBlock"]:has(.runic-header),
    div[data-testid="stVerticalBlockBorderWrapper"],
    .stVerticalBlockBorderWrapper {
        background-color: #0b0c15;
        /* Ultra-Thick Magical Frame */
        border: 5px solid #ffd700 !important;
        border-radius: 12px !important;
        
        /* Maximum Glow Power */
        box-shadow: 
            0 0 25px rgba(255, 215, 0, 0.6) !important, /* Outer Gold Halo */
            0 0 50px rgba(197, 160, 89, 0.4) !important, /* Distant Haze */
            inset 0 0 30px #000000 !important;          /* Deep Void Inner */
            
        padding: 20px !important;
        margin-bottom: 30px !important;
        position: relative;
        z-index: 1;
    }

    /* Epic Invoke Button */
    div.stButton > button[kind="primary"] {
        width: 100%;
        height: 60px;
        background: linear-gradient(135deg, #c5a059 0%, #8a6e3c 100%);
        color: #000;
        font-family: 'Cinzel', serif;
        font-size: 1.5rem;
        font-weight: bold;
        border: 2px solid #ffd700;
        box-shadow: 0 0 20px rgba(197, 160, 89, 0.6);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(197, 160, 89, 0.9);
        color: #fff;
    }

    .runic-header {

        text-align: center;
        color: #c5a059;
        font-size: 1.2rem;
        font-weight: bold;
        border-bottom: 1px solid #4a3b22;
        padding-bottom: 10px;
        margin-bottom: 15px;
        text-shadow: 0 0 5px #c5a059;
    }

    .runic-item {
        display: flex;
        align-items: center;
        background: linear-gradient(90deg, rgba(20, 20, 30, 0.8) 0%, rgba(35, 35, 50, 0.8) 100%);
        border: 1px solid #4a4a60;
        border-left: 4px solid #444;
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }

    .runic-item:hover {
        box-shadow: 0 0 10px rgba(197, 160, 89, 0.2);
        border-color: #c5a059;
    }

    .runic-icon {
        font-size: 24px;
        margin-right: 15px;
        width: 30px;
        text-align: center;
    }

    .runic-content {
        flex-grow: 1;
    }

    .runic-title {
        font-family: 'Lato', sans-serif;
        font-weight: bold;
        font-size: 0.95rem;
        color: #e0e0e0;
        margin: 0;
    }

    .runic-subtitle {
        font-family: 'Lato', sans-serif;
        font-size: 0.8rem;
        color: #888;
        margin: 2px 0 0 0;
    }

    .bullish { border-left-color: #00ff88; }
    .bearish { border-left-color: #ff3344; }
    

    .bullish .runic-icon { color: #00ff88; text-shadow: 0 0 8px rgba(0, 255, 136, 0.5); }
    .bearish .runic-icon { color: #ff3344; text-shadow: 0 0 8px rgba(255, 51, 68, 0.5); }

    /* Spell Cards Styling */
    .spell-card-container {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin-bottom: 5px;
    }
    .spell-card {
        flex: 1;
        background-color: #0b0c15; /* Dark background */
        border: 2px solid; 
        border-radius: 8px; /* Slightly rounded corners */
        padding: 5px 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Cinzel', serif;
        font-weight: bold;
        font-size: 1.0rem;
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
        margin-right: 8px;
        font-size: 1.2rem;
    }
    
    .spell-value {
        color: #fff;
        margin-left: 5px;
        text-shadow: 0 0 5px currentColor;
    }

    /* Arcane Portal Header */
    .arcane-header-container {
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        padding: 20px 0;
        background: radial-gradient(circle at center, rgba(11, 12, 21, 0) 0%, rgba(11, 12, 21, 0.8) 100%);
        border-bottom: 2px solid transparent; 
        border-image: linear-gradient(90deg, transparent, #c5a059, transparent) 1;
    }

    .arcane-title {
        font-family: 'Cinzel', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #f0e6d2; /* Light gold/parchment base */
        text-transform: uppercase;
        letter-spacing: 8px;
        
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

    </style>
""", unsafe_allow_html=True)

# --- ARCANE PORTAL HEADER ---
st.markdown("""
    <div class="arcane-header-container">
        <div class="arcane-title">Arcane Portal</div>
        <div class="arcane-decoration"></div>
    </div>
""", unsafe_allow_html=True)

# Layout Columns
col_left, col_center, col_right = st.columns([0.25, 0.5, 0.25], gap="medium")

# --- CENTER COLUMN: MAIN PORTAL ---
with col_center:
    with st.container(border=True):
        # Interactive Navbar
        if 'active_tab' not in st.session_state: st.session_state.active_tab = 'PORTAL'
        
        def set_tab(t):
            st.session_state.active_tab = t
            
        c1, c2, c3, c4 = st.columns(4)
        c1.button("PORTAL", use_container_width=True, type="primary" if st.session_state.active_tab=='PORTAL' else "secondary", on_click=set_tab, args=('PORTAL',))
        
        c2.button("PROP RISK", use_container_width=True, type="primary" if st.session_state.active_tab=='RISK' else "secondary", on_click=set_tab, args=('RISK',))
        
        # Placeholders
        c3.button("RULES", use_container_width=True, disabled=True)
        c4.button("SPELLBOOK", use_container_width=True, disabled=True)
        
        st.markdown("---")
        
        if st.session_state.active_tab == 'RISK':
            render_prop_risk()
        
        elif st.session_state.active_tab == 'PORTAL':
            # TradingView Widget
            tv_widget_code = """
            <div class="tradingview-widget-container" style="height:100%;width:100%">
              <div id="tradingview_chart" style="height:calc(100% - 32px);width:100%"></div>
              <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
              <script type="text/javascript">
              new TradingView.widget(
              {
              "width": "100%",
              "height": "100%",
              "symbol": "COINBASE:BTCUSD",
              "interval": "60",
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
                "RSI@tv-basicstudies",
                "MASimple@tv-basicstudies"
              ]
            }
              );
              </script>
            </div>
            """
            components.html(tv_widget_code, height=800, scrolling=False)
            
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            
            # Invoke Button
            if st.button("INVOKE SPELL", use_container_width=True, type="primary"):
                st.toast("🔮 Invocation Ritual Started... (Modal Placeholder)")
                # In a real app, this would open a st.dialog or st.expander with the checklist


# --- RIGHT COLUMN: STATS, ORACLE, WIZARD ---
with col_right:
    # 1. Trade Stats
    with st.container(border=True):
        st.markdown('<div class="runic-header">TRADE STATS</div>', unsafe_allow_html=True)
        st.file_uploader("Analyzre Rune (Upload)", type=['png', 'jpg'], label_visibility="collapsed")
        st.markdown("<div style='text-align: center; color: #666; font-size: 0.8rem;'>Upload screenshot for NLP feedback</div>", unsafe_allow_html=True)
    
    # 2. Oracle (Countdown to next 4H candle)
    with st.container(border=True):
        st.markdown('<div class="runic-header">ORACLE</div>', unsafe_allow_html=True)
        
        # Calculate time to next 4H candle (00, 04, 08, 12, 16, 20 UTC)
        now_utc = pd.Timestamp.now('UTC')
        current_hour = now_utc.hour
        next_hour = ((current_hour // 4) + 1) * 4
        if next_hour >= 24: next_hour = 0
        
        # Target time
        if next_hour == 0:
            target = (now_utc + pd.Timedelta(days=1)).normalize()
        else:
            target = now_utc.normalize() + pd.Timedelta(hours=next_hour)
            
        remaining = target - now_utc
        hours = remaining.seconds // 3600
        minutes = (remaining.seconds % 3600) // 60
        seconds = remaining.seconds % 60
        
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        st.markdown(f"""
            <div style="text-align: center;">
                <div style="font-size: 0.8rem; color: #a0c5e8;">NEXT CAST IN</div>
                <div style="font-size: 2.5rem; font-weight: bold; color: white; text-shadow: 0 0 10px #a0c5e8;">{time_str}</div>
            </div>
        """, unsafe_allow_html=True)
    
    # 3. Great Wizard
    with st.container(border=True):
        st.markdown('<div class="runic-header">GREAT WIZARD</div>', unsafe_allow_html=True)
        
        quotes = [
            "The market is a mirror of the mind.",
            "Clarity comes not from the chart, but from the discipline within.",
            "Do not chase the dragon; let it come to you.",
            "Patience is the wizard's greatest spell.",
            "Risk is the mana you pay for the reward you seek.",
            "A calm mind sees the trend; a chaotic mind sees only noise."
        ]
        import random
        selected_quote = random.choice(quotes)
        
        st.markdown(f"""
            <div style="font-family: 'Cinzel', serif; color: #ffd700; text-align: center; font-style: italic; line-height: 1.6;">
                "{selected_quote}"
            </div>
        """, unsafe_allow_html=True)

# --- LEFT COLUMN: MANA, SPELLS, ALERTS ---
with col_left:
    # 1. Mana Pool
    with st.container(border=True):
        st.markdown('<div class="runic-header">MANA POOL</div>', unsafe_allow_html=True)
        
        # Calculate Percentage
        mana_pct = max(0, min(100, (st.session_state.mana / 500) * 100))
        
        # Glowing Gradient Bar
        st.markdown(f"""
            <div style="background-color: #0b0c15; border: 1px solid #444; border-radius: 6px; height: 30px; margin-bottom: 12px; position: relative; box-shadow: inset 0 0 10px #000;">
                <div style="
                    background: linear-gradient(90deg, #00eaff 0%, #00ff88 100%);
                    width: {mana_pct}%; 
                    height: 100%; 
                    border-radius: 5px; 
                    box-shadow: 0 0 15px rgba(0, 255, 136, 0.6); 
                    transition: width 0.5s ease-out;
                "></div>
                <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white; text-shadow: 0 1px 4px black; letter-spacing: 1px;">
                    {st.session_state.mana} / 500
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

        # Adjusted columns to fit -100
        cols = st.columns([1, 1, 1, 1, 1, 1.3, 0.8], gap="small")
        if cols[0].button("-1", key="m_1", use_container_width=True): st.session_state.mana = max(0, st.session_state.mana - 1)
        if cols[1].button("-5", key="m_5", use_container_width=True): st.session_state.mana = max(0, st.session_state.mana - 5)
        if cols[2].button("-10", key="m_10", use_container_width=True): st.session_state.mana = max(0, st.session_state.mana - 10)
        if cols[3].button("-20", key="m_20", use_container_width=True): st.session_state.mana = max(0, st.session_state.mana - 20)
        if cols[4].button("-50", key="m_50", use_container_width=True): st.session_state.mana = max(0, st.session_state.mana - 50)
        if cols[5].button("-100", key="m_100", use_container_width=True): st.session_state.mana = max(0, st.session_state.mana - 100)
        if cols[6].button("↺", key="m_rst", help="Reset", use_container_width=True): st.session_state.mana = 500

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
    with st.container(border=True):
        # Header Row with Refresh Button
        c_btn, c_title, c_emp = st.columns([0.2, 0.6, 0.2])
        with c_btn:
            if st.button("↻", key="refresh_top", help="Refresh"):
                 # Clear Cache
                if 'combined_active_df' in st.session_state:
                    del st.session_state['combined_active_df']
                if 'hist_df' in st.session_state:
                    del st.session_state['hist_df']
                st.rerun()
        with c_title:
             st.markdown('<div class="runic-header" style="font-size: 1rem; border: none; margin-bottom: 0;">ALERTS</div>', unsafe_allow_html=True)
        
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
            selected_short = st.multiselect("Timeframes", options=display_opts, default=display_opts, label_visibility="collapsed")
            
            # Map back for filtering
            selected_tfs = [tf_map_rev.get(x, x) for x in selected_short]
            
            # --- Filter Data ---
            df_display = combined_active.copy()
            
            if show_take_only:
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
                
                # Build HTML for Items
                html_content = ""
                
                for index, row in current_batch.iterrows():
                    is_long = "LONG" in row['Type']
                    direction_class = "bullish" if is_long else "bearish"
                    
                    # Icon Logic
                    asset_name = row['Asset']
                    
                    # SVG Bolt for reliable coloring
                    icon_bolt = """<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" stroke="none" style="display: block; margin: 0 auto;"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"></path></svg>"""
                    
                    icon = icon_bolt
                    if "BTC" in asset_name: icon = "₿"
                    elif "ETH" in asset_name: icon = "Ξ"
                    elif "SOL" in asset_name: icon = "◎"
                    elif "Short" in row['Type']: icon = "⬇"
                    elif "Long" in row['Type']: icon = "⬆"
                    
                    action_text = "BULL" if is_long else "BEAR"
                    signal_desc = f"{asset_name}: {action_text}"
                    
                    # Data Points
                    # Data Points
                    conf = row.get('Confidence', 'N/A')
                    entry_time = row.get('Entry_Time', row.get('Entry Time', 'N/A'))
                    tf = row.get('Timeframe', 'N/A')
                    action_val = row.get('Action', 'TAKE') # Usually '✅ TAKE'

                    pnl_val = row['PnL (%)']
                    pnl_color = "#00ff88" if not str(pnl_val).startswith("-") else "#ff3344"

                    # Price Logic
                    entry_p = row.get('Entry_Price', 'N/A')
                    tp_p = row.get('Take_Profit', 'N/A')
                    sl_p = row.get('Stop_Loss', 'N/A')
                    
                    # Try to get Current Price (if available in DF, else Placeholder)
                    current_p = row.get('Current_Price', row.get('Close', None))
                    current_html = ""
                    if current_p is not None:
                         current_html = f"<span style='color: #ffd700; margin-right: 8px;'>Now: {current_p}</span>"
                    else:
                         # Fallback/Placeholder
                         current_html = "<span style='color: #666; margin-right: 8px; font-size: 0.8em;'>(Live Pending)</span>"

                    # Format Entry/TP/SL
                    details = f"Entry: {entry_p} | TP: {tp_p} | SL: {sl_p}"
                    
                    # Determine action icon based on action_val
                    action_icon = "✅" if "TAKE" in action_val else "➡️" # Default to arrow if not TAKE

                    html_content += f"""
<div class="runic-item {direction_class}" style="padding: 8px;">
<div class="runic-icon" style="font-size: 20px; margin-right: 10px;">{icon}</div>
<div class="runic-content">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div class="runic-title" style="font-size: 0.85rem;">{signal_desc}</div>
<div style="font-weight: bold; color: {pnl_color}; font-size: 0.85rem;">{pnl_val}</div>
</div>
<div style="font-size: 0.75rem; color: #e0e0e0; margin-top: 2px;">
{current_html} {action_val} | Conf: {conf} | {tf}
</div>
<div style="font-size: 0.7rem; color: #aaa; margin-top: 2px;">
{details}
</div>
<div style="font-size: 0.65rem; color: #666; margin-top: 2px;">
Time: {entry_time}
</div>
</div>
</div>
"""
                st.markdown(html_content, unsafe_allow_html=True)
                
                # --- Compact Numbered Pagination ---
                p1, p2, p3 = st.columns([0.2, 0.6, 0.2])
                with p1:
                    if st.button("◀", key="prev_main", disabled=(st.session_state.page_number == 0)):
                        st.session_state.page_number -= 1
                        st.rerun()
                with p3:
                    if st.button("▶", key="next_main", disabled=(st.session_state.page_number >= total_pages - 1)):
                        st.session_state.page_number += 1
                        st.rerun()
                with p2:
                    st.markdown(f"<div style='text-align: center; color: #888; font-size: 0.8rem; padding-top: 5px;'>Page {st.session_state.page_number + 1}/{total_pages}</div>", unsafe_allow_html=True)

        else:
            st.info("No active signals.")



# --- Auto Refresh Logic ---
# Force Refetch every 60 seconds
time.sleep(60)
if 'combined_active_df' in st.session_state:
    del st.session_state['combined_active_df']
if 'hist_df' in st.session_state:
    del st.session_state['hist_df']
st.rerun()

import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import concurrent.futures
import joblib
from data_fetcher import fetch_data
from strategy import WizardWaveStrategy

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
                    "Entry Time": ts_str,
                    "Entry Price": f"{ep:.2f}",
                    "Take Profit": f"{tp_price:.2f}",
                    "Stop Loss": f"{sl_price:.2f}",
                    "PnL (%)": f"{trade['PnL (%)']:.2f}%",
                    "Confidence": f"{entry_conf:.0%}",
                    "Action": rec_action
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

    /* Target native container to look like Runic Box */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #0b0c15;
        border: 2px solid #c5a059 !important;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(197, 160, 89, 0.3);
        padding: 15px;
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
    </style>
""", unsafe_allow_html=True)

with st.container(border=True):
    # Header Row with Refresh Button
    c_btn, c_title, c_emp = st.columns([0.15, 0.7, 0.15])
    with c_btn:
        if st.button("Refresh", key="refresh_top"):
             # Clear Cache
            if 'combined_active_df' in st.session_state:
                del st.session_state['combined_active_df']
            if 'hist_df' in st.session_state:
                del st.session_state['hist_df']
            st.rerun()
    with c_title:
         st.markdown('<div class="runic-header">RUNIC TRADE ALERTS</div>', unsafe_allow_html=True)
    
    # --- Timeframe Filter (Inside Box) ---
    if not combined_active.empty:
        # Get unique timeframes and sort chronologically
        tf_order = {
            "15 Minutes": 0, "30 Minutes": 1, 
            "1 Hour": 2, "4 Hours": 3, 
            "1 Day": 4, "4 Days": 5
        }
        
        unique_tfs = combined_active['Timeframe'].unique().tolist()
        available_tfs = sorted(unique_tfs, key=lambda x: tf_order.get(x, 99))
        
        # Multiselect with Label hidden for cleaner look? User asked "underneath", standard label is fine.
        selected_tfs = st.multiselect("Filter Timeframes", options=available_tfs, default=available_tfs)
        
        # --- Filter Data ---
        df_display = combined_active.copy()
        
        if show_take_only:
             df_display = df_display[df_display['Action'].str.contains("TAKE")]
        
        if selected_tfs:
            df_display = df_display[df_display['Timeframe'].isin(selected_tfs)]
        else:
            st.warning("Please select at least one timeframe.")
            df_display = pd.DataFrame(columns=df_display.columns) # Empty

        # --- Render ---
        if df_display.empty:
            st.info("No active signals match your filters.")
        else:
            # Pagination Logic
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
                
                action_text = "BULLISH" if is_long else "BEARISH"
                signal_desc = f"{asset_name}: {action_text}"
                
                # Data Points
                conf = row['Confidence']
                entry_time = row['Entry Time']
                entry_price = row['Entry Price']
                sl_price = row['Stop Loss']
                tp_price = row['Take Profit']
                pnl_val = row['PnL (%)']
                action_val = row['Action']
                tf = row['Timeframe']

                # Color for PnL
                pnl_color = "#00ff88" if not pnl_val.startswith("-") else "#ff3344"
                
                # IMPORTANT: No indentation inside the HTML string to strictly prevent markdown code block rendering
                html_content += f"""
<div class="runic-item {direction_class}">
<div class="runic-icon">{icon}</div>
<div class="runic-content">
<div style="display: flex; justify-content: space-between; align-items: center;">
<div class="runic-title">{signal_desc}</div>
<div style="font-weight: bold; color: {pnl_color};">{pnl_val}</div>
</div>
<div class="runic-subtitle" style="color: #e0e0e0; margin-top: 4px;">
{action_val} | Conf: {conf} | {tf}
</div>
<div class="runic-subtitle" style="margin-top: 2px;">
Entry: {entry_price} | TP: {tp_price} | SL: {sl_price}
</div>
<div class="runic-subtitle" style="font-size: 0.75rem; color: #666; margin-top: 2px;">
Time: {entry_time}
</div>
</div>
</div>
"""
            st.markdown(html_content, unsafe_allow_html=True)
            
            # --- Advanced Numbered Pagination ---
            # Max visible pages: e.g. 7 (Prev, 1..5, Next)
            
            # Determine range of pages to show
            window = 2
            start_page = max(0, st.session_state.page_number - window)
            end_page = min(total_pages - 1, st.session_state.page_number + window)
            
            # Adjust if close to edges
            if end_page - start_page < 2 * window:
                if start_page == 0:
                    end_page = min(total_pages - 1, start_page + 2 * window)
                elif end_page == total_pages - 1:
                    start_page = max(0, end_page - 2 * window)
            
            page_range = range(start_page, end_page + 1)
            
            # Build Columns: Prev + [Pages] + Next
            total_cols = len(page_range) + 2
            cols = st.columns(total_cols)
            
            # Prev Button
            with cols[0]:
                if st.button("◀", key="prev_main", disabled=(st.session_state.page_number == 0)):
                    st.session_state.page_number -= 1
                    st.rerun()
                    
            # Page Number Buttons
            for i, p_idx in enumerate(page_range):
                col = cols[i+1]
                with col:
                    label = str(p_idx + 1)
                    if p_idx == st.session_state.page_number:
                        st.button(f"[{label}]", key=f"page_{p_idx}", disabled=True)
                    else:
                        if st.button(label, key=f"page_{p_idx}"):
                            st.session_state.page_number = p_idx
                            st.rerun()
            
            # Next Button
            with cols[-1]:
                if st.button("▶", key="next_main", disabled=(st.session_state.page_number >= total_pages - 1)):
                    st.session_state.page_number += 1
                    st.rerun()

    else:
        st.info("No active signals found.")



# --- Auto Refresh Logic ---
# Force Refetch every 60 seconds
time.sleep(60)
if 'combined_active_df' in st.session_state:
    del st.session_state['combined_active_df']
if 'hist_df' in st.session_state:
    del st.session_state['hist_df']
st.rerun()

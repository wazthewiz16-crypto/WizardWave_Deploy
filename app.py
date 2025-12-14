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
st.title("🧙‍♂️ Wizard Wave V9 Signals (Meta-Labeled)")

# Strategy Settings (Hardcoded)
lookback = 29
sensitivity = 1.06
cloud_spread = 0.64
zone_pad = 1.5

st.sidebar.subheader("App Settings")
auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=True)

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
            # Fetch Data
            df = fetch_data(asset['symbol'], asset['type'], timeframe=tf_code, limit=200)
            
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
                all_probs = model.predict_proba(df_strat[features])[:, 1]
                df_strat['model_prob'] = all_probs
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
                threshold = 0.6 # Low threshold for crypto?
                rec_action = "✅ TAKE" if entry_conf > threshold else "❌ SKIP"
                
                active_trade_data = {
                    "_sort_key": sort_ts,
                    "Asset": asset['name'],
                    "Type": trade['Position'],
                    "Timeframe": timeframe_label,
                    "Entry Time": ts_str,
                    "Entry Price": f"{trade['Entry Price']:.2f}",
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
            
            # --- Collect Historical Signals ---
            # Extract last 50 rows where signal_type != NONE
            signals_df = df_strat[df_strat['signal_type'] != 'NONE'].tail(50).copy()
            asset_history = []
            for idx, row in signals_df.iterrows():
                asset_history.append({
                    "_sort_key": idx,
                    "Asset": asset['name'],
                    "Timeframe": timeframe_label,
                    "Time": format_time(idx),
                    "Type": row['signal_type'],
                    "Price": row['close'],
                    "Confidence": f"{row['model_prob']:.0%}",
                    "Model": "✅" if row['model_prob'] > 0.6 else "❌"
                })
            
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

def format_time(ts):
    if pd.isna(ts): return "N/A"
    try:
        if ts.tz is None: ts = ts.tz_localize('UTC')
        ts_est = ts.tz_convert('America/New_York')
        return ts_est.strftime('%Y-%m-%d %H:%M:%S')
    except: return str(ts)

def highlight_confidence(row):
    try:
        val = float(row['Confidence'].strip('%'))
        if val >= 60:
            return ['background-color: rgba(0, 255, 0, 0.2)'] * len(row)
        else:
            return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
    except:
        return [''] * len(row)

# --- Render UI ---

tab_dash, tab_history = st.tabs(["⚡ Active Signals Dashboard", "📜 Search Signal History"])

# We collect data first (Aggregate 4 Timeframes)
# NOTE: To save time for this specific request, checking only 1D and 4H would be faster,
# but the user might want all. I'll stick to running all but optimizing.
# Actually, let's run them all.

status_msg = st.empty()
status_msg.info("Running Analysis on 1H, 4H, 1D, 4D...")

# Run All Timeframes
r1h, a1h, h1h = analyze_timeframe("1 Hour")
r4h, a4h, h4h = analyze_timeframe("4 Hours")
r1d, a1d, h1d = analyze_timeframe("1 Day")
r4d, a4d, h4d = analyze_timeframe("4 Days")

# Clear the status message
status_msg.empty()

# Consolidate
active_dfs = [df for df in [a1h, a4h, a1d, a4d] if not df.empty]
combined_active = pd.concat(active_dfs).sort_values(by='_sort_key', ascending=False) if active_dfs else pd.DataFrame()

all_history = h1h + h4h + h1d + h4d
hist_df = pd.DataFrame(all_history)
if not hist_df.empty:
    hist_df.sort_values(by='_sort_key', ascending=False, inplace=True)

with tab_dash:
    st.subheader("🔥 Active Signal Recommendations (Newest First)")
    
    # Filter Controls
    col_filter, _ = st.columns([0.3, 0.7])
    with col_filter:
        show_take_only = st.checkbox("Show only ✅ TAKE signals", value=True)
    
    if not combined_active.empty:
        df_display = combined_active.copy()
        
        if show_take_only:
             df_display = df_display[df_display['Action'].str.contains("TAKE")]
        
        cols = ["Timeframe", "Asset", "Type", "Entry Time", "Entry Price", "PnL (%)", "Confidence", "Action"]
        st.dataframe(df_display[cols].reset_index(drop=True).style.apply(highlight_confidence, axis=1), use_container_width=True)
    else:
        st.info("No active signals found on 4H/1D.")

with tab_history:
    st.subheader("📜 Recent Signal History & Probabilities")
    
    if not hist_df.empty:
        cols_hist = ["Timeframe", "Asset", "Time", "Type", "Price", "Confidence", "Model"]
        st.dataframe(hist_df[cols_hist].reset_index(drop=True).style.apply(highlight_confidence, axis=1), use_container_width=True)
    else:
        st.info("No history available.")

# Manual Refresh
if st.button("Refresh Analysis"):
    st.rerun()

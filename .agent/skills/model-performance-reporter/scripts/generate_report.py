
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import json
import joblib
import os
import sys
from datetime import datetime, timedelta

# Project Root Hack
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../.."))
sys.path.append(project_root)

from strategy import WizardWaveStrategy
from strategy_scalp import WizardScalpStrategy
import feature_engine 

# --- CONFIG ---
REPORT_FILE = "model_performance_report_180d.md"
DAYS = 180
WARMUP_DAYS = 200 # Extra days to fetch for EMA 200 calc

def fetch_data(symbol, interval, days):
    """Fetch history with buffer."""
    
    # Symbol mapping
    clean_sym = symbol
    if "USDT" in symbol: clean_sym = symbol.replace("/USDT", "-USD")
    elif "BTC/USD" in symbol: clean_sym = "BTC-USD"
    elif "ETH/USD" in symbol: clean_sym = "ETH-USD"
    
    try:
        # yfinance period mapping
        period = "1y"
        if days > 250: period = "2y" # Fetch enough for warmup
        
        df = yf.Ticker(clean_sym).history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        
        df = df.reset_index()
        df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df.set_index('datetime', inplace=True)
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    except:
        return pd.DataFrame()

def simulate_trades(df, strategy, model, tb_config, timeframe):
    """
    Simulates trades and returns a list of trade dicts.
    """
    trades = []
    
    # 1. Strategy & Features
    try:
        df = strategy.apply(df)
        df = feature_engine.calculate_ml_features(df)
    except Exception as e:
        # print(f"Error filtering {timeframe}: {e}")
        return []

    # 2. ML Prediction
    feat_cols = ['volatility', 'rsi', 'ma_dist', 'adx', 'mom', 'bb_width', 'candle_ratio', 'atr_pct', 'mfi'] 
    # NOTE: EXCLUDING 'rvol' to match current production model. Future updates will add it.
    
    # Fill Nans (for features)
    for c in feat_cols:
        if c not in df.columns: df[c] = 0.0
    
    df_clean = df.dropna(subset=feat_cols)
    if df_clean.empty: return []

    try:
        if model:
             df.loc[df_clean.index, 'prob'] = model.predict_proba(df_clean[feat_cols])[:, 1]
        else:
             df['prob'] = 1.0
    except:
        df['prob'] = 0.0
        
    df['prob'] = df['prob'].fillna(0.0)
    
    # 3. Trade Loop (Triple Barrier)
    threshold = 0.42 if timeframe == '1d' else 0.60 # Standard thresholds
    
    # TB Params
    pt = tb_config.get('crypto_pt', 0.04)
    sl = tb_config.get('crypto_sl', 0.02)
    # Adjust for TradFi if needed (lazy check: assume Crypto mostly for now or use config)
    # The caller should pass specific tb_config for the asset class ideally.
    # We will assume Crypto defaults are passed in tb_config
    
    position = None
    entry_price = 0
    entry_time = None
    tp_price = 0
    sl_price = 0
    
    # Limit to last N days for the REPORT
    start_dt = datetime.now() - timedelta(days=DAYS)
    df_sim = df[df.index >= start_dt]
    
    for idx, row in df_sim.iterrows():
        close = row['close']
        high = row['high']
        low = row['low']
        sig = row.get('signal_type', 'NONE')
        prob = row.get('prob', 0.0)
        
        # EXIT
        if position:
            exit_hit = False
            pnl = 0
            reason = ""
            
            if position == 'LONG':
                if low <= sl_price:
                    exit_hit = True
                    pnl = (sl_price - entry_price)/entry_price
                    reason = "SL"
                elif high >= tp_price:
                    exit_hit = True
                    pnl = (tp_price - entry_price)/entry_price
                    reason = "TP"
                # Trend Exit
                # elif close < row['cloud_bottom']: ... (Optional, sticking to TB for std reporting)
                
            elif position == 'SHORT':
                if high >= sl_price:
                    exit_hit = True
                    pnl = (entry_price - sl_price)/entry_price
                    reason = "SL"
                elif low <= tp_price:
                    exit_hit = True
                    pnl = (entry_price - tp_price)/entry_price
                    reason = "TP"

            if exit_hit:
                duration = idx - entry_time
                trades.append({
                    'Entry': entry_time,
                    'Exit': idx,
                    'Duration': duration,
                    'Type': position,
                    'PnL': pnl,
                    'Reason': reason,
                    'Conf': df.loc[entry_time, 'prob']
                })
                position = None
                
        # ENTRY
        if position is None and prob >= threshold:
            if 'LONG' in sig or 'SCALP_LONG' in sig:
                position = 'LONG'
                entry_price = close
                entry_time = idx
                tp_price = entry_price * (1 + pt)
                sl_price = entry_price * (1 - sl)
            elif 'SHORT' in sig or 'SCALP_SHORT' in sig:
                position = 'SHORT'
                entry_price = close
                entry_time = idx
                tp_price = entry_price * (1 - pt)
                sl_price = entry_price * (1 + sl)
                
    return trades

def generate_markdown(all_trades, grouped_stats):
    """Creates the text report."""
    md = f"# 🧙‍♂️ WizardWave Model Performance Report (180 Days)\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    # Summary
    total_trades = len(all_trades)
    if total_trades == 0:
        return md + "**No trades found in the last 180 days.**"
        
    df_t = pd.DataFrame(all_trades)
    win_rate = len(df_t[df_t['PnL'] > 0]) / total_trades * 100
    total_pnl = df_t['PnL'].sum() * 100
    avg_pnl = df_t['PnL'].mean() * 100
    best_trade = df_t['PnL'].max() * 100
    worst_trade = df_t['PnL'].min() * 100
    
    # Hold Times
    winners = df_t[df_t['PnL'] > 0]
    losers = df_t[df_t['PnL'] <= 0]
    avg_hold_w = winners['Duration'].mean() if not winners.empty else timedelta(0)
    avg_hold_l = losers['Duration'].mean() if not losers.empty else timedelta(0)
    
    # Confidence
    conf_w = winners['Conf'].mean() if not winners.empty else 0
    conf_l = losers['Conf'].mean() if not losers.empty else 0
    
    md += "## 📊 Executive Summary\n"
    md += f"- **Net PnL**: `{total_pnl:+.2f}%`\n"
    md += f"- **Win Rate**: `{win_rate:.1f}%` ({len(winners)}/{total_trades})\n"
    md += f"- **Avg Trade**: `{avg_pnl:+.2f}%`\n"
    md += f"- **Best Trade**: `{best_trade:+.2f}%`\n"
    md += f"- **Worst Trade**: `{worst_trade:+.2f}%`\n\n"
    
    md += "## ⏳ Duration & Confidence Insights\n"
    md += "| Metric | Winners 🟢 | Losers 🔴 |\n"
    md += "| :--- | :--- | :--- |\n"
    md += f"| **Avg Hold Time** | {str(avg_hold_w).split('.')[0]} | {str(avg_hold_l).split('.')[0]} |\n"
    md += f"| **Avg ML Confidence** | {conf_w:.1%} | {conf_l:.1%} |\n\n"
    
    md += "## 📂 Asset Breakdown\n"
    md += "| Asset | TF | Trades | Win Rate | Net PnL |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    
    for key, stats in grouped_stats.items():
        sym, tf = key
        md += f"| **{sym}** | {tf} | {stats['Trades']} | {stats['WinRate']:.1f}% | `{stats['PnL']:+.2f}%` |\n"

    md += "\n## 📝 Latest 10 Trades\n"
    md += "| Asset | Date | Type | PnL | Conf | Reason |\n"
    md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    # Sort by Date
    all_trades.sort(key=lambda x: x['Entry'])
    
    for t in all_trades[-10:]:
        date_str = t['Entry'].strftime('%Y-%m-%d')
        pnl_str = f"`{t['PnL']*100:+.2f}%`"
        md += f"| {t['Asset']} | {date_str} | {t['Type']} | {pnl_str} | {t['Conf']:.2f} | {t['Reason']} |\n"
        
    return md

def main():
    print("Initializing Report Generation...")
    
    # Load Config & Models
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
        
    model_1d = joblib.load("model_1d.pkl")
    model_1h = joblib.load("model_1h.pkl")
    
    assets = config['assets']
    strategies = {
        '1d': WizardWaveStrategy(),
        '1h': WizardScalpStrategy()
    }
    
    all_trades = []
    grouped_stats = {}
    
    for symbol in assets:
        print(f"Scanning {symbol}...")
        
        # 1. 1D Scan
        df_d = fetch_data(symbol, '1d', DAYS + WARMUP_DAYS)
        if not df_d.empty:
            tb = config['models']['1d']['triple_barrier']
            t_d = simulate_trades(df_d, strategies['1d'], model_1d, tb, '1d')
            if t_d:
                df_td = pd.DataFrame(t_d)
                stats = {
                    'Trades': len(t_d),
                    'WinRate': len(df_td[df_td['PnL']>0])/len(t_d)*100,
                    'PnL': df_td['PnL'].sum()*100
                }
                grouped_stats[(symbol, '1D')] = stats
                for t in t_d: t['Asset'] = symbol; t['TF'] = '1D'
                all_trades.extend(t_d)
                
        # 2. 1H Scan
        # Skip if not crypto? Or limit to simplify report run time
        if "BTC" in symbol or "ETH" in symbol or "SOL" in symbol:
            df_h = fetch_data(symbol, '1h', 60) # 60 days max for 1h usually via yfinance w/ 1h interval? 
            # Actually yf allows 730d for 1h.
            df_h = fetch_data(symbol, '1h', 180 + WARMUP_DAYS) 
            
            if not df_h.empty:
                 tb = config['models']['1h']['triple_barrier']
                 t_h = simulate_trades(df_h, strategies['1h'], model_1h, tb, '1h')
                 if t_h:
                    df_th = pd.DataFrame(t_h)
                    stats = {
                        'Trades': len(t_h),
                        'WinRate': len(df_th[df_th['PnL']>0])/len(t_h)*100,
                        'PnL': df_th['PnL'].sum()*100
                    }
                    grouped_stats[(symbol, '1H')] = stats
                    for t in t_h: t['Asset'] = symbol; t['TF'] = '1H'
                    all_trades.extend(t_h)

    # Generate Output
    report = generate_markdown(all_trades, grouped_stats)
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\n[OK] Report Generated: {os.path.abspath(REPORT_FILE)}")
    
    # Print Summary to Console
    lines = report.split('\n')
    for line in lines[:20]: # First 20 lines
        print(line)

if __name__ == "__main__":
    main()

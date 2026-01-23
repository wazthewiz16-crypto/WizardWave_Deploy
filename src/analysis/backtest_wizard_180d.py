
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
import sys
import os
# Add project root to sys.path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.strategies.strategy import WizardWaveStrategy
from datetime import datetime, timedelta

def fetch_history_yf(symbol, period='1y', interval='1d'):
    """Fetch history using yfinance for backtesting."""
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            return df
        
        # Standardize
        df = df.reset_index()
        df.rename(columns={
            'Date': 'datetime', 
            'Datetime': 'datetime',
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Volume': 'volume'
        }, inplace=True)
        
        # TZ naive
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_convert(None)
            df.set_index('datetime', inplace=True)
            
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def run_backtest(df, strategy, asset_class, timeframe_label='1D'):
    """Run simulation on the dataframe."""
    df = strategy.apply(df)
    
    trades = []
    
    position = None
    entry_price = 0.0
    entry_time = None
    sl_price = 0.0
    tp_price = 0.0
    
    # --- PARAMS FROM APP ---
    # Defaults
    tp_pct = 0.04
    sl_pct = 0.02
    
    if timeframe_label == '1D':
        if asset_class == 'Crypto':
            tp_pct = 0.09
            sl_pct = 0.03
        elif asset_class == 'TradFi':
            tp_pct = 0.04
            sl_pct = 0.02
        elif asset_class == 'Forex':
            tp_pct = 0.03
            sl_pct = 0.015
    else: # 1H / LTF
        if asset_class == 'Crypto':
            tp_pct = 0.025
            sl_pct = 0.0125
        elif asset_class == 'TradFi':
            tp_pct = 0.005
            sl_pct = 0.0025
        elif asset_class == 'Forex':
            tp_pct = 0.003
            sl_pct = 0.0015
    
    for index, row in df.iterrows():
        close = row['close']
        cloud_top = row.get('cloud_top', 0)
        cloud_bottom = row.get('cloud_bottom', 0)
        signal = row.get('signal_type', 'NONE')
        
        # Exit Logic
        exit_trade = False
        pnl = 0.0
        exit_reason = ""
        
        if position == 'LONG':
            # 1. SL Hit
            if close <= sl_price:
                exit_trade = True
                pnl = (sl_price - entry_price) / entry_price
                exit_reason = "SL"
            # 2. TP Hit
            elif close >= tp_price:
                exit_trade = True
                pnl = (tp_price - entry_price) / entry_price
                exit_reason = "TP"
            # 3. Trend Broken (Cloud Cross)
            elif close < cloud_bottom:
                exit_trade = True
                pnl = (close - entry_price) / entry_price
                exit_reason = "Trend"
                
        elif position == 'SHORT':
            # 1. SL Hit
            if close >= sl_price:
                exit_trade = True
                pnl = (entry_price - sl_price) / entry_price
                exit_reason = "SL"
            # 2. TP Hit
            elif close <= tp_price:
                exit_trade = True
                pnl = (entry_price - tp_price) / entry_price
                exit_reason = "TP"
            # 3. Trend Broken
            elif close > cloud_top:
                exit_trade = True
                pnl = (entry_price - close) / entry_price
                exit_reason = "Trend"

        if exit_trade:
            trades.append({
                'Asset': 'Test',
                'Type': position,
                'Entry Time': entry_time,
                'Exit Time': index,
                'Entry Price': entry_price,
                'Exit Price': tp_price if exit_reason == "TP" else (sl_price if exit_reason == "SL" else close),
                'PnL': pnl,
                'Reason': exit_reason
            })
            position = None
            
        # Entry Logic (If no position)
        if position is None:
            if 'LONG' in signal:
                position = 'LONG'
                entry_price = close
                entry_time = index
                sl_price = entry_price * (1 - sl_pct)
                tp_price = entry_price * (1 + tp_pct)
            elif 'SHORT' in signal:
                position = 'SHORT'
                entry_price = close
                entry_time = index
                sl_price = entry_price * (1 + sl_pct)
                tp_price = entry_price * (1 - tp_pct)
                
        # Reversal Logic (Flip)
        elif position == 'LONG' and 'SHORT' in signal:
            # Close Long
            trades.append({
                'Asset': 'Test',
                'Type': 'LONG',
                'Entry Time': entry_time,
                'Exit Time': index,
                'Entry Price': entry_price,
                'Exit Price': close,
                'PnL': (close - entry_price) / entry_price,
                'Reason': 'Flip'
            })
            # Open Short
            position = 'SHORT'
            entry_price = close
            entry_time = index
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)
        
        elif position == 'SHORT' and 'LONG' in signal:
            # Close Short
            trades.append({
                'Asset': 'Test',
                'Type': 'SHORT',
                'Entry Time': entry_time,
                'Exit Time': index,
                'Entry Price': entry_price,
                'Exit Price': close,
                'PnL': (entry_price - close) / entry_price,
                'Reason': 'Flip'
            })
            # Open Long
            position = 'LONG'
            entry_price = close
            entry_time = index
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
            
    return trades

def print_stats(trades, symbol, tf):
    if not trades:
        print(f"No trades for {symbol} {tf}")
        return

    df_t = pd.DataFrame(trades)
    total_trades = len(df_t)
    wins = len(df_t[df_t['PnL'] > 0])
    win_rate = wins / total_trades * 100
    total_pnl = df_t['PnL'].sum() * 100
    avg_pnl = df_t['PnL'].mean() * 100
    
    print(f"--- {symbol} ({tf}) ---")
    print(f"Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: {total_pnl:.2f}%")
    print(f"Avg Trade: {avg_pnl:.2f}%")
    print("-" * 20)

if __name__ == "__main__":
    assets = [
        # Crypto
        ('BTC-USD', 'Crypto'),
        ('ETH-USD', 'Crypto'),
        # TradFi
        ('^GSPC', 'TradFi'), # SPX
        ('NVDA', 'TradFi'),
        # Forex
        ('EURUSD=X', 'Forex')
    ]
    
    print("BACKTESTING WIZARD WAVE (Last 180 Days) - WITH APP PARAMS...")
    
    strat = WizardWaveStrategy()
    
    for symbol, atype in assets:
        # Test 1D
        df_d = fetch_history_yf(symbol, period='6mo', interval='1d')
        if not df_d.empty:
            trades_d = run_backtest(df_d, strat, atype, '1D')
            print_stats(trades_d, symbol, '1D')
            
        # Test 1H
        df_h = fetch_history_yf(symbol, period='6mo', interval='1h')
        if not df_h.empty:
            trades_h = run_backtest(df_h, strat, atype, '1H')
            print_stats(trades_h, symbol, '1H')


import pandas as pd
import yfinance as yf
from strategy import WizardWaveStrategy
import pandas_ta as ta

symbol = "BTC-USD"
print(f"Checking signals for {symbol} 1D (Improved Logic)...")

# Fetch
ticker = yf.Ticker(symbol)
df = ticker.history(period="1y", interval="1d")
df = df.reset_index()
df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
df.set_index('datetime', inplace=True)

# Apply
strat = WizardWaveStrategy()
df_res = strat.apply(df)

# Check specific columns
print("\n--- Filter Checks (Last 5 samples) ---")
cols = ['close', 'htf_trend', 'kvo', 'kvo_sig', 'trend_flip_bull', 'trend_flip_bear', 'signal_type']
print(df_res[cols].tail(5))

# Count signals
sig_counts = df_res['signal_type'].value_counts()
print("\n--- Signal Counts (Last Year) ---")
print(sig_counts)

# Check if confirmations are ever True
bull_trends = df_res['trend_flip_bull'].sum()
bear_trends = df_res['trend_flip_bear'].sum()
print(f"\nTotal Bull Flips (Filtered): {bull_trends}")
print(f"Total Bear Flips (Filtered): {bear_trends}")

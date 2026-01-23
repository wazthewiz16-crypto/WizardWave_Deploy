
import pandas as pd
import yfinance as yf
from strategy_scalp import WizardScalpStrategy
import pandas_ta as ta

symbol = "BTC-USD"
print(f"Checking signals for {symbol} 1h...")

# Fetch
ticker = yf.Ticker(symbol)
df = ticker.history(period="1mo", interval="1h")
df = df.reset_index()
df.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
df.set_index('datetime', inplace=True)

# Apply
strat = WizardScalpStrategy(lookback=8)
df_res = strat.apply(df)

# Print some rows where filters are checked
print("\n--- Filter Values (Last 10 bars) ---")
cols = ['close', 'cloud_top', 'cloud_bottom', 'rvol', 'adx', 'stoch_k', 'ema_trend', 'signal_type']
print(df_res[cols].tail(10))

# Count signals
sig_counts = df_res['signal_type'].value_counts()
print("\n--- Signal Counts (Last Month) ---")
print(sig_counts)

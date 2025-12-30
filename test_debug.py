
import pandas as pd
from data_fetcher import fetch_data
import os
import joblib
import json

def test_pipeline():
    print("Testing Data Fetcher and Model Loading...")
    
    # 1. Load Config
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
    print("Config loaded.")

    # 2. Test Model Loading
    models = {}
    keys = ["15m", "1h", "4h", "12h", "1d", "4d"]
    for k in keys:
        try:
            m = joblib.load(f"model_{k}.pkl")
            print(f"Loaded model_{k}.pkl: OK")
            models[k] = m
        except Exception as e:
            print(f"Failed to load model_{k}.pkl: {e}")

    # 3. Test Data Fetch (Crypto)
    symbol = "BTC/USDT"
    print(f"\nFetching {symbol} for 15m...")
    df = fetch_data(symbol, "crypto", "15m", limit=100)
    print(f"Data shape: {df.shape}")
    if not df.empty:
        print(df.tail())
    else:
        print("Data is empty!")

    # 4. Test Data Fetch (Trad)
    symbol_trad = "EUR=X"
    print(f"\nFetching {symbol_trad} for 1h...")
    df2 = fetch_data(symbol_trad, "forex", "1h", limit=100)
    print(f"Data shape: {df2.shape}")
    if not df2.empty:
        print(df2.tail())
    else:
        print("Data is empty!")

if __name__ == "__main__":
    test_pipeline()

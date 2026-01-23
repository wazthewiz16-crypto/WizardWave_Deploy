from data_fetcher import fetch_data
from strategy_ichimoku import IchimokuStrategy
import pandas as pd
import traceback

# Mimic app.py logic
TRADFI_CFG = {"tenkan": 20, "kijun": 60, "span_b": 120, "displacement": 30}
CRYPTO_CFG = {"tenkan": 7, "kijun": 21, "span_b": 42, "displacement": 21}

def process_asset(asset):
    try:
        print(f"Processing {asset['name']} ({asset['type']})...")
        is_crypto = (asset['type'] == 'crypto')
        cfg = CRYPTO_CFG if is_crypto else TRADFI_CFG
        
        target_tfs = ["1d"]
        if not is_crypto:
            target_tfs = ["4h", "1d"]
            
        ichi = IchimokuStrategy(**cfg)
        
        total_sig = 0
        
        for tf in target_tfs:
            print(f"  Fetching {tf}...")
            df = fetch_data(asset['symbol'], asset['type'], tf, 500)
            
            if df is None:
                print("    DF is None")
                continue
            if df.empty:
                print("    DF is Empty")
                continue
                
            print(f"    Fetched {len(df)} rows.")
            if len(df) < 160:
                print("    Not enough rows (<160)")
                continue
            
            df = ichi.apply_strategy(df, tf)
            signals = df[df['signal_type'].notna()]
            print(f"    Found {len(signals)} signals.")
            
            if not signals.empty:
                print("    Latest signal:", signals.index[-1], signals.iloc[-1]['signal_type'])
                total_sig += len(signals)
                
        return total_sig
        
    except Exception:
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    assets = [
        {"symbol": "BTC/USDT", "type": "crypto", "name": "Bitcoin"},
        {"symbol": "EURUSD=X", "type": "trad", "name": "EURUSD"}
    ]
    
    for a in assets:
        process_asset(a)

import json
import os
import sys
import time
from datetime import datetime
from data_fetcher import fetch_data
if os.path.exists("pipeline.py"):
    from pipeline import get_asset_type
else:
    # Fallback if pipeline.py import fails
    def get_asset_type(symbol):
        crypto_kw = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'BNB', 'LINK', 'ARB', 'AVAX', 'ADA', 'USDT']
        if any(k in symbol.upper() for k in crypto_kw): return 'crypto'
        if '=X' in symbol: return 'forex'
        return 'trad'

def update_all():
    print(f"--- Daily Data Sync Started ({datetime.now()}) ---")
    
    # Load Assets
    try:
        with open('strategy_config.json', 'r') as f:
            config = json.load(f)
        assets = config.get('assets', [])
    except:
        assets = []
        
    # Default Essentials
    extras = ["DX-Y.NYB", "BTC/USDT", "BTC-USD", "GC=F"]
    assets = list(set(assets + extras))
    
    timeframes = ['15m', '1h', '4h', '1d']
    
    for symbol in assets:
        # Determine Type
        a_type = get_asset_type(symbol)
        if '-' in symbol or '^' in symbol or '=' in symbol: 
            if a_type != 'forex': a_type = 'trad'
            
        for tf in timeframes:
            try:
                # Cache Policy:
                # Since this script runs ONCE daily, we want to ensure we have the LATEST candle.
                # data_fetcher uses TTL. 
                # 1d TTL is typically 12h. 4h is 2h.
                # To guarantee "New Day" freshness, we check file age.
                
                clean_sym = symbol.replace("/", "_").replace("^", "").replace("=", "")
                cache_file = f"market_data_cache/{clean_sym}_{tf}.csv"
                
                if os.path.exists(cache_file):
                    # If file is older than 4 hours, treat as stale for daily sync purposes
                    age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
                    if age_hours > 4:
                        print(f"  [Refetch] {symbol} {tf} (Age: {age_hours:.1f}h)")
                        os.remove(cache_file)
                    else:
                        print(f"  [Fresh] {symbol} {tf}")
                        continue # Skip fetch if fresh
                else:
                    print(f"  [New] {symbol} {tf}")

                # Fetch (Will write to cache)
                # Use large limit to ensure history
                limit = 500 if tf in ['1d', '4d'] else 1000
                fetch_data(symbol, a_type, tf, limit=limit)
                
                # Tiny sleep to respect rate limits if any
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  [Error] {symbol} {tf}: {e}")

    print(f"--- Sync Complete ({datetime.now()}) ---")

if __name__ == "__main__":
    update_all()

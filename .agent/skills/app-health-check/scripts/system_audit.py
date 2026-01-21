import ccxt
import pandas as pd
import os
import time
import requests
from datetime import datetime

def check_api_latency():
    print("\n--- 1. API Latency Test ---")
    try:
        # Try Binance US first for local US users
        try:
            exchange = ccxt.binanceus()
            exchange.load_markets() 
        except:
             exchange = ccxt.binance()
             
        start = time.time()
        exchange.fetch_ticker('BTC/USDT')
        latency = (time.time() - start) * 1000
        print(f"Binance fetch_ticker latency: {latency:.2f} ms")
        if latency > 1000:
            print("WARNING: High Latency (>1000ms)")
            return False
        print("OK: API Connection Stable")
        return True
    except Exception as e:
        print(f"Error: API Error: {e}")
        return False

def check_data_integrity():
    print("\n--- 2. Data Integrity Check ---")
    cache_dir = "market_data_cache"
    if not os.path.exists(cache_dir):
        print("Warning: Cache directory not found.")
        return True # Not critical if it lazy loads
    
    files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
    if not files:
        print("Warning: No cached data found.")
        return True
    
    issues = 0
    now = time.time()
    
    for f in files[:5]: # Check sample
        path = os.path.join(cache_dir, f)
        try:
            # Freshness
            mtime = os.path.getmtime(path)
            age_h = (now - mtime) / 3600
            
            df = pd.read_csv(path)
            if df.empty:
                print(f"Error {f}: Empty file")
                issues += 1
                continue
            
            # NaN Check
            if df.isnull().values.any():
                 print(f"Warning {f}: Contains NaN values")
                 issues += 1
            
            print(f"OK {f}: Valid (Age: {age_h:.1f}h)")
            
        except Exception as e:
            print(f"Error {f}: Read Error {e}")
            issues += 1
            
    if issues > 0:
        print(f"Warning: {issues} data integrity issues detected.")
        return False
    return True

def check_logs():
    print("\n--- 3. Log Audit ---")
    log_file = "app.log"
    if not os.path.exists(log_file):
        print("Info: No app.log found (Clean run).")
        return True
        
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        recent = lines[-100:] if len(lines) > 100 else lines
        errors = [l for l in recent if "ERROR" in l or "Exception" in l or "Traceback" in l]
        
        if errors:
            print(f"Found {len(errors)} recent errors in app.log:")
            for e in errors[:3]:
                print(f"  - {e.strip()}")
            return False
        else:
            print("OK: Log is clean (last 100 lines).")
            return True
            
    except Exception as e:
        print(f"Error reading log: {e}")
        return False

def check_remote_url(url="https://wazthewiz.streamlit.app/"):
    print(f"\n--- 4. Remote URL Check ({url}) ---")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"OK: Website is UP (Status 200)")
            return True
        else:
            print(f"Error: Website returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: Request failed: {e}")
        return False

def run_audit():
    print("--- Starting System Doctor Audit ---")
    
    api_ok = check_api_latency()
    data_ok = check_data_integrity()
    logs_ok = check_logs()
    web_ok = check_remote_url()
    
    print("\n" + "="*30)
    if api_ok and data_ok and logs_ok and web_ok:
        print("Success: SYSTEM HEALTHY")
    else:
        print("Failure: MAINTENANCE REQUIRED")
    print("="*30)

if __name__ == "__main__":
    run_audit()

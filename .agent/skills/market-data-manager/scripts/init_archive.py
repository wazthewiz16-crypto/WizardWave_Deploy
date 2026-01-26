import os
import sys
import pandas as pd
import ccxt
import yfinance as yf
import logging
from datetime import datetime, timedelta, timezone

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataSteward")

ARCHIVE_DIR = os.path.join("data", "parquet_archive")
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def ensure_dirs():
    os.makedirs(os.path.join(ARCHIVE_DIR, "crypto"), exist_ok=True)
    os.makedirs(os.path.join(ARCHIVE_DIR, "tradfi"), exist_ok=True)

def fetch_crypto_history(symbol, timeframe='1m', days=1500): # ~4 years
    """Fetches max available history from Binance"""
    try:
        clean_symbol = symbol.replace(".P", "") # CCXT uses BTC/USDT not BTC/USDT.P usually
        exchange = ccxt.binanceus() # Or binance() if international
        
        # CCXT loop for pagination (fetch_ohlcv limits to 1000 usually)
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
        
        all_ohlcv = []
        while since < exchange.milliseconds():
            ohlcv = exchange.fetch_ohlcv(clean_symbol, timeframe, since, limit=1000)
            if not ohlcv: break
            since = ohlcv[-1][0] + 1 # Move cursor
            all_ohlcv.extend(ohlcv)
            logger.info(f"  Fetched {len(ohlcv)} candles for {symbol}...")
            # Ratelimit sleep handled roughly by CCXT or add time.sleep here
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()

def fetch_tradfi_history(symbol, days=730): 
    """Fetches max available hourly history from yfinance (limit ~730 days)"""
    try:
        ticker = yf.Ticker(symbol)
        # Fetch 1h data
        df = ticker.history(period="730d", interval="1h")
        
        if df.empty: return df
        
        # Normalize
        df.reset_index(inplace=True)
        # yfinance returns 'Datetime' for intraday, 'Date' for daily. Handle both.
        cols = {'Date': 'timestamp', 'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
        df.rename(columns=cols, inplace=True)
        
        # Ensure timestamp is TZ aware
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            if df['timestamp'].dt.tz is None:
                 df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                 df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
             
        df.set_index('timestamp', inplace=True)
        # Filter existing columns
        out_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        return df[out_cols]
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()

def get_save_path(symbol, timeframe, asset_type):
    folder = "crypto" if asset_type == 'crypto' else "tradfi"
    safe_sym = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(ARCHIVE_DIR, folder, f"{safe_sym}_{timeframe}.parquet")

def get_asset_type(symbol):
    # Heuristic matching from pipeline.py logic
    crypto_kw = ['BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'BNB', 'LINK', 'ARB', 'AVAX', 'ADA', 'USDT']
    if any(k in symbol.upper() for k in crypto_kw):
        return 'crypto'
    return 'trad'

def init_archive():
    ensure_dirs()
    
    # Load your config
    import json
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
        
    assets = config.get('assets', [])
    
    for asset in assets:
        # Handle both string (current) and dict (future proof) formats
        if isinstance(asset, str):
            sym = asset
            atype = get_asset_type(sym)
        else:
            sym = asset['symbol']
            atype = asset.get('type', get_asset_type(sym))
        
        logger.info(f"Archiving {sym} ({atype})...")
        
        if atype == 'crypto':
            df = fetch_crypto_history(sym, '1m')
            path = get_save_path(sym, '1m', 'crypto')
        else:
            df = fetch_tradfi_history(sym) # Fetches 1h
            path = get_save_path(sym, '1h', 'tradfi')
            
        if not df.empty:
            df.to_parquet(path)
            logger.info(f"Saved {path} ({len(df)} rows)")
        else:
            logger.warning(f"No data for {sym}")

if __name__ == "__main__":
    init_archive()

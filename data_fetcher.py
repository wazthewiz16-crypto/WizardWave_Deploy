import ccxt
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import streamlit as st

def fetch_data(symbol: str, asset_type: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
    """
    Unified data fetcher for Crypto and Traditional assets.
    
    Args:
        symbol (str): The ticker symbol (e.g., 'BTC/USD', 'AAPL').
        asset_type (str): 'crypto' or 'trad'.
        timeframe (str): Timeframe for candles ('1h', '4h', '1d', '4d').
                         Note: '4h' and '4d' may be resampled from base data.
        limit (int): Number of candles to fetch.
        
    Returns:
        pd.DataFrame: Standardized OHLCV DataFrame.
    """
    
    # --- GENERIC LOCAL CACHE LOGIC ---
    import os
    import time
    
    CACHE_DIR = "market_data_cache"
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    # TTL Configuration (in seconds)
    ttl_map = {
        '1m': 60,
        '5m': 150,
        '15m': 300,    # 5 minutes
        '30m': 900,
        '1h': 1800,    # 30 minutes
        '4h': 7200,    # 2 hours
        '12h': 21600,  # 6 hours
        '1d': 43200,   # 12 hours (since day closes once)
        '4d': 86400    # 24 hours
    }
    
    # Use timeframe-based TTL or default to 1 hour
    current_ttl = ttl_map.get(timeframe, 3600)
    
    clean_symbol = symbol.replace("/", "_").replace("^", "").replace("=", "")
    cache_file = os.path.join(CACHE_DIR, f"{clean_symbol}_{timeframe}.csv")
    
    # Check if file exists and is fresh
    cached_df = pd.DataFrame()
    
    # Try load cache for fallback
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, index_col='datetime', parse_dates=True)
            if not isinstance(cached_df.index, pd.DatetimeIndex):
                cached_df.index = pd.to_datetime(cached_df.index)
        except: pass

    # Check validity for "Fresh Return"
    if not cached_df.empty:
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < current_ttl: 
            # TTL Valid, check Data Stale
             last_ts = cached_df.index[-1]
             time_since = (pd.Timestamp.now() - last_ts).total_seconds()
             max_lag_s = 300000 if timeframe != '4d' else 600000
             
             if time_since < max_lag_s:
                 return cached_df
             else:
                 print(f"Cache for {symbol} stale (Last: {last_ts}). Refetching...")

    base_timeframe = timeframe
    resample_rule = None
    
    if timeframe == '4h':
        base_timeframe = '1h'
        resample_rule = '4h'
        limit = limit * 4 # Fetch more to support resampling
    elif timeframe == '4d':
        base_timeframe = '1d'
        resample_rule = '4d' # Pandas Offset Alias
        limit = limit * 4
    elif timeframe == '12h':
        base_timeframe = '1h'
        resample_rule = '12h'
        limit = limit * 12
        
    df = pd.DataFrame()
    if asset_type == 'crypto':
        df = _fetch_crypto(symbol, base_timeframe, limit)
    elif asset_type == 'trad' or asset_type == 'forex':
        df = _fetch_trad(symbol, base_timeframe, limit)
    else:
        raise ValueError("Invalid asset_type. Must be 'crypto' or 'trad'.")

    # FALLBACK: If fetch failed, use stale cache if available
    if df.empty and not cached_df.empty:
         print(f"Fetch failed for {symbol}, using stale cache.")
         return cached_df

    # Resample if needed
    # Resample if needed
    final_df = df
    if not df.empty and resample_rule:
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        # Resample and drop incomplete bins
        final_df = df.resample(resample_rule).agg(agg_dict).dropna()
        
    # --- SAVE GENERIC CACHE ---
    if not final_df.empty:
        try:
            # CACHE_DIR and clean_symbol logic reused
            clean_symbol = symbol.replace("/", "_").replace("^", "").replace("=", "")
            cache_file = os.path.join(CACHE_DIR, f"{clean_symbol}_{timeframe}.csv")
            final_df.to_csv(cache_file)
        except Exception as e:
            print(f"Error saving cache for {symbol}: {e}")

    return final_df

def _fetch_crypto(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Attributes:
        symbol: e.g. 'BTC/USD'
    """
    try:
        # User requested binanceus
        exchange = ccxt.binanceus() 
        # Optional: load_markets is good practice but slow, fetch_ohlcv often works without it for major pairs
        # exchange.load_markets() 
        
        # CCXT uses 'since' for history, or just 'limit' for most recent
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Clean up
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df
    except Exception as e:
        print(f"Error fetching crypto data for {symbol}: {e}")
        return pd.DataFrame()

def _fetch_trad(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Attributes:
        symbol: e.g. 'QQQ', '^NDX' (Nasdaq 100 uses ^NDX usually, or futures NQ=F)
    """
    try:
        # yfinance logic
        # interval mapping if needed, but '1h', '1d' match common needs
        # yfinance period vs limit. We can use period='max' and tail(limit), or 'y' for year.
        # For recent 100 candles, period='1mo' might be enough for 1h, but let's be safe with '3mo' or max
        # A simpler way is to just fetch last N days.
        
        # If we need exact limit, we fetch more and slice.
        ticker = yf.Ticker(symbol)
        
        # Choose a period that likely covers 'limit' candles
        period_map = {
            '1m': '1d', '5m': '5d', '15m': '5d', '30m': '5d',
            '1h': '3mo', '1d': 'max' # Increased for 4h/4d robustness
        }
        period = period_map.get(timeframe, '1mo')
        
        df = ticker.history(period=period, interval=timeframe)
        
        if df.empty:
            return pd.DataFrame()
            
        # Standardize columns
        df.reset_index(inplace=True)
        # yfinance columns: Date/Datetime, Open, High, Low, Close, Volume, Dividends, Splits
        # Rename to lowercase
        df.rename(columns={
            'Date': 'datetime', 
            'Datetime': 'datetime',
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low', 
            'Close': 'close', 
            'Volume': 'volume'
        }, inplace=True)
        
        # Ensure datetime index
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            # Remove timezone if present to match crypto (usually UTC) or keep it aware.
            # Best is to convert to UTC.
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_convert(None) # naive UTC
            df.set_index('datetime', inplace=True)
        
        return df.tail(limit)[['open', 'high', 'low', 'close', 'volume']]
        
    except Exception as e:
        print(f"Error fetching trad data for {symbol}: {e}")
        return pd.DataFrame()

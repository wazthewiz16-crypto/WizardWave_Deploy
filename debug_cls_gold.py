import pandas as pd
import pandas_ta as ta
from data_fetcher import fetch_data
from strategy_cls import CLSRangeStrategy

def debug_gold_cls():
    symbol = 'GC=F'
    print(f"Debugging CLS Ranges for {symbol}...")
    
    # Fetch Daily Data (HTF)
    df_daily = fetch_data(symbol, asset_type='trad', timeframe='1d', limit=500)
    
    if df_daily.empty:
        print("No data found for GC=F")
        return

    # Instantiate Strategy to use its internal logic logic
    strat = CLSRangeStrategy()
    
    # Replicate HTF Logic
    htf = df_daily.copy()
    htf['atr'] = ta.atr(htf['high'], htf['low'], htf['close'], length=14)
    htf['body_size'] = abs(htf['close'] - htf['open'])
    htf['is_large'] = htf['body_size'] > (htf['atr'] * strat.atr_multiplier)
    
    htf['rolling_max'] = htf['high'].shift(1).rolling(window=strat.swing_window).max()
    htf['rolling_min'] = htf['low'].shift(1).rolling(window=strat.swing_window).min()
    
    htf['swept_low'] = htf['low'] < htf['rolling_min']
    htf['swept_high'] = htf['high'] > htf['rolling_max']
    
    htf['mid_point'] = (htf['high'] + htf['low']) / 2
    htf['close_above_mid'] = htf['close'] > htf['mid_point']
    
    is_cls_bull = htf['is_large'] & htf['swept_low'] & (htf['close_above_mid'] | (htf['close'] > htf['open']))
    is_cls_bear = htf['is_large'] & htf['swept_high'] & ((~htf['close_above_mid']) | (htf['close'] < htf['open']))
    
    htf['cls_type'] = np.where(is_cls_bull, 'BULL', np.where(is_cls_bear, 'BEAR', 'NONE'))
    
    # Filter to show only CLS candles
    cls_candles = htf[htf['cls_type'] != 'NONE'].copy()
    
    print("\n=== IDENTIFIED CLS CANDLES (Last 10) ===")
    cols = ['open', 'high', 'low', 'close', 'cls_type', 'atr', 'rolling_min', 'rolling_max']
    print(cls_candles[cols].tail(10))
    
    # Check specific range active on 2026-01-01
    # Forward fill ranges
    htf['range_high'] = np.where(is_cls_bull | is_cls_bear, htf['high'], np.nan)
    htf['range_low'] = np.where(is_cls_bull | is_cls_bear, htf['low'], np.nan)
    htf['range_high'] = htf['range_high'].ffill()
    htf['range_low'] = htf['range_low'].ffill()
    
    print("\n=== ACTIVE RANGE AROUND 2026-01-01 ===")
    subset = htf[(htf.index >= '2025-12-15') & (htf.index <= '2026-01-05')]
    print(subset[['high', 'low', 'close', 'range_high', 'range_low']])

if __name__ == "__main__":
    import numpy as np # Need numpy imported inside or globally
    debug_gold_cls()

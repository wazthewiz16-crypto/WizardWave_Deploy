from data_fetcher import fetch_data
from strategy_ichimoku import IchimokuStrategy
import pandas as pd

def debug():
    # Fetch Bitcoin 4H
    df = fetch_data("BTC/USDT", "crypto", "4h", 600)
    ichi = IchimokuStrategy()
    df = ichi.apply_strategy(df, "4h")
    
    print("Columns:", df.columns)
    print("Head:\n", df[['close', 'tenkan', 'kijun', 'span_a', 'span_b', 'signal_type']].tail(20))
    
    # Check counts
    print("TK Cross Bull Count:", ((df['tenkan'] > df['kijun']) & (df['tenkan'].shift(1) <= df['kijun'].shift(1))).sum())
    print("Signal Count:", df['signal_type'].value_counts())
    
    # Check why signals fail
    # Filter conditions
    cloud_top = df[['span_a', 'span_b']].max(axis=1)
    above_cloud = df['close'] > cloud_top
    
    past_high = df['high'].shift(30)
    chikou_bull = df['close'] > past_high
    
    tk_bull = (df['tenkan'] > df['kijun']) & (df['tenkan'].shift(1) <= df['kijun'].shift(1))
    
    print("Filtered Counts:")
    print("TK Bull Crosses:", tk_bull.sum())
    print("Above Cloud Condition:", above_cloud.sum())
    print("Chikou Bull Condition:", chikou_bull.sum())
    
    # Overlap
    combined = tk_bull & above_cloud & chikou_bull
    print("Combined Bull:", combined.sum())

if __name__ == "__main__":
    debug()

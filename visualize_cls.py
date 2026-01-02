import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from data_fetcher import fetch_data
from strategy_cls import CLSRangeStrategy

def plot_cls_gold():
    symbol = 'GC=F'
    print(f"Plotting CLS Structure for {symbol}...")
    
    # Fetch Daily Data (HTF)
    df_daily = fetch_data(symbol, asset_type='trad', timeframe='1d', limit=200)
    
    if df_daily.empty:
        print("No data found.")
        return

    strat = CLSRangeStrategy()
    
    # Replicate Logic
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
    
    htf['active_high'] = np.where(is_cls_bull | is_cls_bear, htf['high'], np.nan)
    htf['active_low'] = np.where(is_cls_bull | is_cls_bear, htf['low'], np.nan)
    
    htf['range_high'] = htf['active_high'].ffill()
    htf['range_low'] = htf['active_low'].ffill()
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Candles (Open/Close lines) - simplified
    for idx, row in htf.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([row.name, row.name], [row['low'], row['high']], color='black', linewidth=0.5)
        ax.plot([row.name, row.name], [row['open'], row['close']], color=color, linewidth=3)
    
    # Plot Ranges (Step lines)
    ax.plot(htf.index, htf['range_high'], color='blue', label='CLS Range High', linewidth=1.5)
    ax.plot(htf.index, htf['range_low'], color='purple', label='CLS Range Low', linewidth=1.5)
    
    # Mark CLS events
    cls_events = htf[is_cls_bull | is_cls_bear]
    ax.scatter(cls_events.index, cls_events['high'] * 1.01, color='orange', marker='v', s=50, label='CLS Signal')
    
    # Highlight specific date
    target_date = pd.Timestamp('2025-12-10')
    if target_date in htf.index:
         ax.annotate('Dec 10 CLS', xy=(target_date, htf.loc[target_date]['high']), xytext=(target_date, htf.loc[target_date]['high']+50),
             arrowprops=dict(facecolor='black', shrink=0.05))

    ax.set_title(f"Gold Daily CLS Ranges")
    ax.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('cls_debug_plot.png')
    print("Plot saved to cls_debug_plot.png")

if __name__ == "__main__":
    import numpy as np
    plot_cls_gold()

import pandas as pd
import numpy as np
import pandas_ta as ta

class CLSRangeStrategy:
    """
    CLS Range Strategy (Multi-Timeframe):
    1. HTF (e.g. Daily): Identify CLS Candle (Large, Sweeps Liquidity).
       - The Range is the High/Low of this HTF CLS Candle.
    2. LTF (e.g. 1H): Trade the Deviation & Reclaim.
       - Volume Confirmation: Requires the HTF sweep to have high volume.
    3. Targets:
       - Mid-Range (50% of HTF Range)
       - Full Range (Opposite side of HTF Range)
    """

    def __init__(self, swing_window=10, atr_multiplier=1.5, volume_multiplier=1.2):
        self.swing_window = swing_window
        self.atr_multiplier = atr_multiplier
        self.volume_multiplier = volume_multiplier

    def apply_mtf(self, df_htf: pd.DataFrame, df_ltf: pd.DataFrame) -> pd.DataFrame:
        """
        Applies strategy using HTF for Range Definition and LTF for Entry.
        """
        if df_htf.empty or df_ltf.empty:
            return df_ltf
        
        # --- 1. Process HTF (Define Ranges) ---
        htf = df_htf.copy()
        htf['atr'] = ta.atr(htf['high'], htf['low'], htf['close'], length=14)
        htf['body_size'] = abs(htf['close'] - htf['open'])
        htf['is_large'] = htf['body_size'] > (htf['atr'] * self.atr_multiplier)
        
        # Swing Points (Liquidity)
        htf['rolling_max'] = htf['high'].shift(1).rolling(window=self.swing_window).max()
        htf['rolling_min'] = htf['low'].shift(1).rolling(window=self.swing_window).min()
        
        htf['swept_low'] = htf['low'] < htf['rolling_min']
        htf['swept_high'] = htf['high'] > htf['rolling_max']
        
        # Candle Quality
        htf['mid_point'] = (htf['high'] + htf['low']) / 2
        htf['close_above_mid'] = htf['close'] > htf['mid_point']
        
        # Volume Confirmation
        htf['vol_sma'] = htf['volume'].rolling(20).mean()
        htf['high_vol'] = htf['volume'] > (htf['vol_sma'] * self.volume_multiplier)
        
        # Define CLS Candles
        # Bullish CLS: Large, Swept Low, Strong Close, High Volume
        is_cls_bull = htf['is_large'] & htf['swept_low'] & (htf['close_above_mid'] | (htf['close'] > htf['open'])) & htf['high_vol']
        # Bearish CLS: Large, Swept High, Weak Close, High Volume
        is_cls_bear = htf['is_large'] & htf['swept_high'] & ((~htf['close_above_mid']) | (htf['close'] < htf['open'])) & htf['high_vol']
        
        # Mark Ranges
        htf['cls_active_high'] = np.nan
        htf['cls_active_low'] = np.nan
        
        htf.loc[is_cls_bull, 'cls_active_high'] = htf['high']
        htf.loc[is_cls_bull, 'cls_active_low'] = htf['low']
        
        htf.loc[is_cls_bear, 'cls_active_high'] = htf['high']
        htf.loc[is_cls_bear, 'cls_active_low'] = htf['low']
        
        # Forward Fill Ranges (The "Active" Range)
        htf['range_high'] = htf['cls_active_high'].ffill()
        htf['range_low'] = htf['cls_active_low'].ffill()
        
        # Prepare for Merge
        # We need to map HTF daily values to LTF intraday timestamps.
        # Simplest way: Resample HTF to LTF or MergeAsOf.
        
        # Let's use merge_asof on time.
        # Ensure sorted
        htf = htf.sort_index()
        df_ltf = df_ltf.sort_index()
        
        # Extract only needed columns from HTF
        htf_levels = htf[['range_high', 'range_low']].copy()
        
        # Merge
        # 'backward' direction: For any LTF time, look for the LAST available HTF time.
        # Note: HTF data often has timestamp at 00:00. This is valid for the whole day "open", 
        # but typically High/Low are known at close? 
        # Ideally, we use YESTERDAY'S CLS to trade TODAY (or the completed candle).
        # So we should shift HTF levels by 1 period (1 Day) to assume we trade AFTER the CLS closes.
        
        htf_levels_shifted = htf_levels.shift(1) # We trade the range established by the COMPLETED candle
        
        merged = pd.merge_asof(
            df_ltf, 
            htf_levels_shifted, 
            left_index=True, 
            right_index=True, 
            direction='backward'
        )
        
        # --- 2. Process LTF (Entries) ---
        df = merged.copy()
        
        # Logic: Deviation & Reclaim
        # range_low and range_high come from HTF
        
        # Reclaim Long: Price dips below Range Low, then Closes back above Range Low
        # We also need a "Deviation" confirmation? 
        # Standard: Just the reclaim event is the trigger after a deviation existed.
        
        reclaim_long = (df['close'] > df['range_low']) & (df['close'].shift(1) < df['range_low'])
        reclaim_short = (df['close'] < df['range_high']) & (df['close'].shift(1) > df['range_high'])
        
        # Filter: Ensure Range Exists
        range_exists = df['range_high'].notna() & df['range_low'].notna()
        
        df['long_signal'] = reclaim_long & range_exists
        df['short_signal'] = reclaim_short & range_exists
        
        conditions = [df['long_signal'], df['short_signal']]
        choices = ["CLS_LONG", "CLS_SHORT"]
        
        df['signal_type'] = np.select(conditions, choices, default="NONE")
        
        # --- 3. Targets & SL ---
        df['mid_range'] = (df['range_high'] + df['range_low']) / 2
        
        df['target_price'] = np.where(df['signal_type'] == 'CLS_LONG', df['range_high'],
                                   np.where(df['signal_type'] == 'CLS_SHORT', df['range_low'], np.nan))
                                   
        df['target_mid'] = np.where(df['signal_type'] != 'NONE', df['mid_range'], np.nan)
        
        # Stop Loss: Recent Extreme on LTF (Swing Low/High of deviation)
        # 5-period LTF lookback is fine for precise entry
        df['recent_min'] = df['low'].rolling(12).min() # Increased lookback to capture full deviation better
        df['recent_max'] = df['high'].rolling(12).max()
        
        df['stop_loss'] = np.where(df['signal_type'] == 'CLS_LONG', df['recent_min'],
                                   np.where(df['signal_type'] == 'CLS_SHORT', df['recent_max'], np.nan))
                                   
        # Sanity Check SL (Ensure risk is not inverted)
        # e.g. Long SL must be < Entry
        df['stop_loss'] = np.where((df['signal_type'] == 'CLS_LONG') & (df['stop_loss'] >= df['close']),
                                   df['close'] * 0.99, # Fallback 1%
                                   df['stop_loss'])

        df['stop_loss'] = np.where((df['signal_type'] == 'CLS_SHORT') & (df['stop_loss'] <= df['close']),
                                   df['close'] * 1.01, # Fallback 1%
                                   df['stop_loss'])
                                   
        return df

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Legacy wrapper for single-timeframe data """
        return self.apply_mtf(df, df) # Treat same df as HTF and LTF (old behavior)

import pandas as pd
import numpy as np
import pandas_ta as ta

class MondayRangeStrategy:
    """
    Monday Range Strategy:
    1. Identify High/Low of the most recent Monday.
    2. Trading Logic (active Tue-Sun):
       - LONG: Price deviates below Monday Low, then closes back ABOVE Monday Low.
       - SHORT: Price deviates above Monday High, then closes back BELOW Monday High.
    3. Targets:
       - LONG Target: Monday High
       - SHORT Target: Monday Low
    """
    def __init__(self):
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        
        # Ensure DateTime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Calculate Trend Filter (200 EMA)
        # Note: Need sufficient history for accurate EMA
        df['ema_200'] = ta.ema(df['close'], length=200)

        # 1. Identify Monday Ranges
        # Resample to Daily to get Monday's Full High/Low
        daily = df.resample('D').agg({'high': 'max', 'low': 'min'})
        daily['weekday'] = daily.index.dayofweek # 0 is Monday
        
        # Filter only Mondays
        mondays = daily[daily['weekday'] == 0].copy()
        mondays['mon_high'] = mondays['high']
        mondays['mon_low'] = mondays['low']
        
        # Forward fill these Monday levels to the rest of the week (Tue-Sun)
        # Reindex back to full daily to propagate
        daily_mapped = daily.join(mondays[['mon_high', 'mon_low']])
        daily_mapped[['mon_high', 'mon_low']] = daily_mapped[['mon_high', 'mon_low']].ffill()
        
        # Merge back to Intraday DF
        # We join on the 'date' component
        # Note: If df is 15m bars, we map each bar's date to the daily table
        
        # Create a temp column for merging
        df['temp_date'] = df.index.date
        daily_mapped['temp_date'] = daily_mapped.index.date
        
        # Use merge_asof or simple map? Map is safer for 1-to-many
        # Dict lookup mapping
        h_map = daily_mapped.set_index('temp_date')['mon_high'].to_dict()
        l_map = daily_mapped.set_index('temp_date')['mon_low'].to_dict()
        
        df['mon_high'] = df['temp_date'].map(h_map)
        df['mon_low'] = df['temp_date'].map(l_map)
        
        df.drop(columns=['temp_date'], inplace=True)
        
        # 2. Logic: Deviation & Close Back
        # Signal Generation (Exclude Mondays from *taking* trades based on current range, 
        # though typically we wait for Mon to close, so signals valid Tue onwards)
        
        df['day_of_week'] = df.index.dayofweek
        
        # --- Logic Steps ---
        
        # 1. Identify Deviation Extremes (Pre-calculation for SL)
        # Longs: Deviation Low (Lowest Low while < Mon Low)
        is_below = df['close'] < df['mon_low']
        below_grp = (is_below != is_below.shift()).cumsum()
        low_min = df.groupby(below_grp)['low'].transform('min')
        df['dev_low'] = np.where(is_below, low_min, np.nan)
        
        # Shorts: Deviation High (Highest High while > Mon High)
        is_above = df['close'] > df['mon_high']
        above_grp = (is_above != is_above.shift()).cumsum()
        high_max = df.groupby(above_grp)['high'].transform('max')
        df['dev_high'] = np.where(is_above, high_max, np.nan)

        # 2. Identify Reclaim Events
        # Long Reclaim: Prev Close < Mon Low, Curr Close > Mon Low
        reclaim_long = (df['close'] > df['mon_low']) & (df['close'].shift(1) < df['mon_low'])
        # Short Reclaim: Prev Close > Mon High, Curr Close < Mon High
        reclaim_short = (df['close'] < df['mon_high']) & (df['close'].shift(1) > df['mon_high'])
        
        # 3. Setup Window (Retest must happen within 12 bars of reclaim)
        # We forward fill the Reclaim Event and the Deviation Extreme associated with it
        
        # We need the dev_low from the *moment* of reclaim (shift 1)
        df['setup_sl_long'] = np.where(reclaim_long, df['dev_low'].shift(1), np.nan)
        df['setup_sl_short'] = np.where(reclaim_short, df['dev_high'].shift(1), np.nan)
        
        # Forward fill active setup for 12 bars? 
        # Pandas ffill(limit=12) works on NaNs.
        df['active_long_setup'] = df['setup_sl_long'].ffill(limit=12)
        df['active_short_setup'] = df['setup_sl_short'].ffill(limit=12)
        
        # 4. Trigger Retest Entry
        # Long: Active Setup, Low touches Mon Low (within 0.3%?), Close holds (>= Mon Low * 0.998?)
        # User said "retest... not just breakout".
        # Let's say Low <= Mon Low * 1.002 (came close) AND Close > Mon Low (held)
        # AND it is NOT the reclaim candle itself (shift 1 reclaim must be false? or just ensure we are in window)
        
        # Valid Day (Tue-Sun)
        valid_day = df['day_of_week'] != 0
        
        # Valid Session (High Volume: 07:00 to 21:00 UTC - London & NY)
        # Assuming df index is UTC (which it is from data_fetcher)
        valid_time = (df.index.hour >= 7) & (df.index.hour <= 21)
        
        # Trend Filter (200 EMA)
        trend_long = df['close'] > df['ema_200']
        trend_short = df['close'] < df['ema_200']
        
        # Long Entry
        # Condition: Have active setup (not NaN)
        has_long_setup = df['active_long_setup'].notna()
        # Condition: Retest (Low dipped near Mon Low). 0.2% buffer
        retest_dip_long = df['low'] <= (df['mon_low'] * 1.002)
        # Condition: Held (Close is above Mon Low - 0.1% tolerance?)
        held_long = df['close'] >= df['mon_low']
        
        long_signal = has_long_setup & retest_dip_long & held_long & valid_day & valid_time & trend_long & (~reclaim_long)
        
        # Short Entry
        has_short_setup = df['active_short_setup'].notna()
        retest_peak_short = df['high'] >= (df['mon_high'] * 0.998)
        held_short = df['close'] <= df['mon_high']
        
        short_signal = has_short_setup & retest_peak_short & held_short & valid_day & valid_time & trend_short & (~reclaim_short)

        # 5. Assign Targets & SLs
        
        # Signal Type
        conditions = [long_signal, short_signal]
        choices = ["MONDAY_LONG", "MONDAY_SHORT"]
        df['signal_type'] = np.select(conditions, choices, default="NONE")
        
        # Targets
        df['target_price'] = np.where(df['signal_type'] == 'MONDAY_LONG', df['mon_high'], 
                                   np.where(df['signal_type'] == 'MONDAY_SHORT', df['mon_low'], np.nan))
                                   
        # Mid-Range Target (50% of range)
        mid_price = (df['mon_high'] + df['mon_low']) / 2
        df['target_mid'] = np.where(df['signal_type'] != 'NONE', mid_price, np.nan)
                                   
        # Stop Loss: Middle of Extreme & Level
        # SL = (Dev + Level) / 2
        
        # Create temp columns for SL calc
        # active_long_setup holds the Dev Low
        df['rough_sl_long'] = (df['active_long_setup'] + df['mon_low']) / 2
        df['rough_sl_short'] = (df['active_short_setup'] + df['mon_high']) / 2
        
        df['stop_loss'] = np.where(df['signal_type'] == 'MONDAY_LONG', df['rough_sl_long'], 
                                   np.where(df['signal_type'] == 'MONDAY_SHORT', df['rough_sl_short'], np.nan))
                                   
        # 6. R:R Filter
        entry = df['close']
        risk = abs(entry - df['stop_loss'])
        reward = abs(df['target_price'] - entry)
        
        df['rr_ratio'] = np.where((risk > 0) & (df['signal_type'] != 'NONE'), reward / risk, 0.0)
        
        valid_rr = (df['rr_ratio'] >= 2.0) & (df['rr_ratio'] <= 5.0)
        df['signal_type'] = np.where((df['signal_type'] != 'NONE') & (~valid_rr), "NONE", df['signal_type'])
        
        mask_invalid = df['signal_type'] == 'NONE'
        df.loc[mask_invalid, ['target_price', 'stop_loss', 'rr_ratio']] = np.nan
        
        return df

    def get_active_trade(self, df: pd.DataFrame) -> dict:
        """
        Check last candle for signal.
        """
        if df.empty: return None
        if 'signal_type' not in df.columns:
            df = self.apply(df)
            
        last = df.iloc[-1]
        if 'MONDAY' in last['signal_type']:
             # Simple simulated return
             return {
                 "Position": "LONG" if "LONG" in last['signal_type'] else "SHORT",
                 "Type": last['signal_type'],
                 "Entry Time": last.name,
                 "Entry Price": last['close'],
                 "Target": last['target_price'],
                 "PnL (%)": 0.0 # Just entered
             }
        return None

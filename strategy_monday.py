import pandas as pd
import numpy as np

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
        
        # Logic:
        # We need to detect the 'cross back'.
        # Previous Intraday candle was 'deviated' (or just low < mon_low), Current close > mon_low
        # Re-entry triggers:
        # LONG: Low(any recent) < MonLow, but NOW Close > MonLow.
        # Specifically "Close back inside". This implies the *break* happened recently.
        # Simplest Trigger: prev_close < mon_low AND curr_close > mon_low ?
        # OR prev_low < mon_low AND curr_close > mon_low? 
        # "Deviate under" -> Price WAS under. "Close back inside" -> Price IS NOW inside (above Low).
        # We effectively check for a Bullish Engulfing or simple crossover of the Level.
        
        # Check "Cross Up" on Monday Low
        cross_up_mon_low = (df['close'] > df['mon_low']) & (df['close'].shift(1) <= df['mon_low'])
        # Ensure we actually deviated (Low was below). 
        # If shift(1) close <= mon_low, then by def it was below or at.
        
        # Check "Cross Down" on Monday High
        cross_down_mon_high = (df['close'] < df['mon_high']) & (df['close'].shift(1) >= df['mon_high'])
        
        # Valid Days: Tue(1) to Sun(6). 
        valid_day = df['day_of_week'] != 0 
        
        df['long_signal'] = cross_up_mon_low & valid_day
        df['short_signal'] = cross_down_mon_high & valid_day
        
        # 3. Assign Signal Types
        conditions = [
            df['long_signal'],
            df['short_signal']
        ]
        choices = [
            "MONDAY_LONG",
            "MONDAY_SHORT"
        ]
        
        df['signal_type'] = np.select(conditions, choices, default="NONE")
        
        # Props for UI/Analysis
        df['target_price'] = np.where(df['signal_type'] == 'MONDAY_LONG', df['mon_high'], 
                                   np.where(df['signal_type'] == 'MONDAY_SHORT', df['mon_low'], np.nan))
                                   
        # Stop Loss? Usually below deviation low (Swing Low). 
        # Hard to map exact swing low without loop/rolling. 
        # For now, we can leave SL blank or standard. User didn't specify.
        
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

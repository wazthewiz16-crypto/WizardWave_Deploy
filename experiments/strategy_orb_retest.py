import pandas as pd
import numpy as np

class ORBStrategy_Retest:
    def __init__(self, session_start_est="09:30", rr_ratio=3.0):
        self.session_str = session_start_est
        self.rr_ratio = rr_ratio

    def apply(self, df_5m: pd.DataFrame, asset_type: str = 'trad', htf_trend_map: dict = None) -> pd.DataFrame:
        """
        Applies ORB Strategy with 5m Confirmation and Retest.
        Input: 5m Dataframe.
        Logic:
        1. Define Range (First 15m = 3x 5m bars).
        2. Wait for Confirmation: 5m Candle Closes outside Range.
        3. Enter on Retest: Future Low touches Range High (Long) or High touches Range Low (Short).
        """
        if df_5m.empty: return df_5m
        df = df_5m.copy()
        
        # Timezone Setup
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        try:
            if df.index.tz is None:
                # Naive UTC -> ET
                df_tz = df.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                df_tz = df.index.tz_convert('America/New_York')
        except:
            df_tz = df.index
            
        df['local_time'] = df_tz
        df['session_day'] = df['local_time'].dt.date
        
        # Target Time
        t_h, t_m = map(int, self.session_str.split(':'))
        
        # Identify Range Bars (9:30, 9:35, 9:40)
        # 15m window = 3 bars starting at t_h:t_m
        
        days = df['session_day'].unique()
        
        df['signal_type'] = None
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        for day in days:
            day_mask = (df['session_day'] == day)
            day_df = df[day_mask]
            
            # Find 9:30 bar index
            start_mask = (day_df['local_time'].dt.hour == t_h) & (day_df['local_time'].dt.minute == t_m)
            if not start_mask.any(): continue
            
            start_idx = day_df.index[day_df['local_time'] == day_df.loc[start_mask].iloc[0]['local_time']].tolist()[0]
            
            # Get first 3 bars (15m)
            # Make sure we have 3 bars
            range_df = day_df.loc[start_idx:].head(3)
            # Validate times (must be consecutive 5m: 9:30, 9:35, 9:40)
            # Actually, just taking first 3 bars of session is safer if gaps exist?
            # Or strict 15m duration.
            # strict check:
            if len(range_df) < 3: continue
            
            r_high = range_df['high'].max()
            r_low = range_df['low'].min()
            r_height = r_high - r_low
            
            if r_height == 0: continue
            
            # Trading Phase (After 9:45)
            # Start checking from 4th bar
            trading_df = day_df.loc[range_df.index[-1]:].iloc[1:]
            
            state = "WAIT_CONFIRMATION" # -> "WAIT_RETEST" -> "FILLED"
            direction = None
            
            entry_lvl = 0.0
            sl_lvl = 0.0
            tp_lvl = 0.0
            
            for t, row in trading_df.iterrows():
                
                if state == "WAIT_CONFIRMATION":
                    # Check for CLOSE outside range
                    if row['close'] > r_high:
                        # Confirmed Long
                        if htf_trend_map and htf_trend_map.get(day) != 'UP':
                            # Trend filter rejection
                            continue
                            
                        direction = "LONG"
                        entry_lvl = r_high # Retest Level
                        sl_lvl = r_low
                        tp_lvl = entry_lvl + (r_height * self.rr_ratio)
                        state = "WAIT_RETEST"
                    elif row['close'] < r_low:
                        if htf_trend_map and htf_trend_map.get(day) != 'DOWN':
                            continue
                            
                        direction = "SHORT"
                        entry_lvl = r_low
                        sl_lvl = r_high
                        tp_lvl = entry_lvl - (r_height * self.rr_ratio)
                        state = "WAIT_RETEST"
                        
                elif state == "WAIT_RETEST":
                    # Check if Price touches Entry Level
                    # If LONG: Low <= Entry
                    # If SHORT: High >= Entry
                    
                    filled = False
                    if direction == "LONG":
                        if row['low'] <= entry_lvl:
                            filled = True
                    else: # SHORT
                        if row['high'] >= entry_lvl:
                            filled = True
                            
                    if filled:
                        # Trade Executed
                        df.at[t, 'signal_type'] = f"ORB_{direction}"
                        df.at[t, 'entry_price'] = entry_lvl
                        df.at[t, 'stop_loss'] = sl_lvl
                        df.at[t, 'take_profit'] = tp_lvl
                        break # One trade per day
                        
                    # Invalidations? 
                    # If Long, and Close < Range Low? (Failed breakout before retest)
                    # For simplicty, keep waiting until EOD.
        
        return df

import pandas as pd
import numpy as np

class ORBStrategy:
    def __init__(self, session_start_est="09:30", tp_mult=2.0):
        self.session_str = session_start_est
        self.tp_mult = tp_mult

    def apply(self, df: pd.DataFrame, asset_type: str = 'trad') -> pd.DataFrame:
        """
        Applies 15m ORB Strategy.
        df must be 15m timeframe.
        """
        if df.empty: return df
        df = df.copy()
        
        # Ensure Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # Timezone conversion for logic
        # Assuming DF input is UTC (from data_fetcher)
        # We need to detect local time for TradFi (09:30 ET)
        
        # Working with copy to avoid affecting original index if needed
        # But for applying filtering, we need ET column.
        
        try:
            # Check if tz aware
            if df.index.tz is None:
                # Assume UTC if crypto, or Unknown. 
                # data_fetcher returns naive UTC usually, or yfinance returns offset-naive locla? 
                # Let's verify standard: data_fetcher usually converts to Naive UTC.
                df_tz = df.index.tz_localize('UTC').tz_convert('America/New_York')
            else:
                df_tz = df.index.tz_convert('America/New_York')
        except:
            # If conversion fails, assume it's already local or something?
            # Fallback to pure time matching if UTC
            df_tz = df.index
        
        df['local_time'] = df_tz
        df['session_day'] = df['local_time'].dt.date
        
        # Identify Session Start Candles
        target_h, target_m = map(int, self.session_str.split(':'))
        
        # Helper to identify the 15m range candle
        # It's the candle that Starts at 09:30 and Ends at 09:45
        df['is_orb_candle'] = (df['local_time'].dt.hour == target_h) & (df['local_time'].dt.minute == target_m)
        
        # Strategy Columns
        df['signal_type'] = None
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        # Iterate by Day to Isolate Ranges
        # This is somewhat slow but accurate for ORB state management
        
        days = df['session_day'].unique()
        
        for day in days:
            day_mask = (df['session_day'] == day)
            day_df = df[day_mask]
            
            # Find ORB Candle
            orb_row = day_df[day_df['is_orb_candle']]
            
            if orb_row.empty:
                continue
                
            range_high = orb_row['high'].iloc[0]
            range_low = orb_row['low'].iloc[0]
            range_height = range_high - range_low
            
            # Skip Doji/Zero range
            if range_height == 0: continue
            
            # Valid Entry Window: After 09:45
            # Assuming 15m bars, next bar is 09:45
            
            orb_idx = orb_row.index[0]
            
            # Look for Breakout in subsequent bars of same day
            # We enforce "Same Day" exit for TradFi usually? 
            # Or continuous for Crypto.
            
            # For this backtest: Breakout Entry -> Hold until TP/SL or End of Session
            
            position = None # 'LONG' or 'SHORT'
            
            subsequent_bars = day_df.loc[orb_idx:].iloc[1:] # Skip the ORB candle itself
            
            for t, row in subsequent_bars.iterrows():
                if position is None:
                    # Check Entry
                    if row['high'] > range_high:
                        # LONG Breakout (Assume entry at Range High + slippage?)
                        # Or Close > Range High? Video usually says Break of High.
                        position = 'LONG'
                        entry_price = range_high
                        sl = range_low
                        tp = entry_price + (range_height * self.tp_mult)
                        
                        df.at[t, 'signal_type'] = 'ORB_LONG'
                        df.at[t, 'entry_price'] = entry_price
                        df.at[t, 'stop_loss'] = sl
                        df.at[t, 'take_profit'] = tp
                        
                        # One trade per day per direction? Or just one per day?
                        # Simple ORB: One per day usually.
                        break 
                        
                    elif row['low'] < range_low:
                        # SHORT Breakout
                        position = 'SHORT'
                        entry_price = range_low
                        sl = range_high
                        tp = entry_price - (range_height * self.tp_mult)
                        
                        df.at[t, 'signal_type'] = 'ORB_SHORT'
                        df.at[t, 'entry_price'] = entry_price
                        df.at[t, 'stop_loss'] = sl
                        df.at[t, 'take_profit'] = tp
                        break
        
        return df


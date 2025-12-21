import pandas as pd
import pandas_ta as ta
import numpy as np

class WizardScalpStrategy:
    """
    A faster, more aggressive variation of the WizardWave strategy designed for scalping
    on lower timeframes (15m, 1h, 4h).
    
    Key Characteristics:
    - Faster Lookback (Default: 8)
    - Tighter Cloud Spread
    - RSI Momentum Filter (Avoid buying tops / selling bottoms)
    """
    def __init__(self, 
                 lookback: int = 8, 
                 sensitivity: float = 1.0, 
                 cloud_spread: float = 0.4, 
                 use_rsi_filter: bool = True):
        self.lookback = lookback
        self.sensitivity = sensitivity
        self.cloud_spread = cloud_spread
        self.use_rsi_filter = use_rsi_filter

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        close = df['close']

        # --- 1. Fast Cloud Calculation ---
        # "Mango" EMAs with shorter periods
        eff_len_d1 = max(1, int(round(self.lookback / self.sensitivity)))
        eff_len_d2 = max(1, int(round(eff_len_d1 * self.cloud_spread)))
        
        df['mango_d1'] = ta.ema(close, length=eff_len_d1)
        df['mango_d2'] = ta.ema(close, length=eff_len_d2)

        df['cloud_top'] = df[['mango_d1', 'mango_d2']].max(axis=1)
        df['cloud_bottom'] = df[['mango_d1', 'mango_d2']].min(axis=1)
        
        # --- 2. RSI Filter & Bonus Filters ---
        df['rsi'] = ta.rsi(close, length=14)
        
        # ADX Filter (Avoid Choppiness) - Only trade if trend strength > 20
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if not adx_df.empty and 'ADX_14' in adx_df.columns:
            df['adx'] = adx_df['ADX_14']
        else:
            df['adx'] = 0
            
        # EMA Trend Filter (Align with Medium Term Trend)
        df['ema_trend'] = ta.ema(close, length=50)

        # --- 3. Signal Logic ---
        # Condition 1: Price Crosses Cloud (Breakout)
        # Condition 2: Price Pullback to Cloud (Trend Join) -> Simplified to "Above Cloud" = Bullish
        
        df['is_above_cloud'] = close > df['cloud_top']
        df['is_below_cloud'] = close < df['cloud_bottom']
        
        # Flip Detection
        df['prev_above'] = df['is_above_cloud'].shift(1).fillna(False)
        df['prev_below'] = df['is_below_cloud'].shift(1).fillna(False)
        
        df['bull_cross'] = df['is_above_cloud'] & (~df['prev_above'])
        df['bear_cross'] = df['is_below_cloud'] & (~df['prev_below'])
        
        # RST Logic: Don't Long if RSI > 75, Don't Short if RSI < 25 (Prevent FOMO at extremes)
        if self.use_rsi_filter:
            can_long = (df['rsi'] < 75) & (df['adx'] > 20) & (close > df['ema_trend'])
            can_short = (df['rsi'] > 25) & (df['adx'] > 20) & (close < df['ema_trend'])
        else:
            can_long = True
            can_short = True
            
        # Signals
        df['long_signal'] = df['bull_cross'] & can_long
        df['short_signal'] = df['bear_cross'] & can_short
        
        # Assign Types
        conditions = [
            (df['long_signal']),
            (df['short_signal'])
        ]
        choices = [
            "SCALP_LONG",
            "SCALP_SHORT"
        ]
        df['signal_type'] = np.select(conditions, choices, default="NONE")
        
        # Store Trend State for UI
        df['is_bullish'] = df['is_above_cloud']
        df['is_bearish'] = df['is_below_cloud']

        return df

    def get_active_trade(self, df: pd.DataFrame) -> dict:
        """
        Simulates scalping trade logic for the current active state.
        For scalping: 
        - Exit on Close crossing Cloud (Trend Change)
        - Tighter Hard SL (e.g. 1-2%) logic should be handled by the caller/app/pipeline
        """
        if df.empty:
            return None
            
        if 'signal_type' not in df.columns:
            df = self.apply(df)
            
        # Scan backward to find last valid signal that is still active (Trend held)
        # For Scalp, we only care about the *latest* signal if the trend is still valid
        
        # Simple State Machine
        position = None
        entry_price = 0.0
        entry_time = None
        
        for index, row in df.iterrows():
            signal = row['signal_type']
            close = row['close']
            top = row['cloud_top']
            bottom = row['cloud_bottom']
            
            # Entry
            if 'SCALP_LONG' in signal:
                position = 'LONG'
                entry_price = close
                entry_time = index
            elif 'SCALP_SHORT' in signal:
                position = 'SHORT'
                entry_price = close
                entry_time = index
                
            # Exit (Trail Stop Logic / Trend Flip)
            if position == 'LONG':
                if close < bottom: # Closed below cloud
                    position = None
            elif position == 'SHORT':
                if close > top: # Closed above cloud
                    position = None
                    
        if position:
            curr = df.iloc[-1]['close']
            pnl = (curr - entry_price)/entry_price * 100 if position == 'LONG' else (entry_price - curr)/entry_price * 100
            
            return {
                "Position": position,
                "Entry Time": entry_time,
                "Entry Price": entry_price,
                "PnL (%)": pnl
            }
            
        return None

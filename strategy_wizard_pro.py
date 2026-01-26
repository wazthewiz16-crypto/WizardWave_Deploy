import pandas as pd
import pandas_ta as ta
import numpy as np

class WizardWaveProStrategy:
    """
    Enhanced WizardWave Strategy (Pro-Max Version)
    Improvements:
    - Dynamic ATR Trailing Stops (Adaptive Risk)
    - Momentum Squeeze (Rising ADX confirmation)
    - RSI Extention Filter (Prevents buying the literal local top)
    """
    def __init__(self, 
                 lookback: int = 29, 
                 sensitivity: float = 1.06, 
                 cloud_spread: float = 0.64, 
                 atr_mult: float = 2.5,
                 rsi_upper: int = 75,
                 rsi_lower: int = 25):
        self.lookback = lookback
        self.sensitivity = sensitivity
        self.cloud_spread = cloud_spread
        self.atr_mult = atr_mult
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        close = df['close']

        # 1. Base Cloud Logic (Same as Original for consistency)
        eff_len_d1 = max(1, int(round(self.lookback / self.sensitivity)))
        eff_len_d2 = max(1, int(round(eff_len_d1 * self.cloud_spread)))
        
        df['mango_d1'] = ta.ema(close, length=eff_len_d1)
        df['mango_d2'] = ta.ema(close, length=eff_len_d2)
        df['cloud_top'] = df[['mango_d1', 'mango_d2']].max(axis=1)
        df['cloud_bottom'] = df[['mango_d1', 'mango_d2']].min(axis=1)
        
        # 2. ATR for Volatility (Dynamic Buffer & Trailing SL)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Noise Filter Buffer (0.5x ATR)
        df['cloud_top_buf'] = df['cloud_top'] + (df['atr'] * 0.5).fillna(0)
        df['cloud_bottom_buf'] = df['cloud_bottom'] - (df['atr'] * 0.5).fillna(0)
        
        # 3. Momentum & Extention Filters
        df['rsi'] = ta.rsi(close, length=14)
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty:
            df['adx'] = adx_df['ADX_14']
        else:
            df['adx'] = 0
            
        # Rising ADX (Trend Acceleration)
        df['adx_rising'] = df['adx'] > df['adx'].shift(1)
        
        # 4. Signal Logic
        df['is_above_cloud'] = close > df['cloud_top_buf']
        df['is_below_cloud'] = close < df['cloud_bottom_buf']
        
        df['prev_above'] = df['is_above_cloud'].shift(1).fillna(False).astype(bool)
        df['prev_below'] = df['is_below_cloud'].shift(1).fillna(False).astype(bool)
        
        # Trend Flips
        bull_flip = df['is_above_cloud'] & (~df['prev_above'])
        bear_flip = df['is_below_cloud'] & (~df['prev_below'])
        
        # APPLY PRO FILTERS
        # Only enter if not already overextended and trend is accelerating
        long_can_enter = (df['rsi'] < self.rsi_upper) & df['adx_rising']
        short_can_enter = (df['rsi'] > self.rsi_lower) & df['adx_rising']
        
        df['long_signal'] = bull_flip & long_can_enter
        df['short_signal'] = bear_flip & short_can_enter
        
        conditions = [df['long_signal'], df['short_signal']]
        choices = ["PRO_LONG", "PRO_SHORT"]
        df['signal_type'] = np.select(conditions, choices, default="NONE")
        
        return df

    def run_simulation(self, df: pd.DataFrame, initial_capital=1000):
        """
        Backtest simulation with Trailing Stops
        """
        if 'signal_type' not in df.columns:
            df = self.apply(df)
            
        trades = []
        position = None
        entry_price = 0.0
        trailing_sl = 0.0
        
        for idx, row in df.iterrows():
            close = row['close']
            atr = row['atr']
            signal = row['signal_type']
            
            # --- MANAGE ACTIVE TRADE ---
            if position == 'LONG':
                # Update Trailing SL
                new_sl = close - (atr * self.atr_mult)
                trailing_sl = max(trailing_sl, new_sl)
                
                # Exit Logic (Hit SL or Cloud Flip)
                if close <= trailing_sl or close < row['cloud_bottom']:
                    pnl = (close - entry_price) / entry_price
                    trades.append(pnl - 0.001) # 0.1% fee
                    position = None
            
            elif position == 'SHORT':
                # Update Trailing SL
                new_sl = close + (atr * self.atr_mult)
                trailing_sl = min(trailing_sl, new_sl) if trailing_sl > 0 else new_sl
                
                # Exit Logic (Hit SL or Cloud Flip)
                if close >= trailing_sl or close > row['cloud_top']:
                    pnl = (entry_price - close) / entry_price
                    trades.append(pnl - 0.001) # 0.1% fee
                    position = None
            
            # --- NEW ENTRY ---
            if position is None:
                if signal == "PRO_LONG":
                    position = 'LONG'
                    entry_price = close
                    trailing_sl = entry_price - (atr * self.atr_mult)
                elif signal == "PRO_SHORT":
                    position = 'SHORT'
                    entry_price = close
                    trailing_sl = entry_price + (atr * self.atr_mult)
                    
        return trades

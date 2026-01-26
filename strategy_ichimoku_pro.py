import pandas as pd
import pandas_ta as ta
import numpy as np

class IchimokuProStrategy:
    """
    Enhanced Ichimoku 'Kumo King' Strategy
    Improvements:
    - Cloud Thickness Filter: Avoids 'thin' cloud breakouts which are often noise.
    - Kijun Rejection: Adds signals for trend pullbacks (re-entry).
    - Future Twist Detection: Forecasts sentiment shift 30 bars ahead.
    """
    def __init__(self, 
                 tenkan=20, 
                 kijun=60, 
                 span_b=120, 
                 displacement=30, 
                 adx_threshold=22,
                 thickness_mult=0.6):
        self.tenkan_period = tenkan
        self.kijun_period = kijun
        self.span_b_period = span_b
        self.displacement = displacement
        self.adx_threshold = adx_threshold
        self.thickness_mult = thickness_mult
        
    def calculate_ichimoku(self, df):
        df = df.copy()
        high, low, close = df['high'], df['low'], df['close']
        
        # Core Lines
        df['tenkan'] = (high.rolling(window=self.tenkan_period).max() + low.rolling(window=self.tenkan_period).min()) / 2
        df['kijun'] = (high.rolling(window=self.kijun_period).max() + low.rolling(window=self.kijun_period).min()) / 2
        
        # Current Cloud (Shifted from 30 bars ago)
        df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(self.displacement)
        df['span_b'] = ((high.rolling(window=self.span_b_period).max() + low.rolling(window=self.span_b_period).min()) / 2).shift(self.displacement)
        
        # Future Cloud (Forecast for 30 bars ahead - look at unshifted values)
        df['future_span_a'] = (df['tenkan'] + df['kijun']) / 2
        df['future_span_b'] = (high.rolling(window=self.span_b_period).max() + low.rolling(window=self.span_b_period).min()) / 2
        
        # Indicators for Filtering
        df['atr'] = ta.atr(high, low, close, length=14)
        adx_df = ta.adx(high, low, close, length=14)
        df['adx'] = adx_df['ADX_14'] if adx_df is not None else 0
        
        return df

    def apply_strategy(self, df):
        if df.empty or len(df) < self.span_b_period: return pd.DataFrame()
        
        df = self.calculate_ichimoku(df)
        close = df['close']
        
        # 1. Cloud Boundaries
        df['cloud_top'] = df[['span_a', 'span_b']].max(axis=1)
        df['cloud_bottom'] = df[['span_a', 'span_b']].min(axis=1)
        df['cloud_thickness'] = abs(df['span_a'] - df['span_b'])
        
        # 2. TK Cross
        tk_bull = (df['tenkan'] > df['kijun']) & (df['tenkan'].shift(1) <= df['kijun'].shift(1))
        tk_bear = (df['tenkan'] < df['kijun']) & (df['tenkan'].shift(1) >= df['kijun'].shift(1))
        
        # 3. Kijun Rejection (The Pullback Entry)
        kijun_rej_bull = (df['low'] <= df['kijun']) & (close > df['kijun']) & (close > df['cloud_top'])
        kijun_rej_bear = (df['high'] >= df['kijun']) & (close < df['kijun']) & (close < df['cloud_bottom'])
        
        # 4. Filter: Cloud Thickness (Volatility Floor)
        thickness_ok = df['cloud_thickness'] > (df['atr'] * self.thickness_mult)
        
        # 5. Filter: ADX (Trend Strength)
        adx_ok = df['adx'] > self.adx_threshold
        
        # 6. Filter: Future Cloud Sentiment
        future_bullish = df['future_span_a'] > df['future_span_b']
        future_bearish = df['future_span_a'] < df['future_span_b']
        
        # 7. Combined Logic
        full_bull = (tk_bull | kijun_rej_bull) & (close > df['cloud_top']) & thickness_ok & adx_ok & future_bullish
        full_bear = (tk_bear | kijun_rej_bear) & (close < df['cloud_bottom']) & thickness_ok & adx_ok & future_bearish
        
        df['signal_type'] = "NONE"
        df.loc[full_bull, 'signal_type'] = "LONG"
        df.loc[full_bear, 'signal_type'] = "SHORT"
        
        # Exits
        df['exit_long'] = close < df['kijun']
        df['exit_short'] = close > df['kijun']
        
        return df

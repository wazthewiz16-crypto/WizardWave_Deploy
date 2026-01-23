import pandas as pd
import pandas_ta as ta
import numpy as np

class IchimokuStrategy:
    def __init__(self, tenkan=20, kijun=60, span_b=120, displacement=30, use_adx_filter=True, adx_threshold=20):
        self.tenkan_period = tenkan
        self.kijun_period = kijun
        self.span_b_period = span_b
        self.displacement = displacement
        self.use_adx_filter = use_adx_filter
        self.adx_threshold = adx_threshold
        
    def calculate_ichimoku(self, df):
        """
        Calculates Ichimoku Cloud components based on custom 20/60/120/30 settings.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line): (Max(High, 20) + Min(Low, 20)) / 2
        tenkan = (high.rolling(window=self.tenkan_period).max() + low.rolling(window=self.tenkan_period).min()) / 2
        
        # Kijun-sen (Base Line): (Max(High, 60) + Min(Low, 60)) / 2
        kijun = (high.rolling(window=self.kijun_period).max() + low.rolling(window=self.kijun_period).min()) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
        # Projected forward by displacement (30)
        span_a = ((tenkan + kijun) / 2).shift(self.displacement)
        
        # Senkou Span B (Leading Span B): (Max(High, 120) + Min(Low, 120)) / 2
        # Projected forward by displacement (30)
        span_b = ((high.rolling(window=self.span_b_period).max() + low.rolling(window=self.span_b_period).min()) / 2).shift(self.displacement)
        
        # Chikou Span (Lagging Span): Close shifted backward by displacement (30)
        # For signal check at time T, we compare Close[T] vs Prices[T-30]
        # We don't necessarily need to add a column shifted -30 because that's "future" data relative to the index.
        # Instead, we'll handle Chikou checks logically in the signal generation.
        
        df = df.copy()
        df['tenkan'] = tenkan
        df['kijun'] = kijun
        df['span_a'] = span_a
        df['span_b'] = span_b
        
        return df

    def apply_strategy(self, df, timeframe_name=""):
        """
        Generates Ichimoku signals:
        1. TK Cross (Tenkan crosses Kijun)
        2. Filter: Price relative to Cloud (Breakout/Trend)
        3. Filter: Chikou (Lagging Span) validation
        """
        if df.empty: return pd.DataFrame()
        
        df = self.calculate_ichimoku(df)
        
        # Identify TK Crosses
        # Bullish: Tenkan crosses above Kijun
        # Bearish: Tenkan crosses below Kijun
        
        prev_tenkan = df['tenkan'].shift(1)
        prev_kijun = df['kijun'].shift(1)
        
        # Cross Logic
        tk_cross_bull = (df['tenkan'] > df['kijun']) & (prev_tenkan <= prev_kijun)
        tk_cross_bear = (df['tenkan'] < df['kijun']) & (prev_tenkan >= prev_kijun)
        
        # --- Filters ---
        
        # 1. Cloud Filter (Price > Cloud for Bull, Price < Cloud for Bear)
        # Cloud Top/Bottom for current time
        # Note: 'span_a' and 'span_b' in the DF are already shifted, so they represent the cloud AT THIS MOMENT.
        cloud_top = df[['span_a', 'span_b']].max(axis=1)
        cloud_bottom = df[['span_a', 'span_b']].min(axis=1)
        
        above_cloud = df['close'] > cloud_top
        below_cloud = df['close'] < cloud_bottom
        
        # 2. Chikou Filter (Close vs Past Price)
        # Compare current close to High of T-30 (Bull) or Low of T-30 (Bear)
        # We need to shift the 'high' and 'low' columns FORWARD by 30 to compare current row with past data?
        # No.
        # Current Row T. Chikou for T is Close[T] plotted at T-30.
        # Check: Is Chikou > Price at T-30?
        # So Is Close[T] > High[T-30]?
        
        past_high = df['high'].shift(self.displacement)
        past_low = df['low'].shift(self.displacement)
        
        chikou_bull = df['close'] > past_high
        chikou_bear = df['close'] < past_low
        
        # Combine Signals
        signals = pd.Series(0, index=df.index)
        
        # Strong Buy: TK Cross Bull + Above Cloud + Chikou Bull
        # (User Note: "Crosses... outside the cloud carrying higher reliability")
        
        # Strong Buy: TK Cross Bull + Above Cloud + Chikou Bull
        # (User Note: "Crosses... outside the cloud carrying higher reliability")
        
        full_bull = tk_cross_bull & above_cloud & chikou_bull
        full_bear = tk_cross_bear & below_cloud & chikou_bear
        
        # 3. ADX Filter (Trend Strength)
        if self.use_adx_filter:
            try:
                # ADX requires a bit of lookback
                if len(df) > 20:
                    adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
                    if adx_df is not None and not adx_df.empty:
                        df['adx'] = adx_df['ADX_14']
                        full_bull = full_bull & (df['adx'] > self.adx_threshold)
                        full_bear = full_bear & (df['adx'] > self.adx_threshold)
                    else:
                        pass # Fallback if adx fails
            except Exception as e:
                print(f"ADX Error: {e}")
                pass
        
        # We can also add "Weak" signals (Inside Cloud) or "Neutral" signals, 
        # but for this automated test, let's stick to "High Quality" signals as requested (Strong Trend).
        
        # Initialize signal_type
        df['signal_type'] = None
        
        df.loc[full_bull, 'signal_type'] = "LONG"
        df.loc[full_bear, 'signal_type'] = "SHORT"
        
        # --- Exit Logic (for Backtesting) ---
        # Exit when price crosses Kijun in opposite direction?
        # Bull Trade Exit: Close < Kijun
        # Bear Trade Exit: Close > Kijun
        
        df['exit_long'] = df['close'] < df['kijun']
        df['exit_short'] = df['close'] > df['kijun']
        
        return df

if __name__ == "__main__":
    pass

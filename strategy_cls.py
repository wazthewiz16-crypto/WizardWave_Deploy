import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import os
from feature_engine import calculate_ml_features

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
        
        # Load ML Model
        self.ml_model = None
        self.ml_threshold = 0.30
        self.ml_features = []
        
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'north_star_ml_model.pkl')
            if os.path.exists(model_path):
                data = joblib.load(model_path)
                self.ml_model = data['model']
                self.ml_threshold = data['threshold']
                self.ml_features = data['features']
                print(f"[SUCCESS] North Star ML Filter Loaded. Threshold: {self.ml_threshold}")
            else:
                print("[WARN] ML Model not found. Running in Raw Mode.")
        except Exception as e:
            print(f"[WARN] Failed to load ML Model: {e}")

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
        
        return df

    def apply_mtf_forex(self, df_htf: pd.DataFrame, df_ltf: pd.DataFrame) -> pd.DataFrame:
        """
        Specialized Logic for Forex (High Mean Reversion, No Volume Data).
        - Swing Window: Smaller (3).
        - Candle Size: Ignored (ranges can be tight).
        - Volume: Ignored.
        - Focus: Wick Sweeps (Liquidity Grabs).
        """
        if df_htf.empty or df_ltf.empty:
            return df_ltf
        
        htf = df_htf.copy()
        htf['atr'] = ta.atr(htf['high'], htf['low'], htf['close'], length=14)
        
        # 1. Identify Liquidity Levels (Swing Highs/Lows)
        window = 3 # Forex respects shorter term pivots
        htf['rolling_max'] = htf['high'].shift(1).rolling(window=window).max()
        htf['rolling_min'] = htf['low'].shift(1).rolling(window=window).min()
        
        # 2. Check for Sweeps (Wicks)
        # Bullish Sweep: Low < Rolling Min, but Close > Rolling Min (Reclamation)
        # OR just Low < Rolling Min is enough to define the "Sweep Candle"
        htf['swept_low'] = htf['low'] < htf['rolling_min']
        htf['swept_high'] = htf['high'] > htf['rolling_max']
        
        # 3. Candle Quality (Close Strength)
        # We want the candle that swept to close somewhat favorably or at least not strictly bearish
        htf['close_above_mid'] = htf['close'] > ((htf['high'] + htf['low']) / 2)
        
        # Forex CLS Definition:
        # Just requires a Sweep. No need for "Large Body" or "Volume".
        is_cls_bull = htf['swept_low'] & htf['close_above_mid'] # Sweep Low + Close Top Half
        is_cls_bear = htf['swept_high'] & (~htf['close_above_mid']) # Sweep High + Close Bottom Half
        
        # Mark Ranges (Same as Standard)
        htf['cls_active_high'] = np.nan
        htf['cls_active_low'] = np.nan
        
        htf.loc[is_cls_bull, 'cls_active_high'] = htf['high']
        htf.loc[is_cls_bull, 'cls_active_low'] = htf['low']
        
        htf.loc[is_cls_bear, 'cls_active_high'] = htf['high']
        htf.loc[is_cls_bear, 'cls_active_low'] = htf['low']
        
        htf['range_high'] = htf['cls_active_high'].ffill()
        htf['range_low'] = htf['cls_active_low'].ffill()
        
        # Merge (Identical to Standard)
        htf = htf.sort_index()
        df_ltf = df_ltf.sort_index()
        htf_levels = htf[['range_high', 'range_low']].copy()
        htf_levels_shifted = htf_levels.shift(1)
        
        merged = pd.merge_asof(df_ltf, htf_levels_shifted, left_index=True, right_index=True, direction='backward')
        
        # LTF Entry Logic
        df = merged.copy()
        
        # Reclaim logic is same
        reclaim_long = (df['close'] > df['range_low']) & (df['close'].shift(1) < df['range_low'])
        reclaim_short = (df['close'] < df['range_high']) & (df['close'].shift(1) > df['range_high'])
        
        range_exists = df['range_high'].notna() & df['range_low'].notna()
        
        df['long_signal'] = reclaim_long & range_exists & (df['close'] > df['open'])
        df['short_signal'] = reclaim_short & range_exists & (df['close'] < df['open'])
        
        conditions = [df['long_signal'], df['short_signal']]
        choices = ["CLS_LONG", "CLS_SHORT"]
        
        df['signal_type'] = np.select(conditions, choices, default="NONE")
        
        # Targets (Same)
        df['mid_range'] = (df['range_high'] + df['range_low']) / 2
        df['target_price'] = np.where(df['signal_type'] == 'CLS_LONG', df['range_high'],
                                   np.where(df['signal_type'] == 'CLS_SHORT', df['range_low'], np.nan))
                                   
        # Stop Loss (Same)
        df['recent_min'] = df['low'].rolling(12).min()
        df['recent_max'] = df['high'].rolling(12).max()
        
        df['stop_loss'] = np.where(df['signal_type'] == 'CLS_LONG', df['recent_min'] * 0.999, # Tight Forex SL
                                   np.where(df['signal_type'] == 'CLS_SHORT', df['recent_max'] * 1.001, np.nan))
        
        return df

    def apply_north_star(self, df_htf: pd.DataFrame, df_ltf: pd.DataFrame) -> pd.DataFrame:
        """
        'North Star' Refined Strategy: Reversal / Fake Breakout.
        1. Core: Liquidity Sweep of HTF Range (Fake Breakout).
        2. Session: Only Trade during London/NY Killzones (07-11, 13-17 UTC approx).
        3. Structure: Sweep + Strong Close (Reclaim).
        4. No Trend Filter: We catch reversals at the edges.
        """
        if df_htf.empty or df_ltf.empty:
            return df_ltf
            
        # --- 1. HTF Ranges (Liquidity Pools) ---
        htf = df_htf.copy()
        
        # Identify Liquidity Levels (Swing Highs/Lows)
        window = 20 # SIGNIFICANT FLIP: 20 Days (Monthly Levels)
        htf['rolling_max'] = htf['high'].shift(1).rolling(window=window).max()
        htf['rolling_min'] = htf['low'].shift(1).rolling(window=window).min()
        
        htf['swept_low'] = htf['low'] < htf['rolling_min']
        htf['swept_high'] = htf['high'] > htf['rolling_max']
        
        # Strict Close: Must close strictly in the opposing tercile (Top 35% / Bottom 35%)
        # "Clear Reversal" = Wick is long, Body is closing strongly back in range.
        htf['range_len'] = htf['high'] - htf['low']
        htf['close_pos'] = (htf['close'] - htf['low']) / htf['range_len']
        
        # Size Filter: Range must be > 0.8 * ATR (Significant Candle)
        if 'atr' not in htf.columns:
             htf['atr'] = ta.atr(htf['high'], htf['low'], htf['close'], length=14)
        
        htf['is_large'] = htf['range_len'] > (htf['atr'] * 0.8)
        
        # Bullish: Close in Top 35%
        htf['strong_bull_close'] = htf['close_pos'] > 0.65
        # Bearish: Close in Bottom 35%
        htf['strong_bear_close'] = htf['close_pos'] < 0.35
        
        # Setup Definition: Purely based on Sweep + Rejection + Size
        
        # Bullish Setup: Sweep Low + Strong Close + Large
        is_setup_bull = htf['swept_low'] & htf['strong_bull_close'] & htf['is_large']
        
        # Bearish Setup: Sweep High + Weak Close + Large
        is_setup_bear = htf['swept_high'] & htf['strong_bear_close'] & htf['is_large']
        
        # Mark Ranges
        htf['range_high'] = np.where(is_setup_bull | is_setup_bear, htf['high'], np.nan) 
        htf['range_low'] = np.where(is_setup_bull | is_setup_bear, htf['low'], np.nan)
        
        # Forward Fill Active Zones
        htf['range_high'] = htf['range_high'].ffill()
        htf['range_low'] = htf['range_low'].ffill()
        
        # Merge to LTF
        htf = htf.sort_index()
        df_ltf = df_ltf.sort_index()
        
        htf_levels = htf[['range_high', 'range_low']].copy()
        htf_levels_shifted = htf_levels.shift(1) # Trade next day
        
        merged = pd.merge_asof(df_ltf, htf_levels_shifted, left_index=True, right_index=True, direction='backward')
        df = merged.copy()
        
        # --- 2. Session Filter ---
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour
            
        # Strict Killzones
        london_open = (df['hour'] >= 7) & (df['hour'] <= 11) # Extended slightly
        ny_open = (df['hour'] >= 13) & (df['hour'] <= 17)
        session_active = london_open | ny_open
        
        # --- 3. Entry Logic (Reclaim + Confirmation) ---
        reclaim_long = (df['close'] > df['range_low']) & (df['close'].shift(1) < df['range_low'])
        reclaim_short = (df['close'] < df['range_high']) & (df['close'].shift(1) > df['range_high'])
        
        range_exists = df['range_high'].notna()
        
        # Candle Color Confirmation (Model 2)
        strong_close_long = df['close'] > df['open']
        strong_close_short = df['close'] < df['open']
        
        # Signals (No Trend Filter)
        df['long_signal'] = reclaim_long & range_exists & session_active & strong_close_long
        df['short_signal'] = reclaim_short & range_exists & session_active & strong_close_short
        
        conditions = [df['long_signal'], df['short_signal']]
        choices = ["CLS_LONG", "CLS_SHORT"]
        
        df['signal_type'] = np.select(conditions, choices, default="NONE")
        
        # Targets and SL
        df['target_price'] = np.where(df['signal_type'] == 'CLS_LONG', df['range_high'],
                                   np.where(df['signal_type'] == 'CLS_SHORT', df['range_low'], np.nan))
                                   
        df['stop_loss'] = np.where(df['signal_type'] == 'CLS_LONG', 
                                   np.maximum(df['low'].rolling(5).min(), df['close'] * 0.99),
                                   np.where(df['signal_type'] == 'CLS_SHORT', 
                                            np.minimum(df['high'].rolling(5).max(), df['close'] * 1.01), np.nan))
                                   
        # --- 4. ML Filtering ---
        if self.ml_model:
            # Only process if we have signals
            signals_idx = df[df['signal_type'] != 'NONE'].index
            
            if len(signals_idx) > 0:
                # Calculate Features
                # We need to calculate features for the WHOLE dataframe to get accurate lag/rolling values
                # Then we only predict for the signal rows
                try:
                    df_features = calculate_ml_features(df.copy())
                    
                    # Extract features for signal points
                    # Ensure all features exist
                    missing_feats = [f for f in self.ml_features if f not in df_features.columns]
                    if not missing_feats:
                        X_signals = df_features.loc[signals_idx, self.ml_features]
                        
                        # Predict
                        probs = self.ml_model.predict_proba(X_signals)[:, 1]
                        
                        # Filter
                        # Create a boolean series aligned with df index
                        keep_mask = pd.Series(False, index=df.index)
                        keep_mask.loc[signals_idx] = probs >= self.ml_threshold
                        
                        # Update signal_type: Set to NONE if below threshold
                        # Track probability
                        df.loc[signals_idx, 'ml_prob'] = probs
                        
                        # Invalidate signals below threshold
                        # Get indices to drop
                        drop_idx = df.index[~keep_mask & (df['signal_type'] != 'NONE')]
                        df.loc[drop_idx, 'signal_type'] = 'NONE'
                    else:
                        print(f"[WARN] ML Filter Skipped. Missing Features: {missing_feats}")

                except Exception as e:
                    print(f"[WARN] ML Prediction Failed: {e}")
                    pass
                                   
        return df

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Legacy wrapper for single-timeframe data """
        return self.apply_mtf(df, df) # Treat same df as HTF and LTF (old behavior)

    @staticmethod
    def resample_weekly_from_4h(df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        Creates Synthetic Weekly Candles from 4H Data.
        Reason: Fetching '1wk' data directly often fails or has insufficient history.
        """
        if df_4h.empty: return pd.DataFrame()
        
        logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample to Weekly (W-MON) ending on Monday
        df_weekly = df_4h.resample('W-MON').agg(logic)
        df_weekly.dropna(inplace=True)
        
        return df_weekly

    @staticmethod
    def resample_daily_from_1h(df_1h: pd.DataFrame) -> pd.DataFrame:
        """ Creates Synthetic Daily Candles from 1H Data """
        if df_1h.empty: return pd.DataFrame()
        
        logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_daily = df_1h.resample('D').agg(logic)
        df_daily.dropna(inplace=True)
        
        return df_daily

    @staticmethod
    def resample_monthly_from_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
        """ Creates Synthetic Monthly Candles from Daily Data """
        if df_daily.empty: return pd.DataFrame()
        
        logic = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample to Month End
        df_monthly = df_daily.resample('M').agg(logic)
        df_monthly.dropna(inplace=True)
        
        return df_monthly

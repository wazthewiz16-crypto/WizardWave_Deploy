import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import joblib
from feature_engine import calculate_ml_features

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
                 use_rsi_filter: bool = True,
                 use_vol_filter: bool = True,
                 daily_loss_limit: float = -2.0): # -2% Daily Cap
        self.lookback = lookback
        self.sensitivity = sensitivity
        self.cloud_spread = cloud_spread
        self.use_rsi_filter = use_rsi_filter
        self.use_vol_filter = use_vol_filter
        self.daily_loss_limit = daily_loss_limit
        
        # Load ML Filter
        self.model_data = None
        model_path = os.path.join(os.path.dirname(__file__), 'wizard_scalp_ml_model.pkl')
        if os.path.exists(model_path):
            try:
                self.model_data = joblib.load(model_path)
                print("[SUCCESS] Scalp ML Filter Loaded.")
            except:
                print("[WARNING] Failed to load Scalp ML.")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        
        df = df.copy()
        close = df['close']
        
        # HACK: Detect High-Cap Crypto (BTC, ETH, DOGE) vs Others
        # We want the strict ATR filters ONLY for the major pairs that we validated.
        # This price check is a heuristic to avoid changing pipeline signatures.
        # ETH ~ 2000+, BTC ~ 60000+. 
        # DOGE ~ 0.10. SOL ~ 150.
        # If mean price > 500 (BTC/ETH) OR if price < 1 and vol > 1M (DOGE heuristic)? 
        # Simpler: Just check if volatility matches crypto profile?
        # Let's stick to the user request: "Only BTC and ETH".
        # BTC is definitely > 10000. ETH is definitely > 1000.
        
        current_price = close.iloc[-1] if not close.empty else 0
        mean_price = close.mean()
        is_btc_eth = (mean_price > 500) 

        # --- 1. ATR for Volatility (Noise Reduction) ---
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = df['atr'] / close

        # --- 2. Fast Cloud Calculation ---
        # "Mango" EMAs with shorter periods
        eff_len_d1 = max(1, int(round(self.lookback / self.sensitivity)))
        eff_len_d2 = max(1, int(round(eff_len_d1 * self.cloud_spread)))
        
        df['mango_d1'] = ta.ema(close, length=eff_len_d1)
        df['mango_d2'] = ta.ema(close, length=eff_len_d2)

        # Dynamic Padding: use a small fraction of ATR to "harden" the cloud
        # This helps ignore small spikes that don't represent a true trend shift
        df['cloud_top_raw'] = df[['mango_d1', 'mango_d2']].max(axis=1)
        df['cloud_bottom_raw'] = df[['mango_d1', 'mango_d2']].min(axis=1)
        
        # Enhanced Logic for BTC/ETH: Pad cloud by 0.2 * ATR
        # Standard Logic for Others: No padding (Original)
        if is_btc_eth:
            padding = df['atr'] * 0.2
        else:
            padding = 0
            
        df['cloud_top'] = df['cloud_top_raw'] + padding
        df['cloud_bottom'] = df['cloud_bottom_raw'] - padding
        
        # --- 3. Momentum Filters ---
        df['rsi'] = ta.rsi(close, length=14)
        
        # StochRSI (Only used in enhanced logic)
        stoch_rsi = ta.stochrsi(df['rsi'], length=14, rsi_length=14, k=3, d=3)
        if stoch_rsi is not None and not stoch_rsi.empty:
            df['stoch_k'] = stoch_rsi['STOCHRSIk_14_14_3_3']
        else:
            df['stoch_k'] = 50.0

        # ADX Filter
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None and not adx_df.empty and 'ADX_14' in adx_df.columns:
            df['adx'] = adx_df['ADX_14']
        else:
            df['adx'] = 0
            
        # Volume Filter (RVOL)
        if 'volume' in df.columns:
            df['vol_avg'] = df['volume'].rolling(20).mean()
            df['rvol'] = df['volume'] / df['vol_avg'].replace(0, 1)
        else:
            df['rvol'] = 1.5

        # EMA Trend Filter
        df['ema_trend'] = ta.ema(close, length=50)

        # --- 4. Signal Logic ---
        df['is_above_cloud'] = close > df['cloud_top']
        df['is_below_cloud'] = close < df['cloud_bottom']
        
        # Flip Detection
        df['prev_above'] = df['is_above_cloud'].shift(1).fillna(False).astype(bool)
        df['prev_below'] = df['is_below_cloud'].shift(1).fillna(False).astype(bool)
        
        df['bull_cross'] = df['is_above_cloud'] & (~df['prev_above'])
        df['bear_cross'] = df['is_below_cloud'] & (~df['prev_below'])
        
        long_conditions = True
        short_conditions = True
        
        if self.use_vol_filter:
             long_conditions &= (df['rvol'] > 1.0)
             short_conditions &= (df['rvol'] > 1.0)
            
        if self.use_rsi_filter:
            if is_btc_eth:
                # ENHANCED LOGIC (StochRSI + ADX + Trend)
                long_conditions &= (df['stoch_k'] < 80) & (df['adx'] > 25) & (close > df['ema_trend'])
                short_conditions &= (df['stoch_k'] > 20) & (df['adx'] > 25) & (close < df['ema_trend'])
            else:
                # ORIGINAL LOGIC (RSI levels)
                long_conditions &= (df['rsi'] < 75) & (df['adx'] > 20) & (close > df['ema_trend'])
                short_conditions &= (df['rsi'] > 25) & (df['adx'] > 20) & (close < df['ema_trend'])
            
        can_long = long_conditions
        can_short = short_conditions
            
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
        
        # --- 5. ML Meta-Labeling Filter ---
        if self.model_data and 'signal_type' in df.columns:
            df = calculate_ml_features(df)
            features = self.model_data['features']
            model = self.model_data['model']
            thresh = self.model_data['threshold']
            
            sig_mask = df['signal_type'] != 'NONE'
            if sig_mask.any():
                X = df.loc[sig_mask, features].fillna(0).values
                probs = model.predict_proba(X)[:, 1]
                
                # Apply filter: if prob < threshold, reset signal to NONE
                # We do this using a list comprehension or mapping back to the df
                prob_idx = 0
                for idx, row in df[sig_mask].iterrows():
                    if probs[prob_idx] < thresh:
                        df.at[idx, 'signal_type'] = "NONE"
                    prob_idx += 1

        # --- 6. Triple Barrier Generation (SL/TP) ---
        # Fixed 1.0% SL / 1.5% TP for Scalps
        df['stop_loss'] = 0.0
        df['target_price'] = 0.0
        
        long_mask = df['signal_type'] == 'SCALP_LONG'
        short_mask = df['signal_type'] == 'SCALP_SHORT'
        
        df.loc[long_mask, 'stop_loss'] = df['close'] * 0.99 # 1.0%
        df.loc[long_mask, 'target_price'] = df['close'] * 1.015 # 1.5%
        
        df.loc[short_mask, 'stop_loss'] = df['close'] * 1.01
        df.loc[short_mask, 'target_price'] = df['close'] * 0.985
        
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

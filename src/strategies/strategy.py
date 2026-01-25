import pandas as pd
import pandas_ta as ta
import numpy as np

class WizardWaveStrategy:
    def __init__(self, 
                 lookback: int = 29, 
                 sensitivity: float = 1.06, 
                 cloud_spread: float = 0.64, 
                 zone_pad_pct: float = 1.5,
                 use_rev_entry: bool = True):
        self.lookback = lookback
        self.sensitivity = sensitivity
        self.cloud_spread = cloud_spread
        self.zone_pad_pct = zone_pad_pct
        self.use_rev_entry = use_rev_entry

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies strat logic to DataFrame.
        Expects columns: ['open', 'high', 'low', 'close', 'volume']
        Returns df with signal columns.
        """
        if df.empty:
            return df
        
        # Working on a copy
        df = df.copy()
        close = df['close']

        # --- 2. CALCULATIONS ---
        # Apply Sensitivity
        # eff_len_d1 = math.max(1, math.round(master_lookback / sensitivity))
        eff_len_d1 = max(1, int(round(self.lookback / self.sensitivity)))
        
        # eff_len_d2 = math.max(1, math.round(eff_len_d1 * cloud_spread))
        eff_len_d2 = max(1, int(round(eff_len_d1 * self.cloud_spread)))

        # Mango D1 (Slow)
        # Pine: ta.ema(src, eff_len_d1)
        df['mango_d1'] = ta.ema(close, length=eff_len_d1)

        # Mango D2 (Fast)
        # Pine: ta.ema(src, eff_len_d2)
        df['mango_d2'] = ta.ema(close, length=eff_len_d2)

        # Cloud Boundaries
        df['cloud_top'] = df[['mango_d1', 'mango_d2']].max(axis=1)
        df['cloud_bottom'] = df[['mango_d1', 'mango_d2']].min(axis=1)
        
        # --- IMPROVEMENT 1: ATR Bands (Noise Reduction) ---
        # Add a buffer around the cloud based on ATR to prevent whipsaws in ranging markets
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        atr_mult = 0.5 
        df['cloud_top_buf'] = df['cloud_top'] + (df['atr'] * atr_mult).fillna(0)
        df['cloud_bottom_buf'] = df['cloud_bottom'] - (df['atr'] * atr_mult).fillna(0)

        # Fixed Zones (for UI)
        pad = self.zone_pad_pct / 100.0
        df['zone_upper'] = df['cloud_top'] * (1 + pad)
        df['zone_lower'] = df['cloud_bottom'] * (1 - pad)
        
        # --- IMPROVEMENT 2: Klinger Volume Oscillator (KVO) ---
        # Detect true buying pressure behind moves
        kvo = ta.kvo(df['high'], df['low'], df['close'], df['volume'], fast=34, slow=55, signal=13)
        if kvo is not None and not kvo.empty:
            df['kvo'] = kvo['KVO_34_55_13']
            df['kvo_sig'] = kvo['KVOs_34_55_13']
        else:
            df['kvo'] = 0
            df['kvo_sig'] = 0
            
        # --- IMPROVEMENT 3: MTF Trend Confirmation (Simulated) ---
        # We can't easily fetch other timeframes inside 'apply' without passing them.
        # Simulation: Use a very slow EMA (200) on current TF as a proxy for HTF Trend
        df['htf_trend'] = ta.ema(close, length=200)

        # --- 3. LOGIC & SIGNALS ---
        # State Logic: Must break the BUFFERED cloud to flip trend
        df['is_above_cloud'] = close > df['cloud_top_buf']
        df['is_below_cloud'] = close < df['cloud_bottom_buf']
        
        # Trend States
        df['is_bullish'] = df['is_above_cloud']
        df['is_bearish'] = df['is_below_cloud']

        # Bid Zone Logic
        df['in_bid_zone'] = (close <= df['zone_upper']) & (close >= df['zone_lower'])

        # Trend Flip Logic
        df['prev_is_bullish'] = df['is_bullish'].shift(1).fillna(False).astype(bool)
        df['prev_is_bearish'] = df['is_bearish'].shift(1).fillna(False).astype(bool)

        df['trend_flip_bull'] = df['is_bullish'] & (~df['prev_is_bullish'])
        df['trend_flip_bear'] = df['is_bearish'] & (~df['prev_is_bearish'])
        
        # Volatility & Volume Confirmation
        # Bull Flip Valid ONLY if KVO > Signal OR Close > HTF Trend
        # This filters out "fake pumps" against the macro trend
        
        # Default confirmations to True for safety (if data missing)
        kvo_bull = True
        kvo_bear = True
        
        if 'kvo' in df.columns and 'kvo_sig' in df.columns:
            # Handle cases where KVO might be NaN even if column exists
            # Fillna with 0 for comparison
            k = df['kvo'].fillna(0)
            ks = df['kvo_sig'].fillna(0)
            if not k.eq(0).all(): # If we actually have data
                kvo_bull = k > ks
                kvo_bear = k < ks
                
                
        # HTF Confirm (EMA 200)
        # Handle NaN/None in htf_trend (first 200 bars)
        htf_trend = df['htf_trend'].fillna(0)
        htf_bull = close > htf_trend
        htf_bear = close < htf_trend
        
        # Combined Confirmation (OR Logic: Volume OR Momenta)
        bull_confirm = kvo_bull | htf_bull
        bear_confirm = kvo_bear | htf_bear
        
        df['trend_flip_bull'] = df['trend_flip_bull'] & bull_confirm
        df['trend_flip_bear'] = df['trend_flip_bear'] & bear_confirm

        # --- 4. STRATEGY EXECUTION ---
        # Long: (Bullish AND InZone) OR (RevEntry AND BullFlip)
        long_zone = df['is_bullish'] & df['in_bid_zone']
        long_rev = self.use_rev_entry & df['trend_flip_bull']
        df['long_signal'] = long_zone | long_rev

        # Short: (Bearish AND InZone) OR (RevEntry AND BearFlip)
        short_zone = df['is_bearish'] & df['in_bid_zone']
        short_rev = self.use_rev_entry & df['trend_flip_bear']
        df['short_signal'] = short_zone | short_rev
        
        # Categorize Signal Type for the UI
        conditions = [
            (long_rev),
            (long_zone),
            (short_rev),
            (short_zone)
        ]
        choices = [
            "LONG_REV",
            "LONG_ZONE",
            "SHORT_REV",
            "SHORT_ZONE"
        ]
        df['signal_type'] = np.select(conditions, choices, default="NONE")

        # --- 5. SL/TP Generation ---
        # 3.3% SL / 10.0% TP for Wave Swing Trades (3R Ratio)
        df['stop_loss'] = 0.0
        df['target_price'] = 0.0
        
        long_mask = df['signal_type'].str.contains('LONG', na=False)
        short_mask = df['signal_type'].str.contains('SHORT', na=False)
        
        df.loc[long_mask, 'stop_loss'] = df['close'] * (1 - 0.033)
        df.loc[long_mask, 'target_price'] = df['close'] * (1 + 0.10)
        
        df.loc[short_mask, 'stop_loss'] = df['close'] * (1 + 0.033)
        df.loc[short_mask, 'target_price'] = df['close'] * (1 - 0.10)

        return df

    def get_active_trade(self, df: pd.DataFrame) -> dict:
        """
        Simulates strategy over the df to finds the *current* active trade if any.
        Returns dict with trade details or None.
        """
        if df.empty:
            return None
        
        # Ensure signals are calculated
        if 'signal_type' not in df.columns:
            df = self.apply(df)
            
        # Simulation State
        position = None # 'LONG' or 'SHORT'
        entry_price = 0.0
        entry_time = None
        sl_price = 0.0
        
        # Hard Stop Loss % (Fixed from Pine Script)
        sl_pct = 0.0333
        
        for index, row in df.iterrows():
            close = row['close']
            cloud_top = row['cloud_top']
            cloud_bottom = row['cloud_bottom']
            signal = row['signal_type']
            
            # --- EXIT LOGIC ---
            if position == 'LONG':
                # Hard SL
                if close <= sl_price:
                    position = None # Exit SL
                # Trend Broken
                elif close < cloud_bottom:
                    position = None # Exit Trend
            elif position == 'SHORT':
                # Hard SL
                if close >= sl_price:
                    position = None # Exit SL
                # Trend Broken
                elif close > cloud_top:
                    position = None # Exit Trend
                    
            # --- ENTRY LOGIC (Only if no position) ---
            if position is None:
                if 'LONG' in signal:
                    position = 'LONG'
                    entry_price = close
                    entry_time = index
                    sl_price = entry_price * (1 - sl_pct)
                elif 'SHORT' in signal:
                    position = 'SHORT'
                    entry_price = close
                    entry_time = index
                    sl_price = entry_price * (1 + sl_pct)
                    
            # --- REVERSAL ENTRY (Flip Logic implied by "use_rev_entry" in signal gen) ---
            # If we are already in a position, the Pine script strategy.entry("Long"...) 
            # would reverse a Short position if a Long signal fires.
            # So we check signals even if we have a position.
            
            if position == 'SHORT' and 'LONG' in signal:
                 # Flip Short to Long
                position = 'LONG'
                entry_price = close
                entry_time = index
                sl_price = entry_price * (1 - sl_pct)
                
            elif position == 'LONG' and 'SHORT' in signal:
                # Flip Long to Short
                position = 'SHORT'
                entry_price = close
                entry_time = index
                sl_price = entry_price * (1 + sl_pct)

        # End of Loop
        if position is not None:
            # Calculate PnL
            curr_price = df.iloc[-1]['close']
            if position == 'LONG':
                pnl_pct = (curr_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - curr_price) / entry_price * 100
                
            return {
                "Position": position,
                "Entry Time": entry_time,
                "Entry Price": entry_price,
                "Current Price": curr_price,
                "SL Price": sl_price,
                "PnL (%)": pnl_pct
            }
        
        return None

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.stats import entropy

class SpellSnipesFeatures:
    
    @staticmethod
    def get_weights_ffd(d, thres, lim):
        """
        Fractional Differentiation Weights (Fixed Window)
        Source: Lopez de Prado, Chapter 5
        """
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres: break
            w.append(w_)
            k += 1
            if k >= lim: break
        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    @staticmethod
    def frac_diff_ffd(series, d, thres=1e-5):
        """
        Constant width window fractional differentiation.
        """
        # 1) Compute weights for the longest series
        w = SpellSnipesFeatures.get_weights_ffd(d, thres, len(series))
        width = len(w) - 1
        
        # 2) Apply weights to values
        df = {}
        # Rolling dot product
        # For simplicity/speed in backtest, we might use standard difference
        # or a fast approximation. Implementing full rolling dot is slow in python loops.
        # We'll use a simplified pandas rolling apply if possible, or simple diff for speed
        # if series is long.
        # Implementation of full FFD:
        
        output = []
        # Pre-process nan
        series_f = series.fillna(method='ffill').dropna()
        vals = series_f.values
        
        if len(vals) < width: return pd.Series(index=series.index, dtype=float)
        
        # Fast convolution?
        # Only feasible for short windows. If d=0.4, window might be long.
        # Let's pivot to "Standard Differencing" if FFD is too complex for this snippet?
        # No, user asked for it. 
        # Using a fixed small window approximation (e.g. window=20)
        
        # Vectorized implementation via convolution
        # w is kernel. 
        import scipy.signal
        # mode='valid' returns only parts where full overlap
        # w needs to be flattened
        res = scipy.signal.convolve(vals, w.flatten(), mode='valid')
        
        # Re-index
        out_series = pd.Series(res, index=series_f.index[width:])
        return out_series

    @staticmethod
    def calculate_entropy(series, window=20):
        """
        Rolling Shannon Entropy of discretized returns.
        """
        # Discretize returns into bins to calc probability distribution
        # Rolling apply
        def _ent(x):
            # Hist
            counts, _ = np.histogram(x, bins=5, density=True)
            # Remove zeros for log
            counts = counts[counts > 0]
            return entropy(counts)
            
        return series.rolling(window=window).apply(_ent, raw=True)

    @staticmethod
    def calculate_vpin_features(df, bucket_vol=1000):
        """
        Approximates VPIN using Volume Bars (or Time Bars if Volume Bars hard to construct).
        Given we have Time Bars (15m/1h), we approximate VPIN on Time Bars directly.
        
        VPIN = |BuyVol - SellVol| / TotalVol (Smoothed)
        """
        df = df.copy()
        # 1. Tick Rule Approximation (Bulk Classification)
        # Buy Vol = Vol * (Close - Low) / (High - Low) ?? 
        # Or standard: if Close > Open -> Buy Vol = Vol?
        # Let's use: if Close > PrevClose -> Buy.
        # Improved: Proportional measure (Pessimist/Optimist)
        
        # Simple Bulk Classification
        delta = df['close'].diff()
        
        df['buy_vol'] = np.where(delta > 0, df['volume'], 0)
        df['sell_vol'] = np.where(delta < 0, df['volume'], 0)
        
        # For 0 change, split?
        zero_mask = (delta == 0)
        df.loc[zero_mask, 'buy_vol'] = df.loc[zero_mask, 'volume'] * 0.5
        df.loc[zero_mask, 'sell_vol'] = df.loc[zero_mask, 'volume'] * 0.5
        
        # Rolling Imbalance
        window = 20 # bars
        
        roll_buy = df['buy_vol'].rolling(window=window).sum()
        roll_sell = df['sell_vol'].rolling(window=window).sum()
        total_vol = df['volume'].rolling(window=window).sum()
        
        # VPIN (Order Flow Toxicity)
        # High VPIN = High Imbalance
        df['VPIN'] = (roll_buy - roll_sell).abs() / total_vol
        
        # Signed Flow (Directional 0-1)
        # OFI Proxy
        df['OFI_Proxy'] = (roll_buy - roll_sell) / total_vol
        
        return df

    @staticmethod
    def add_features(df):
        if df.empty: return df
        df = df.copy()
        
        # 1. Structural Features
        # Fractional Diff (d=0.4 preservation)
        # Note: FFD reduces length. For sim, we might stick to Returns or Log Returns
        # if FFD is too computationally heavy for live app.
        # User asked for it. We'll verify length. 
        # If dataset is small, FFD cuts off start.
        
        # d=0.4
        df['close_ffd'] = SpellSnipesFeatures.frac_diff_ffd(df['close'], d=0.4)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # Entropy (Regime)
        df['entropy'] = SpellSnipesFeatures.calculate_entropy(df['log_ret'], window=20)
        
        # Autocorrelation (Regime)
        df['autocorr'] = df['log_ret'].rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
        
        # 2. VPIN / Flow
        df = SpellSnipesFeatures.calculate_vpin_features(df)
        
        # 3. Technicals (VWAP, BB, OBV)
        if 'volume' in df.columns:
            # VWAP requires 'high','low','close','volume'
            # pandas_ta vwap might require datetime index
            try:
                # Custom VWAP since ta.vwap relies on session anchors
                # Rolling VWAP for intraday
                cv = (df['high'] + df['low'] + df['close']) / 3
                cum_vol = df['volume'].cumsum()
                cum_pv = (cv * df['volume']).cumsum()
                df['vwap'] = cum_pv / cum_vol
                
                # VWAP Reversion: Distance
                df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']
            except: pass
            
            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])
            # OBV Divergence Proxy (Slope of OBV vs Slope of Price)
            df['obv_slope'] = ta.slope(df['obv'], length=10)
            df['price_slope'] = ta.slope(df['close'], length=10)
            df['obv_div'] = np.where(np.sign(df['obv_slope']) != np.sign(df['price_slope']), 1, 0)

        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        if bb is not None:
             df = pd.concat([df, bb], axis=1)
             # BB Width (Volatility)
             # BB %B (Oscillator)
        
        return df

    @staticmethod
    def get_triple_barrier_labels(df, sl_tp_ratio=1.0, barrier_len=20, min_ret=0.002):
        """
        Triple Barrier Method
        Outcome: -1 (SL), 0 (Time Limit), 1 (TP)
        """
        labels = pd.Series(index=df.index, data=0)
        
        # Daily Volatility for Dynamic Barriers?
        # User implies Standard or Dynamic. Let's use Dynamic Volatility based
        # vol = df['log_ret'].rolling(20).std()
        
        future_window = barrier_len
        
        for t in range(len(df) - future_window):
            curr_slice = df.iloc[t]
            future_slice = df.iloc[t+1 : t+1+future_window]
            
            entry = curr_slice['close']
            if entry == 0: continue
            
            # Dynamic Barriers based on recent volatility or fixed %?
            # Used Fixed % for simplicity in V1 (User: "Triple barrier method")
            # Usually implies [PT, SL, Time]
            # Let's use ATR based or Fixed.
            # 15m Scalping: Target ~0.5%?
            
            target = min_ret
            sl = -min_ret * sl_tp_ratio
            
            # Find first barrier touch
            
            returns = (future_slice['close'] - entry) / entry
            
            # Check TP (Upper Barrier)
            tp_hit = returns[returns >= target].first_valid_index()
            # Check SL (Lower Barrier)
            sl_hit = returns[returns <= sl].first_valid_index()
            
            outcome = 0 # Time Limit
            end_time = future_slice.index[-1]
            
            if tp_hit and sl_hit:
                if tp_hit < sl_hit: outcome = 1
                else: outcome = -1
            elif tp_hit:
                outcome = 1
            elif sl_hit:
                outcome = -1
            
            labels.iloc[t] = outcome
            
        return labels

import pandas as pd
import pandas_ta as ta
import numpy as np

def calculate_ml_features(df):
    """
    Calculates technical features for the ML model.
    Shared logic between pipeline.py (Training) and app.py (Inference).
    
    Features:
    - volatility (20 period rolling std dev of returns)
    - rsi (14)
    - ma_dist (Distance from SMA 50)
    - adx (Trend Strength)
    - mom (Momentum ROC 10)
    - rvol (Relative Volume)
    - bb_width (Bollinger Band Width - Volatility State)
    - candle_ratio (Body / High-Low Range - Conviction)
    """
    df = df.copy()
    
    # Ensure necessary columns exist
    if df.empty or 'close' not in df.columns:
        return df

    # 1. Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # 2. RSI
    df['rsi'] = ta.rsi(df['close'], length=14)
    
    # 3. MA Distance
    df['sma50'] = ta.sma(df['close'], length=50)
    df['ma_dist'] = (df['close'] / df['sma50']) - 1
    
    # 4. ADX (Trend Strength)
    if 'high' in df.columns and 'low' in df.columns:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if not adx_df.empty and 'ADX_14' in adx_df.columns:
            df['adx'] = adx_df['ADX_14']
        else:
            df['adx'] = 0
    else:
        df['adx'] = 0
            
    # 5. Momentum
    df['mom'] = ta.roc(df['close'], length=10)
    
    # --- NEW FEATURES (Signal Improvement Plan) ---
    
    # 6. Relative Volume (RVOL)
    # Ratio of current volume to 20-period average volume
    if 'volume' in df.columns:
        df['avg_vol'] = df['volume'].rolling(20).mean()
        # Avoid division by zero
        df['rvol'] = df['volume'] / df['avg_vol'].replace(0, 1)
        # Cap infs
        df['rvol'] = df['rvol'].replace([np.inf, -np.inf], 0)
    else:
        df['rvol'] = 1.0 # Neutral

    # 7. Bollinger Band Width (Volatility Squeeze/Expansion)
    # (Upper - Lower) / Middle
    bb = ta.bbands(df['close'], length=20, std=2)
    if bb is not None and not bb.empty:
        # Robust column extraction
        # Pandas TA names can be BBU_20_2.0 or BBU_20_2 etc.
        # We find columns that start with BBU, BBL, BBM
        cols = bb.columns
        bbu_col = [c for c in cols if c.startswith('BBU')][0]
        bbl_col = [c for c in cols if c.startswith('BBL')][0]
        bbm_col = [c for c in cols if c.startswith('BBM')][0]
        
        bbu = bb[bbu_col]
        bbl = bb[bbl_col]
        bbm = bb[bbm_col]
        
        df['bb_width'] = (bbu - bbl) / bbm
    else:
        df['bb_width'] = 0

    # 8. Candle Body Ratio (Conviction)
    # Abs(Close - Open) / (High - Low)
    if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns:
        body_size = (df['close'] - df['open']).abs()
        range_size = (df['high'] - df['low'])
        # Avoid div by zero
        df['candle_ratio'] = body_size / range_size.replace(0, 1)
    else:
        df['candle_ratio'] = 0.5 # Neutral
        
    # 9. ATR (Volatility Normalization)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    # Normalize ATR as % of Close
    df['atr_pct'] = df['atr'] / df['close']
    
    # 10. MFI (Money Flow Index - Volume + Price)
    if 'volume' in df.columns:
        mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        if mfi is not None:
             df['mfi'] = mfi
        else:
             df['mfi'] = 50 # Neutral
    else:
        df['mfi'] = 50

    # --- WIZARD INDICATOR FEATURES (User Requested) ---
    # Normalize absolute levels to relative distances for ML stability
    
    # Mango D1
    if 'mango_d1' in df.columns:
        df['mango_d1_dist'] = (df['close'] - df['mango_d1']) / df['close']
    else:
        df['mango_d1_dist'] = 0.0

    # Mango D2
    if 'mango_d2' in df.columns:
        df['mango_d2_dist'] = (df['close'] - df['mango_d2']) / df['close']
    else:
        df['mango_d2_dist'] = 0.0

    # Upper Bid Zone
    if 'zone_upper' in df.columns:
        df['upper_zone_dist'] = (df['close'] - df['zone_upper']) / df['close']
    else:
        df['upper_zone_dist'] = 0.0

    # Lower Bid Zone
    if 'zone_lower' in df.columns:
        df['lower_zone_dist'] = (df['close'] - df['zone_lower']) / df['close']
    else:
        df['lower_zone_dist'] = 0.0


    # Cloud Distance (Overextension)
    if 'cloud_top' in df.columns and 'cloud_bottom' in df.columns:
        # Distance from the relevant cloud boundary
        # If Price > Cloud => Distance from Cloud Top
        # If Price < Cloud => Distance from Cloud Bottom
        df['cloud_top_dist'] = (df['close'] - df['cloud_top']) / df['close']
        df['cloud_bot_dist'] = (df['close'] - df['cloud_bottom']) / df['close']
        
        # Combined 'Cloud Distance' feature
        # If Long Setup: We want to be close to cloud top (Breakout) or cloud bottom (support)? 
        # Actually WizardWave buys on:
        # 1. Breakout (Close > Top)
        # 2. Pullback (Close > Top but dipping)
        # Raw distance is good enough for Tree to split on.
    else:
        df['cloud_top_dist'] = 0.0
        df['cloud_bot_dist'] = 0.0

    # Regime Filter (Bull/Bear)
    if 'sma50' in df.columns: # reuse sma50 available
        # Calculate EMA 200 for long term trend context
        df['ema200'] = ta.ema(df['close'], length=200)
        df['regime_bull'] = (df['close'] > df['ema200']).astype(int)
    else:
        df['regime_bull'] = 0 # Default Bearish/Neutral

    # Fill any remaining NaNs (e.g. at start of dataframe)
    df.fillna(0, inplace=True)
    
    return df


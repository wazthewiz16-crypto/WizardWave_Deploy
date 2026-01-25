import pandas as pd
import pandas_ta as ta
import numpy as np

def get_frac_diff_weights(d, size):
    """Lopez de Prado Fractional Differentiation weights."""
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w

def frac_diff(series, d, thres=0.01):
    """
    Apply Fractional Differentiation to a series.
    This preserves more memory than a standard d=1 difference.
    """
    # 1. Generate weights
    weights = get_frac_diff_weights(d, len(series))
    
    # 2. Find weight cutoff (skip small weights for efficiency)
    # Using a simple fixed window for stability in production
    window = 100 
    if len(series) < window: return series.diff().fillna(0)
    
    weights = get_frac_diff_weights(d, window)
    res = series.rolling(window).apply(lambda x: np.dot(x, weights).item(), raw=True)
    return res.fillna(0)

def calculate_ml_features(df, macro_df=None, crypto_macro_df=None):
    """
    Calculates technical features for the ML model.
    Shared logic between pipeline.py (Training) and app.py (Inference).
    
    macro_df: Optional dataframe containing global indicators like DXY.
    crypto_macro_df: Optional dataframe containing BTC-USD for altcoin correlation.
    """
    df = df.copy()
    
    # --- 0. Pre-Feature: Macro Integration ---
    # --- 0. Pre-Feature: Macro Integration ---
    if macro_df is not None and not macro_df.empty:
        # Align macro data to the main df index
        # Fix: Drop duplicates in macro index to allow reindexing
        if not macro_df.index.is_unique:
            macro_df = macro_df.loc[~macro_df.index.duplicated(keep='last')]
            
        macro_aligned = macro_df['close'].reindex(df.index, method='ffill')
        if not macro_aligned.isna().all():
            df['dxy_close'] = macro_aligned
            df['dxy_ret'] = df['dxy_close'].pct_change()
            df['dxy_corr'] = df['close'].pct_change().rolling(20).corr(df['dxy_ret'])
            df['dxy_dist'] = df['dxy_close'] / df['dxy_close'].rolling(50).mean() - 1
        else:
            df['dxy_ret'] = 0.0
            df['dxy_corr'] = 0.0
            df['dxy_dist'] = 0.0
    else:
        df['dxy_ret'] = 0.0
        df['dxy_corr'] = 0.0
        df['dxy_dist'] = 0.0

    if crypto_macro_df is not None and not crypto_macro_df.empty:
        # Correlation with BTC (Crypto Beta)
        if not crypto_macro_df.index.is_unique:
            crypto_macro_df = crypto_macro_df.loc[~crypto_macro_df.index.duplicated(keep='last')]
            
        btc_aligned = crypto_macro_df['close'].reindex(df.index, method='ffill')
        if not btc_aligned.isna().all():
            df['btc_corr'] = df['close'].pct_change().rolling(20).corr(btc_aligned.pct_change())
            # BTC Momentum as a feature
            df['btc_mom'] = btc_aligned.pct_change(10)
        else:
            df['btc_corr'] = 0.0
            df['btc_mom'] = 0.0
    else:
        df['btc_corr'] = 0.0
        df['btc_mom'] = 0.0
    
    # Ensure necessary columns exist
    if df.empty or 'close' not in df.columns:
        return df

    # 1. Volatility & Stationarity
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    
    # 1b. Fractional Differentiation (Preserves price memory better than returns)
    # Using d=0.4 as standard FML starting point
    df['close_frac'] = frac_diff(df['close'], d=0.4)
    
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

    # 11. Seasonality / Cycle Features (Crypto 6-Month Cycle)
    # Pivot Points: March (3) and September (9)
    if 'datetime' in df.columns:
        # Ensure dt accessor works
        dt_col = df['datetime'].dt
    else:
        # If index is datetime
        dt_col = df.index
        
    # Cyclical Month Encoding (So Dec is close to Jan)
    df['month_sin'] = np.sin(2 * np.pi * dt_col.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt_col.month / 12)
    
    # Distance to Pivot (March/Sept)
    # We want a feature that peaks near Mar/Sept or resets
    # 3 = March, 9 = Sept.
    # Dist = min(abs(month - 3), abs(month - 9)) ?
    # Let's map months 3 & 9 to 0 (Pivot), others higher.
    # 3->0, 4->1, 9->0, 10->1, 12->3, 1->2...
    # Easier: Just let RF learn from Sine/Cosine which is precise.
    
    # Explicit "Cycle Regime" Feature: 1 for Mar-Aug, -1 for Sept-Feb?
    # User Theory: Two 6-month blocks.
    # Mar(3) to Aug(8) => Cycle A
    # Sept(9) to Feb(2) => Cycle B
    df['cycle_regime'] = np.where((dt_col.month >= 3) & (dt_col.month <= 8), 1, 0)

    # Fill any remaining NaNs (e.g. at start of dataframe)
    df.fillna(0, inplace=True)
    
    return df


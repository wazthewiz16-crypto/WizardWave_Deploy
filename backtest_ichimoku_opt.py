import pandas as pd
import numpy as np
import json
import pandas_ta as ta
from data_fetcher import fetch_data
from strategy_ichimoku import IchimokuStrategy
from pipeline import get_asset_type
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('future.no_silent_downcasting', True)

class IchimokuAdvanced(IchimokuStrategy):
    def __init__(self, adx_threshold=None, lookahead=True, bounce=False):
        super().__init__()
        self.adx_threshold = adx_threshold
        self.lookahead = lookahead
        self.bounce = bounce

    def apply_strategy(self, df, timeframe_name=""):
        if df.empty: return pd.DataFrame()
        
        df = self.calculate_ichimoku(df)
        
        # --- Indicators ---
        if self.adx_threshold:
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None and not adx.empty:
                df['adx'] = adx['ADX_14']
            else:
                df['adx'] = 0

        # --- Logic ---
        # 1. TK Cross (Baseline)
        prev_tenkan = df['tenkan'].shift(1)
        prev_kijun = df['kijun'].shift(1)
        tk_cross_bull = (df['tenkan'] > df['kijun']) & (prev_tenkan <= prev_kijun)
        tk_cross_bear = (df['tenkan'] < df['kijun']) & (prev_tenkan >= prev_kijun)

        # 2. Cloud Filter
        cloud_top = df[['span_a', 'span_b']].max(axis=1)
        cloud_bottom = df[['span_a', 'span_b']].min(axis=1)
        above_cloud = df['close'] > cloud_top
        below_cloud = df['close'] < cloud_bottom

        # 3. Chikou Filter
        past_high = df['high'].shift(self.displacement)
        past_low = df['low'].shift(self.displacement)
        chikou_bull = df['close'] > past_high
        chikou_bear = df['close'] < past_low
        
        # 4. Future Cloud (Lookahead)
        future_span_a = df['span_a'].shift(-26) # Real data has it forward, but check indexing
        # calculated verify: calculate_ichimoku shifts spans forward by displacement (30).
        # So at index T, span_a is the FUTURE cloud for T+0? 
        # No. Std Ichimoku: Span A is plotted 26 periods ahead.
        # In Pandas, usually we simple shift() it so the value aligns with calculation time, OR we align it with plot time.
        # Let's check calculate_ichimoku:
        # span_a = (...).shift(30).
        # This means at time T, 'span_a' holds the value that was generated at T-30? 
        # No. It means value generated at T is pushed to T+30.
        # Wait, shift(30) pushes data DOWN (forward index). 
        # So at index T, we see the value from T-30.
        # This represents "The Cloud Above Us Now" (Created in the past).
        # CORRECT.
        # So "Future Cloud" (The one we are building now for T+30) is the UN-SHIFTED value.
        # We need to calculate unshifted components.
        
        tenkan_raw = (df['high'].rolling(20).max() + df['low'].rolling(20).min()) / 2
        kijun_raw = (df['high'].rolling(60).max() + df['low'].rolling(60).min()) / 2
        future_a = (tenkan_raw + kijun_raw) / 2
        
        h120 = df['high'].rolling(120).max()
        l120 = df['low'].rolling(120).min()
        future_b = (h120 + l120) / 2
        
        future_green = future_a > future_b
        future_red = future_a < future_b

        # --- Signals ---
        df['signal_type'] = None
        
        # Base Signal
        bull_signal = tk_cross_bull & above_cloud & chikou_bull
        bear_signal = tk_cross_bear & below_cloud & chikou_bear
        
        # Apply Filters
        if self.adx_threshold:
            bull_signal = bull_signal & (df['adx'] > self.adx_threshold)
            bear_signal = bear_signal & (df['adx'] > self.adx_threshold)
            
        if self.lookahead:
            bull_signal = bull_signal & future_green
            bear_signal = bear_signal & future_red

        df.loc[bull_signal, 'signal_type'] = 'LONG'
        df.loc[bear_signal, 'signal_type'] = 'SHORT'
        
        # Exit Logic
        df['exit_long'] = df['close'] < df['kijun']
        df['exit_short'] = df['close'] > df['kijun']
        
        return df

def run_report():
    print("Generating Ichimoku Optimization Report...")
    
    with open('strategy_config.json', 'r') as f:
        config = json.load(f)
    assets = config['assets']
    
    configs = [
        {'name': '1. Baseline', 'adx': None, 'look': False},
        {'name': '2. Trend (ADX>20)', 'adx': 20, 'look': False},
        {'name': '3. Future Cloud', 'adx': None, 'look': True},
        {'name': '4. Quality (ADX+Future)', 'adx': 20, 'look': True}
    ]
    
    results = []
    
    # Test on 4H Timeframe (Swing)
    timeframe = '4h'
    
    # Filter for Crypto only for now as requested by user context ("crypto trades" in previous turn? No, "ichimoku cloud signals" general)
    # But user typically focuses on everything. Let's do top 5 assets.
    test_assets = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'EURUSD=X', 'GC=F']
    
    for symbol in test_assets:
        try:
            asset_type = get_asset_type(symbol)
            fetch_type = 'trad' if asset_type in ['forex', 'trad'] else 'crypto'
            if '-' in symbol or '^' in symbol or '=' in symbol: fetch_type = 'trad'
            
            df = fetch_data(symbol, asset_type=fetch_type, timeframe=timeframe, limit=1000)
            if df.empty: continue
            
            cutoff = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=90)
            
            for cfg in configs:
                strat = IchimokuAdvanced(adx_threshold=cfg['adx'], lookahead=cfg['look'])
                df_strat = strat.apply_strategy(df.copy())
                
                if 'signal_type' not in df_strat.columns: continue
                
                signals = df_strat[df_strat['signal_type'].notna()]
                signals = signals[signals.index >= cutoff]
                
                # Simulate PnL
                count = 0
                wins = 0
                total_pnl = 0.0
                
                for t, row in signals.iterrows():
                    entry = row['close']
                    s_type = row['signal_type']
                    
                    # Exit Loop
                    future = df_strat.loc[df_strat.index > t]
                    if future.empty: continue
                    
                    exit_price = entry # Default BE
                    
                    if s_type == 'LONG':
                        # Find First Exit
                        exits = future[future['exit_long']]
                        if not exits.empty:
                            exit_price = exits.iloc[0]['close']
                        else:
                            exit_price = future.iloc[-1]['close']
                        pnl = (exit_price - entry) / entry
                    else:
                        exits = future[future['exit_short']]
                        if not exits.empty:
                            exit_price = exits.iloc[0]['close']
                        else:
                            exit_price = future.iloc[-1]['close']
                        pnl = (entry - exit_price) / entry
                        
                    count += 1
                    if pnl > 0: wins += 1
                    total_pnl += pnl
                    
                if count > 0:
                    results.append({
                        "Config": cfg['name'],
                        "Asset": symbol,
                        "Signals": count,
                        "WinRate": round((wins/count)*100, 1),
                        "AvgPnL": round((total_pnl/count)*100, 2)
                    })
                    
        except Exception as e:
            print(f"Error {symbol}: {e}")
            
    if results:
        res_df = pd.DataFrame(results)
        summary = res_df.groupby('Config').agg(
            Signals=('Signals', 'sum'),
            WinRate=('WinRate', 'mean'),
            AvgPnL=('AvgPnL', 'mean') 
        ).reset_index()
        
        print("\n=== ICHIMOKU OPTIMIZATION REPORT (4H / 90 Days) ===")
        print(summary.to_string(index=False))
        
        # Pick winner
        best = summary.sort_values('AvgPnL', ascending=False).iloc[0]
        print(f"\nWinner: {best['Config']}")

if __name__ == "__main__":
    run_report()

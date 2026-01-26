import os
import pandas as pd
import logging

class DataLoader:
    def __init__(self, archive_dir=None):
        if not archive_dir:
            # Default relative path
            # We are in .agent/skills/market-data-manager/scripts/
            # We need to go up 5 levels to get to project root? No.
            # 1. scripts
            # 2. market-data-manager
            # 3. skills
            # 4. .agent
            # 5. [Root]
            
            base = os.path.dirname(os.path.abspath(__file__)) 
            self.archive_dir = os.path.abspath(os.path.join(base, "../../../../data/parquet_archive"))
        else:
            self.archive_dir = archive_dir
            
    def load_data(self, symbol, timeframe='1d', asset_type='crypto'):
        """
        Loads data from local parquet archive.
        Auto-resamples 1m/1h base data to requested timeframe.
        """
        folder = "crypto" if asset_type == 'crypto' else "tradfi"
        safe_sym = symbol.replace("/", "_").replace(":", "_")
        
        # Determine Base Resolution
        # Crypto we store 1m
        # TradFi we store 1d (for this demo)
        base_tf = '1m' if asset_type == 'crypto' else '1d'
        
        filename = f"{safe_sym}_{base_tf}.parquet"
        path = os.path.join(self.archive_dir, folder, filename)
        
        if not os.path.exists(path):
            print(f"[Warn] No local data for {symbol}. Path: {path}")
            return pd.DataFrame()
            
        df = pd.read_parquet(path)
        
        # Resample if needed
        # Simple pandas resample map
        tf_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '1h': '1H', '4h': '4H', '12h': '12H',
            '1d': '1D', '4d': '4D'
        }
        target_rule = tf_map.get(timeframe)
        
        if target_rule and target_rule != base_tf:
            # Resample Logic
            logic = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            if 'close' in df.columns:
                df = df.resample(target_rule).agg(logic).dropna()
        
        return df

# Singleton for easy import
_loader = DataLoader()
def load_data(symbol, timeframe='1d', asset_type='crypto'):
    return _loader.load_data(symbol, timeframe, asset_type)

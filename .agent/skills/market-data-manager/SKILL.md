---
name: market-data-manager
description: Manages a local high-performance data warehouse using Parquet to eliminate API bottlenecks. Archives, synchronizes, and serves market data for backtesting.
---

# Market Data Manager (The Data Steward)

This skill creates a local "Source of Truth" for market data, decoupling your R&D workflow from slow/rate-limited APIs. By maintaining a local archive of 1-minute OHLCV data, you can run multi-year backtests in seconds.

## 🛠️ The Toolkit

This skill provides scripts to:
1.  **Initialize Archive (`init_archive.py`)**: Bulk-downloads historical data for all assets in your `strategy_config.json`.
2.  **Sync (`sync_data.py`)**: Incrementally updates the archive. Run this daily/weekly to keep data fresh without re-downloading history.
3.  **Serve (`data_loader.py`)**: A drop-in replacement for `data_fetcher.py` that checks the local cache first, then falls back to API.

## 📁 Data Structure
Data is stored in `data/parquet_archive/` with the following structure:
```
data/
└── parquet_archive/
    ├── crypto/
    │   ├── BTC_USDT.p_1m.parquet
    │   └── ETH_USDT.p_5m.parquet
    └── tradfi/
        ├── SPX_1d.parquet
        └── DXY_4h.parquet
```

## 🚀 Usage

### 1. Initialize the Archive
Download maximum available history for all your configured assets.
```bash
python .agent/skills/market-data-manager/scripts/init_archive.py
```

### 2. Update Data (Daily Routine)
Pull only the new candles since the last update.
```bash
python .agent/skills/market-data-manager/scripts/sync_data.py
```

### 3. Usage in Strategy Scripts
Instead of `data_fetcher.fetch_data`, use the loader:
```python
from market_data_manager.loader import load_data

# Loads instantly from disk (or fetches if missing)
df = load_data("BTC/USDT", timeframe="4h") 
```

## 🧬 Benefits
- **Speed**: Loading 5 years of Parquet data takes milliseconds vs. seconds/minutes for API.
- **Reliability**: No internet connection needed for backtesting. No API rate limits.
- **Consistency**: All backtests run on the exact same immutable dataset.

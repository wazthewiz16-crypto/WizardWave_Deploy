# WizardWave Integration Guide (for Wizard Portal)

This guide documents how to integrate the **WizardWave Signal Engine** into the **Wizard Portal** application. The goal is to enable the Portal to display "Runic Trade Alerts" by consuming the output of this engine.

## 1. Project Structure & Dependencies
The WizardWave engine relies on the following files in this directory:
- `strategy.py`: Core trading strategy logic (`WizardWaveStrategy`).
- `data_fetcher.py`: Unified data fetching (Crypto via CCXT, TradFi via yfinance).
- `pipeline.py`: Utilities for feature engineering (`calculate_ml_features`).
- `model.pkl`: The trained Random Forest classifier.
- `strategy_config.json`: Configuration for assets, timeframes, and T/SL parameters.

**Dependencies:**
- `pandas`, `pandas_ta`, `numpy`, `ccxt`, `yfinance`, `scikit-learn`, `joblib`

## 2. API Interface Abstraction
Currently, the signal logic is embedded in `app.py` (Streamlit). To integrate with Wizard Portal (likely a Flask/FastAPI/Next.js backend), you must extract the logic into a standalone python module (e.g., `runic_signals.py`).

### Proposed Function Signature
The Wizard Portal backend should call a function like this:

```python
def get_runic_alerts(timeframes=['1h', '4h', '1d']) -> List[Dict]:
    """
    Generates a list of active trade alerts (Runic Alerts) for the Wizard Portal.
    """
    # ... logic ...
```

### 3. Implementation Steps (For Antigravity)

**Step 1: Load Resources**
Load the configuration and model once on startup.
```python
import joblib
import json
from strategy import WizardWaveStrategy

# Load Config
with open('strategy_config.json', 'r') as f:
    config = json.load(f)

# Load Model
model = joblib.load('model.pkl')

# Initialize Strategy
strat = WizardWaveStrategy(
    lookback=config['lookback_candles'], 
    sensitivity=1, 
    cloud_spread=True, 
    zone_pad_pct=0.0
)
```

**Step 2: Replicate `process_asset` Logic**
Copy and adapt the `process_asset` function from `app.py`.
- **Input:** Asset Dictionary (`{'symbol': 'BTC/USDT', 'type': 'crypto', ...}`) & Timeframe.
- **Output:** Alert Dictionary (matches `runic_alerts_spec.md`).
- **Key Logic Preservation:**
    - Use `fetch_data` with a limit of ~600 candles.
    - Apply `strat.apply(df)`.
    - Apply `calculate_ml_features(df_strat)`.
    - `model.predict_proba` for confidence.
    - `strategy.get_active_trade(df)` to find entry signal.
    - **Crucial:** Calculate TP/SL using the asset-specific percentages (Crypto: 8%/5%, Trad: 3%/4%).

**Step 3: Parallel Execution**
Use `ThreadPoolExecutor` (as seen in `app.py`) to process the list of assets concurrently. This is vital for performance.

**Step 4: Return JSON-Serializable Data**
Ensure the output is a list of plain dictionaries (convert Pandas Timestamps to Strings) so it can be easily served via a REST API to the Wizard Portal frontend.

## 4. Configuration
Refer to `strategy_config.json` for the definitive list of artifacts (Assets) and their types ('crypto'/'trad'). This file should be the source of truth for the Portal's scanner.

## 5. Performance Notes
- **Caching:** The Portal should implement server-side caching (Redis or memory) for the alerts, refreshing every 15 minutes (or 5m for crypto), to avoid hitting API rate limits on every user page load.
- **Rate Limits:** `yfinance` and `binanceus` (via ccxt) have rate limits. Keep `max_workers` in ThreadPool around 10.

# WizardWave System Architecture & Logic

## 1. Data Pipeline Overview

The system operates on two parallel tracks: 
1. **Interactive Frontend (`app.py`)**: Runs locally in the User's browser session, fetching data on-demand or via background threads.
2. **Background Monitor (`monitor_signals.py`)**: Runs independently to send Discord alerts.

### 1.1 Signal Generation Process (Standard)
1.  **Data Fetching**: `fetch_data` retrieves OHLCV data from Yahoo Finance (Stocks/Forex) or CCXT (Crypto).
2.  **Strategy Application**:
    *   **HTF (High Timeframe, >1H)**: Uses `WizardWaveStrategy` (Cloud + Trend + RSI Filters).
    *   **LTF (Low Timeframe, <1H)**: Uses `WizardScalpStrategy` (Faster Cloud, Lookback=8).
3.  **Feature Engineering**: `calculate_ml_features` computes 10 technical indicators used by the ML model.
4.  **ML Inference**: The relevant pre-trained model (`htf` or `ltf`) predicts the probability of a successful trade.
5.  **Threshold Check**:
    *   **Frontend**: Confirms if Confidence > 40%.
    *   **Monitor**: Currently requires Confidence > 55% (Discrepancy).
6.  **Alerting**: If "TAKE" criteria are met, the signal is displayed or sent to Discord.

### 1.2 Identified Discrepancies (To Be Resolved)
Currently, `monitor_signals.py` and `app.py` differ in logic, leading to alerting mismatches:
*   **Thresholds**: `app.py` uses **40%**, while Monitor uses **55%**.
*   **12H Ensemble**: `app.py` uses a weighted ensemble (80% HTF + 20% LTF) for the 12H timeframe. The Monitor uses pure HTF logic.
*   **Dynamic Barriers**: `app.py` calculates dynamic volatility (Sigma) for crypto stops. The Monitor defaults to static barriers.

## 2. Machine Learning Models

The system uses two distinct pre-trained classifiers stored as `.pkl` files.

### 2.1 Model Specifications
*   **HTF Model (`model_htf.pkl`)**: Optimized for swing trades (4H, 12H, 1D).
*   **LTF Model (`model_ltf.pkl`)**: Optimized for scalping (15m, 1H).
*   **Algorithm**: RandomForestClassifier / GradientBoosting (based on `joblib` artifacts).

### 2.2 Input Features (10 Total)
1.  **Volatility**: 20-period rolling standard deviation of returns.
2.  **RSI**: Relative Strength Index (14).
3.  **MA Distance**: Distance of Close from 50 SMA.
4.  **ADX**: Trend Strength (14).
5.  **Momentum**: ROC (10).
6.  **RVOL**: Relative Volume (Volume / 20-SMA Volume).
7.  **BB Width**: Bollinger Band Width (Vol Squeeze metic).
8.  **Candle Body Ratio**: Abs(Open-Close) / (High-Low).
9.  **ATR %**: Normalized Average True Range.
10. **MFI**: Money Flow Index.

### 2.3 Ensemble Logic (12 Hour Special Case)
The 12-Hour timeframe is unique. It blends predictions to capture daily trend stability with hourly precision:
```python
Probability = (0.8 * HTF_Prob) + (0.2 * LTF_Prob)
```

## 3. Configuration Parameters

### 3.1 Strategies
*   **LTF Strategy**: Lookback = 8, Sensitivity = 1.0.
*   **HTF Strategy**: Lookback/Sensitivity defined in `strategy_config.json`.

### 3.2 Triple Barriers (Take Profit / Stop Loss)
Defined in `strategy_config.json`.
*   **Crypto**: Dynamic (based on Sigma) or Static (e.g., 1.5% TP / 0.5% SL).
*   **TradFi**: Static (e.g., 0.7% TP / 0.3% SL).
*   **Forex**: Static (e.g., 0.3% TP / 0.9% SL).

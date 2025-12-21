# 🧙‍♂️ WizardWave: AI-Powered Trading Signal System

**WizardWave** is a comprehensive, gamified trading dashboard that combines classic technical analysis with modern machine learning to generate, filter, and manage high-probability trading signals. Built with a fantasy-themed "Arcane Portal" interface, it transforms the dry task of monitoring markets into an engaging experience while enforcing strict risk management protocols.

Dashboard Preview
<img width="2011" height="868" alt="image" src="https://github.com/user-attachments/assets/f01385ba-7cfd-4cc0-b1e4-58a16b91319d" />

---

## ✨ Key Features

### 🔮 The Arcane Portal (Dashboard)
- **Dual-Engine Signal Feed**: Real-time display of active trade setups across Crypto, Forex, and Indices, split into High Timeframe (Trend) and Low Timeframe (Scalp) engines.
- **Runic Alerts**: A dedicated sidebar for quick-glance status updates on active signals across 15m, 1h, 4h, 12h, 1d, and 4d timeframes.
- **Realms (Market Sessions)**: A visual 24-hour timeline showing the overlap of major global trading sessions (Sydney, Tokyo, London, New York).
- **The Oracle**: An automated countdown to high-impact economic events (CPI, NFP, FOMC) to help you avoid volatility spikes.

### 🧠 AI Meta-Labeling
- **Smart Filtering**: Raw technical signals are passed through specialized Random Forest Classifiers.
- **Hybrid Confidence Scoring**:
  - **HTF Model**: Optimized for 1D and 4D trend following.
  - **LTF Model**: Optimized for 15M, 1H, and 4H scalping/swings.
  - **Ensemble Logic**: 12H signals use a weighted mix (80% HTF / 20% LTF) for high-conviction entries.
- **Meta-Labeling**: The system advises "TAKE" or "SKIP" based on whether the signal matches the characteristics of historically profitable trades.

### 🛡️ Risk Management Shield
- **Gamified Risk**: Capital is treated as "Mana". Taking trades costs Mana, limiting overtrading.
- **Prop Firm Tracking**: Native support for tracking multiple prop firm accounts, including drawdown limits and profit targets.
- **Position Sizing**: Automatic calculation of risk per trade based on account size and specific risk percentage (0.25%, 0.5%, 1.0%).

---

## ⚙️ How It Works: The Signal Pipeline

The system operates on a specialized dual-pipeline that transforms raw market data into actionable intelligence.

### 1. Data Ingestion (`data_fetcher.py`)
- **Crypto**: Fetches real-time OHLCV data using `ccxt` (BinanceUS).
- **Traditional**: Fetches Stocks, Forex, and Indices data using `yfinance`.
- **Resampling**: Handles custom timeframes like 12H and 4D via robust resampling logic.

### 2. Dual Strategy Logic
**A. High Timeframe (HTF) - `strategy.py`**:
- **Goal**: Capture major multi-day trends.
- **Logic**: "WizardWave" Trend Following.
- **Timeframes**: 12H, 1D, 4D.
- **Triggers**: Cloud Breakouts and Zone Pullbacks.

**B. Low Timeframe (LTF) - `strategy_scalp.py`**:
- **Goal**: Capture intraday swings and scalps.
- **Logic**: "WizardScalp" Momentum.
- **Timeframes**: 15M, 1H, 4H, 12H (secondary).
- **Triggers**: Cloud Crosses with ADX > 20 and EMA Trend Filtering.

### 3. Advanced Feature Engineering (`feature_engine.py`)
Signals are enriched with 8 distinct features before classification:
- **Volatility**: Rolling standard deviation.
- **RSI**: Relative Strength Index.
- **MA Distance**: Mean reversion potential.
- **ADX**: Trend strength intensity.
- **Momentum**: Rate of Change (ROC).
- **RVOL**: Relative Volume (Volume Conviction).
- **Bollinger Width**: Volatility Squeeze/Expansion state.
- **Candle Ratio**: Price action conviction (Body vs Range).

### 4. Meta-Labeling Classification
- **Two Specialized Models**: 
    - `model_htf.pkl`: Trained on thousands of daily/4-day signals.
    - `model_ltf.pkl`: Trained on faster intraday data.
- **Triple Barrier Method**: Models are trained using dynamic Profit Targets and Stop Losses tailored to the asset class (Crypto vs. TradFi).

---

## 🛠️ Installation & Setup

1. **Prerequisites**:
   - Python 3.8+
   - pip

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Retrain the Models (Optional)**:
   To update the ML models with the latest data:
   ```bash
   python pipeline.py
   ```

---

## 📂 Project Structure

- **`app.py`**: Main application logic (Dashboard, UI, Signal Inference).
- **`pipeline.py`**: Training pipeline for generating `model_htf.pkl` and `model_ltf.pkl`.
- **`strategy.py`**: HTF Strategy Logic (WizardWave).
- **`strategy_scalp.py`**: LTF Strategy Logic (WizardScalp).
- **`feature_engine.py`**: Shared library for calculating technical features.
- **`data_fetcher.py`**: Wrapper for data APIs.
- **`strategy_config.json`**: Central configuration for assets, strategies, and model parameters.

---

## 🧩 Technology Stack

- **Frontend**: Streamlit (Python)
- **Data Analysis**: Pandas, NumPy, Pandas-TA
- **Machine Learning**: Scikit-Learn (Random Forest Ensemble)
- **Data Feeds**: CCXT, yFinance

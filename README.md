# 🧙‍♂️ WizardWave: AI-Powered Trading Signal System

**WizardWave** is a comprehensive, gamified trading dashboard that combines classic technical analysis with modern machine learning to generate, filter, and manage high-probability trading signals. Built with a fantasy-themed "Arcane Portal" interface, it transforms the dry task of monitoring markets into an engaging experience while enforcing strict risk management protocols.

Dashboard Preview
<img width="2011" height="868" alt="image" src="https://github.com/user-attachments/assets/f01385ba-7cfd-4cc0-b1e4-58a16b91319d" />

### New Interfaces
**Active Runic Alerts**:
<img width="800" alt="runic_alerts_new" src="./documentation/runic_alerts_v2.png" />

**Signal History with NY Session Highlights**:
<img width="800" alt="signal_history_new" src="./documentation/signal_history_v2.png" />

---

## ✨ Key Features

### 🔮 The Arcane Portal (Dashboard)
- **Multi-Strategy Feed**: Real-time display of active trade setups across Crypto, Forex, and Indices, now supporting **three** distinct engines (WizardWave, WizardScalp, Daily CLS Range).
- **Runic Alerts V2**: A dedicated "Datapad" sidebar for live signal monitoring.
  - **Rich Cards**: Displays Entry, Price, Net PnL, TP/SL, and Strategy Type in a compact, readable card.
  - **Live Updates**: Open positions remain visible ("Active") to keep you in the loop until closed.
  - **Dynamic Styling**: Active signals display "Confidence" in high-visibility light orange.
- **Signal History**: A comprehensive log of all past signals.
  - **Strategy Filter**: Filter history by specific strategy (e.g., "Daily CLS Range" vs "WizardWave").
  - **Session Highlighting**: Trades executed during the New York Session (8 AM - 5 PM ET) are automatically highlighted.
- **Realms (Market Sessions)**: A visual 24-hour timeline showing the overlap of major global trading sessions.

### 🧠 Logic & Strategies
**A. WizardWave (HTF)**:
- **Goal**: Capture major multi-day trends (12H, 1D, 4D).
- **Type**: AI-Filtered Trend Following.

**B. WizardScalp (LTF)**:
- **Goal**: Capture intraday swings and scalps (15M, 1H, 4H).
- **Type**: AI-Filtered Momentum.

**C. Daily CLS Range (Mean Reversion)**:
- **Goal**: Identify "Continuated Liquidity Sweep" candles on the Daily timeframe.
- **Execution**: Precision execution on the 1H timeframe upon range reclaim.
- **Risk**: Rule-based (Non-AI) strategy with strict 2-target profit taking.
- **Safety**: Built-in logic prevents opening duplicate positions on the same asset.

### 🔔 Smart Alerts & Discord
- **Automated Monitoring**: Background service (`monitor_signals.py`) scans for signals 24/7.
- **Enhanced Notifications**: Discord alerts feature prettified asset names (e.g., `EUR/USD 🇪🇺`), explicit EST timestamps, and Confidence/R:R metrics.
- **Deduplication**: Smart filtering preventing alert spam for existing signals.

### 🛡️ Risk Management Shield
- **Gamified Risk**: Capital is treated as "Mana". Taking trades costs Mana.
- **Prop Firm Tracking**: Native support for tracking multiple prop firm accounts with visual progress bars for Profit Targets and Drawdown Limits.
- **Position Size Calculator**: **NEW!** Embedded calculator to instantly determine lot size/units based on Entry, Stop Loss, and Risk ($) amount.
- **Manual Mode**: Toggle to filter strict high-confidence (>60%) setups only.

---

## ⚙️ How It Works: The Signal Pipeline

The system operates on a specialized multi-pipeline that transforms raw market data into actionable intelligence.

### 1. Data Ingestion (`data_fetcher.py`)
- **Crypto**: Fetches real-time OHLCV data using `ccxt` (BinanceUS).
- **Traditional**: Fetches Stocks, Forex, and Indices data using `yfinance`.
- **Resampling**: Handles custom timeframes like 12H and 4D via robust resampling logic.

### 2. Strategy Logic
**A. High Timeframe (HTF) - `strategy.py`**:
- "WizardWave" Trend Following targeting Cloud Breakouts and Zone Pullbacks.

**B. Low Timeframe (LTF) - `strategy_scalp.py`**:
- "WizardScalp" Momentum targeting Cloud Crosses with ADX > 20 and EMA Trend Filtering.

**C. Range Reversion - `strategy_cls.py`**:
- "Daily CLS Range" identifying H/L sweeps of the previous candle body.

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
   *To start the background alert monitor:*
   ```bash
   python monitor_signals.py
   ```

4. **Retrain the Models (Optional)**:
   To update the ML models with the latest data:
   ```bash
   python pipeline.py
   ```

---

## 📂 Project Structure

- **`app.py`**: Main application logic (Dashboard, UI, Signal Inference).
- **`monitor_signals.py`**: Background service for generating Discord alerts.
- **`pipeline.py`**: Training pipeline for generating `model_htf.pkl` and `model_ltf.pkl`.
- **`strategy.py`**: HTF Strategy Logic (WizardWave).
- **`strategy_scalp.py`**: LTF Strategy Logic (WizardScalp).
- **`strategy_cls.py`**: Daily CLS Range Strategy Logic.
- **`feature_engine.py`**: Shared library for calculating technical features.
- **`data_fetcher.py`**: Wrapper for data APIs.
- **`strategy_config.json`**: Central configuration for assets, strategies, and model parameters.

---

## 🧩 Technology Stack

- **Frontend**: Streamlit (Python)
- **Data Analysis**: Pandas, NumPy, Pandas-TA
- **Machine Learning**: Scikit-Learn (Random Forest Ensemble)
- **Data Feeds**: CCXT, yFinance

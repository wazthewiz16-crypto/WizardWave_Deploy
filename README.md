# 🧙‍♂️ WizardWave: AI-Powered Trading Signal System

**WizardWave** is a comprehensive, gamified trading dashboard that combines classic technical analysis with modern machine learning to generate, filter, and manage high-probability trading signals. Built with a fantasy-themed "Arcane Portal" interface, it transforms the dry task of monitoring markets into an engaging experience while enforcing strict risk management protocols.

Dashboard Preview
<img width="2011" height="868" alt="image" src="https://github.com/user-attachments/assets/f01385ba-7cfd-4cc0-b1e4-58a16b91319d" />

---

## ✨ Key Features

### 🔮 The Arcane Portal (Dashboard)
- **Live Signal Feed**: Real-time display of active trade setups across Crypto, Forex, and Indices.
- **Runic Alerts**: A dedicated sidebar for quick-glance status updates on active signals.
- **Realms (Market Sessions)**: A visual 24-hour timeline showing the overlap of major global trading sessions (Sydney, Tokyo, London, New York).
- **The Oracle**: An automated countdown to high-impact economic events (CPI, NFP, FOMC) to help you avoid volatility spikes.

### 🧠 AI Meta-Labeling
- **Smart Filtering**: Raw technical signals are not blindly trusted. They are passed through a **Random Forest Classifier**.
- **Confidence Scoring**: Each signal is assigned a probability score (0-100%).
- **Meta-Labeling**: The system advises "TAKE" or "SKIP" based on whether the signal matches the characteristics of historically profitable trades.

### 🛡️ Risk Management Shield
- **Gamified Risk**: Capital is treated as "Mana". Taking trades costs Mana, limiting overtrading.
- **Prop Firm Tracking**: Native support for tracking multiple prop firm accounts, including drawdown limits and profit targets.
- **Position Sizing**: Automatic calculation of risk per trade based on account size and specific risk percentage (0.25%, 0.5%, 1.0%).

---

## ⚙️ How It Works: The Signal Pipeline

The system operates on a linear pipeline that transforms raw market data into actionable intelligence.

### 1. Data Ingestion (`data_fetcher.py`)
- **Crypto**: Fetches real-time OHLCV data using `ccxt` (BinanceUS).
- **Traditional**: Fetches Stocks, Forex, and Indices data using `yfinance`.
- **Standardization**: All data is normalized to a common format for processing.

### 2. Strategy Logic (`strategy.py`)
The core "WizardWave" strategy is a trend-following system based on dynamic cloud structures.
- **Trend Definition**: Uses two custom EMAs ("Mango D1" & "Mango D2") to form a "Cloud".
    - Price > Cloud = **Bullish**
    - Price < Cloud = **Bearish**
- **Signal Triggers**:
    - **Zone Entries**: Validated pullbacks into the "Bid Zone" (the area between the cloud bands) during a trend.
    - **Reversals**: "Trend Flip" events where price forcefully breaks through the cloud structure.

### 3. ML Feature Engineering (`pipeline.py`)
Before a signal is shown, it is enriched with secondary features:
- **Volatility**: Rolling standard deviation of returns.
- **RSI**: Relative Strength Index for overbought/oversold conditions.
- **MA Distance**: Distance from the 50-period SMA (mean reversion potential).
- **ADX**: Trend strength intensity.
- **Momentum**: Rate of Change (ROC).

### 4. Meta-Labeling Classification
- A pre-trained **Random Forest Model (`model.pkl`)** analyzes these features.
- It predicts the probability of the trade hitting its Profit Target (PT) before its Stop Loss (SL).
- **Triple Barrier Method**: The model was trained using a labeling method that considers time limits, profit targets, and stop losses to define "success."

---

## 🛠️ Installation & Setup

1. **Prerequisites**:
   - Python 3.8+
   - pip

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure `ccxt`, `yfinance`, `pandas`, `pandas_ta`, `streamlit`, `scikit-learn`, `matplotlib` are included)*

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. **Retrain the Model (Optional)**:
   To update the ML model with the latest data:
   ```bash
   python pipeline.py
   ```

---

## 📂 Project Structure

- **`app.py`**: The main Streamlit application entry point. Handles UI, session state, and user interaction.
- **`strategy.py`**: Contains the `WizardWaveStrategy` class with the core technical logic.
- **`pipeline.py`**: The end-to-end Machine Learning pipeline (Data -> Features -> Training -> Backtesting).
- **`data_fetcher.py`**: Wrapper for `ccxt` and `yfinance` to handle multi-asset data fetching.
- **`model.pkl`**: The serialized trained Random Forest model.
- **`strategy_config.json`**: Configuration for assets, timeframes, and model parameters.

---

## 🧩 Technology Stack

- **Frontend**: Streamlit (Python)
- **Data Analysis**: Pandas, NumPy, Pandas-TA
- **Machine Learning**: Scikit-Learn (Random Forest)
- **Data Feeds**: CCXT, yFinance
- **Visualization**: Matplotlib (Backtests), Streamlit Native Charts


## How to Host 24/7 on Streamlit Cloud

This application is ready to be hosted on **Streamlit Cloud** (Free Community Tier).

### Step 1: Push to GitHub
1.  Create a **New Repository** on GitHub (e.g., `wizard-wave-signals`).
2.  Open a terminal in this folder (`WizardWave_Deploy`).
3.  Run the following commands:
    ```bash
    git init
    git add .
    git commit -m "Initial deploy"
    git branch -M main
    git remote add origin https://github.com/YOUR_USERNAME/wizard-wave-signals.git
    git push -u origin main
    ```

### Step 2: Deploy on Streamlit
1.  Go to [share.streamlit.io](https://share.streamlit.io/).
2.  Click **"New app"**.
3.  Select your GitHub repository (`wizard-wave-signals`).
4.  Set **Main file path** to `app.py`.
5.  Click **"Deploy"**.

### Notes
- **Model File**: The `model.pkl` is included. Streamlit Cloud has a limit of ~1GB, so typical models work fine.
- **Dependencies**: The `requirements.txt` file ensures all libraries (`pandas`, `pandas_ta`, `sklearn`, `ccxt`, `yfinance`) are installed automatically.
- **24/7 Uptime**: Streamlit Cloud will keep the app running. If it goes to sleep after inactivity, simply visiting the URL wakes it up.

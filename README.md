# 🧙‍♂️ WizardWave: AI-Powered Trading Signal System

**WizardWave** is a comprehensive, gamified trading dashboard that combines classic technical analysis, real-time proprietary indicator data, and modern machine learning to generate, filter, and manage high-probability trading signals. Built with a fantasy-themed "Arcane Portal" interface, it transforms the dry task of monitoring markets into an engaging experience while enforcing strict risk management protocols.

![WizardWave Dashboard](https://github.com/user-attachments/assets/f01385ba-7cfd-4cc0-b1e4-58a16b91319d)

---

## ✨ Key Features

### 🔮 The Arcane Portal (Dashboard)
- **Multi-Strategy Feed**: Real-time display of active trade setups across Crypto, Forex, and Indices.
- **Fractal Alignment (NEW)**: Real-time "Trend" and "Bid Zone" confirmation scraped directly from your TradingView **Mango Dynamic** indicators using a headless browser engine.
- **Runic Alerts**: A dedicated "Datapad" sidebar for live signal monitoring with "Confidence" scores.
- **Session Highlighting**: Trades executed during the New York Session (8 AM - 5 PM ET) are automatically highlighted.
- **Visual Realism**: Matches your actual charting environment by targeting **Perpetual Contract** (.P) charts for crypto assets.

### 🧠 Logic & Strategies
1. **WizardWave (Trend)**: Captures major multi-day trends. 
   - **Pro-Max Edition (NEW)**: Active on 12H timeframes. Uses Dynamic ATR Trailing Stops, Rising ADX confirmation, and RSI overextension filters for high-conviction entries.
2. **WizardScalp (Momentum)**: Intraday swings. Currently paused on 1h/15m timeframes to prioritize higher-timeframe alpha.
3. **Daily CLS Range**: Identifies "Continuated Liquidity Sweeps" on the Daily timeframe for precision 1H entries.
4. **Ichimoku Cloud**: Traditional cloud breakout signals.
   - **Kumo King Upgrade (NEW)**: Deployed for Crypto and Metals. Uses **Cloud Thickness Filters** to avoid noise and **Kijun Rejection** logic to capture trend pullbacks with high precision.
5. **Monday Range**: Automates the "Monday High/Low" deviation strategy.

### 🛡️ Risk Management Schema
- **Gamified "Mana" System**: Capital is treated as "Mana". Taking trades costs Mana based on risk size.
- **Prop Firm Tracker**: Native support for tracking Drawdown Limits and Profit Targets.
- **Triple Barrier ML Labeling**: Models are trained using Fixed Time Horizons, Profit Targets, and Stop Losses to determine "True" signal quality.
- **Dynamic Risk Adjustment**: Pro-Max models automatically adjust Stop Loss placement based on real-time market volatility (ATR).

---

## ⚙️ The Data Pipeline (Architecture)

The system uses a **Dual-Pipeline** approach to ensure signal accuracy:

### 1. Market Data Pipeline (Python/Pandas)
*   **Source**: `ccxt` (Crypto) and `yfinance` (TradFi).
*   **Processing**: `data_fetcher.py` pulls raw OHLCV data.
*   **Feature Engineering**: `feature_engine.py` calculates 12+ features (RSI, Volatility, MFI, Cycle Regime, etc.).
*   **Inference**: `app.py` loads specific ML models (4h, 12h, 1d, 4d) to predict signal probability.
*   **Hybrid Execution**: Rule-based "Pro-Max" and Ichimoku logic works alongside ML models for multi-layered validation.

### 2. Indicator Verification Pipeline (Playwright Scraper)
*   **Source**: Your actual **TradingView** charts (Headless Browser).
*   **Agent**: `scrape_tv_indicators.py`.
*   **Mechanism**:
    1.  Launches a headless Chromium browser.
    2.  Navigates to your specific chart layout (`qR1XTue9`).
    3.  **Maximizes** the chart (Alt+Enter) to isolate data.
    4.  **Hovers** over the latest candle to ensure "Data Window" freshness.
    5.  **Extracts** "Trend" (Bullish/Bearish) and "Bid Zone" (Yes/No) using robust Regex parsing.
*   **Output**: Saves to `mango_dynamic_data.json`, which the Dashboard reads to populate the "Fractal Alignment" table.
*   **Self-Healing**: If data is missing or ambiguous, it toggles the Data Window (`Alt+D`) or takes a **Debug Screenshot** (`debug_view_*.png`) for diagnosis.

### 3. Execution (The App)
*   **Streamlit**: The frontend combines these two data streams.
*   **Background Monitor**: `monitor_signals.py` runs as a daemon to send Discord alerts even when the UI is closed.

---

## 🛠️ Installation & Setup

1. **Prerequisites**:
   - Python 3.10+
   - Node.js (for Playwright dependencies)

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Install Browser Engines**:
   Required for the TradingView scraper to work:
   ```bash
   playwright install chromium
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```
   *The Scraper and Monitor will auto-start in the background.*

---

## 📂 Project Structure

- **`app.py`**: The "Arcane Portal" Dashboard UI.
- **`scrape_tv_indicators.py`**: **(NEW)** The autonomous agent that watches your TradingView charts.
- **`monitor_signals.py`**: Background daemon that sends Discord alerts.
- **`pipeline.py`**: Machine Learning training pipeline (Triple Barrier Method).
- **`feature_engine.py`**: Centralized library for all technical indicators and ML features.
- **`user_grimoire.json`**: Persists your "Mana" (Risk) and "Spells" (Trade Limits).
- **`mango_dynamic_data.json`**: The live state of your proprietary indicators (shared between Scraper and App).

---

## 🔧 Troubleshooting

### "Fractal Alignment" Data is Inaccurate
If the "Trend" or "Bid Zone" on the dashboard doesn't match your TradingView chart:
1.  **Check Screenshots**: Look in the project folder for files named `debug_view_[ASSET]_[TIMEFRAME].png`.
2.  **Verify Layout**: These images show *exactly* what the bot sees.
    *   If the bot sees a different chart than you, ensure you are using the same Layout ID (`qR1XTue9`).
    *   If the bot sees "Spot" data but you trade "Perps", the code now defaults to `.P` (Perpetual) tickers for crypto.
3.  **Data Window**: Ensure the "Data Window" on your TradingView layout is enabled and includes the "Mango Dynamic" values.

### Scraper Logs
Check `scraper_debug.log` for detailed execution traces.
```bash
tail -f scraper_debug.log
```

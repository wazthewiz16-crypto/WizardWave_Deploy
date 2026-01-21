# 🚀 ML Wizard Signals: Future Roadmap

This document keeps track of planned improvements and experimental features to transform the Arcane Portal into a hedge-fund-grade trading system.

## 🔭 Vision
To move from basic technical indicators to a **Multi-Factor Quantitative System** that combines Macro, Microstructure, and Advanced Machine Learning.

---

## 🛠️ Planned Features

### 1. 🌐 The "Shadow" Macro Filter (Active)
*   **Concept**: Use cross-asset correlations (inter-market analysis) to filter signals.
*   **Targets**: 
    *   **DXY (Dollar Index)**: Inverse correlation with Gold, Nasdaq, and Crypto.
    *   **USDT.D (Tether Dominance)**: Inverse correlation with the Crypto market.
*   **Status**: 🏗️ In Progress

### 2. 🎭 Market Regime Detection (HMM)
*   **Concept**: Use Hidden Markov Models to detect if the market is in a "Trending," "Chippy," or "Crash" regime.
*   **Goal**: Automatically switch between Trend-Following (Wizard Wave) and Mean-Reversion strategies.

### 3. 📊 Institutional Flow (Order Flow / Volume Profile)
*   **Concept**: Calculate Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
*   **Goal**: Filter trades that are entering "High Volume Nodes" (resistance) vs "Low Volume Nodes" (fast moves).

### 4. 🧠 Meta-Labeling & Kelly Bet Sizing
*   **Concept**: Train a second ML model to predict the *confidence* of the first model's signal.
*   **Goal**: Move from fixed 0.3 exposure to dynamic "High Conviction" vs "Low Conviction" position sizing using the Kelly Criterion.

### 🦅 5. Fractal Alignment Score
*   **Concept**: A score showing how many timeframes (1H, 4H, 1D) are aligned in the same direction.
*   **UI**: A "Traffic Light" indicator for each asset.

---

## 🧪 Experimental Ideas
*   **Twitter/X Sentiment Sentiment Analysis**: Feed localized crypto sentiment scores into the ML.
*   **Liquidation Heatmaps**: Estimate where high-leverage liquidations are sitting to predict "Short Squeezes."
*   **Fractional Differentiation Tuning**: Testing different `d` values for specific asset classes (e.g., d=0.3 for Forex vs d=0.5 for Crypto).

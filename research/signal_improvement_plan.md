# Research & Improvement Plan for Signal Quality

## 1. Executive Summary
The current signal generation system uses a trend-following strategy ("WizardWave") filtered by a Random Forest meta-model. While functional, the signals are "good but need tweaking."
This document outlines research findings and actionable improvements to enhance signal precision (win rate) and robustness (profitability), focusing initially on lower-risk, high-impact changes.

**Core Findings:**
1.  **Low Feature Count:** The ML model uses only 5 basic technical features, limiting its ability to discern complex market regimes.
2.  **Rigid Labeling Logic:** The Triple Barrier Method uses fixed Profit Take (PT) and Stop Loss (SL) percentages for all assets within a class (Crypto vs. Trad), ignoring individual asset volatility.
3.  **Time Limit Noise:** The "Vertical Barrier" (time limit) for labeling is likely too long for hourly signals, leading to noisy "drift" labels rather than true signal validation.
4.  **Static Logic:** Key parameters (thresholds, lookbacks) are hardcoded across `pipeline.py`, `strategy.py`, and `app.py`, making optimization difficult.

---

## 2. Proposed Improvements

### Phase 1: High Impact, Low Effort (Immediate)

#### A. Volatility-Based Dynamic Labeling (ATR)
Instead of fixed PT (8%) / SL (5%) for all crypto, use Average True Range (ATR) multiples.
*   **Why:** A stable asset like BTC might never hit an 8% target in a reasonable time, while a volatile meme coin might hit it in minutes. Fixed targets penalize stable assets.
*   **Action:**
    *   Calculate `ATR(14)` at the time of signal entry.
    *   Set `PT = Entry + (Multiplier * ATR)` and `SL = Entry - (Multiplier * ATR)`.
    *   Typical Multipliers: 2.0 to 3.0 for PT, 1.0 to 1.5 for SL.

#### B. Optimize Time Barriers
The current logic converts `time_limit_days` (40) into bars. For 1-hour candles, this is 960 hours (40 days), which is far too long for a swing trade signal.
*   **Why:** If a trade takes 40 days to resolve, it's not a response to the signal anymore; it's market drift.
*   **Action:** Reduce time barriers significantly.
    *   `1h` timeframe -> Max 24-48 hours (24-48 bars).
    *   `4h` timeframe -> Max 3-5 days (18-30 bars).
    *   If the price hasn't moved by then, the signal was likely a "false alarm" (choppy market). Validate this by labeling "timeout" as 0 (for filter) or Neutral.

#### C. Feature Expansion
Add 3-5 high-value features to give the model more context.
*   **Volume:** `Volume Oscillator` or `Relative Volume (RVOL)` (Is the move supported by volume?).
*   **Trend Strength:** `ADX` is already there, but `MACD Histogram` or `Ichimoku Cloud Distance` adds nuance.
*   **Volatility State:** `Bollinger Band Width` (Is the market squeezing or expanding?).
*   **Candle Patterns:** `Body / Total Range` ratio (Indecision vs. Conviction candles).

---

### Phase 2: Medium Term (Robustness)

#### D. Hyperparameter Tuning
The Random Forest is using default `n_estimators=100` and `max_depth=5`.
*   **Action:** Run a Grid Search or Bayesian Optimization to find optimal depth and tree count. `max_depth=5` might be underfitting; increasing it slightly (e.g., 8-12) might capture better non-linear patterns without overfitting if sample size allows.

#### E. Codebase Unification
`app.py` re-implements logic found in `pipeline.py` (e.g., `calculate_ml_features`, `simulate_history_stateful`).
*   **Risk:** Optimizing `pipeline.py` without updating `app.py` leads to **Model Drift** (Backtest says X, App shows Y).
*   **Action:** Refactor `calc_features` and active trade logic into a shared module (e.g., `utils.py`) imported by both.

---

## 3. Implementation Plan

**Step 1: Feature Importance Analysis (Validation)**
Before changing the model, we must know what currently works.
*   Run a script to print feature importance of the current 5 features.
*   *Expected output:* If one feature dominates completely, the model might be over-reliant on it.

**Step 2: Implement ATR Labeling**
Modify `pipeline.py`'s `apply_triple_barrier` function.
*   Add `atr` calculation.
*   Replace fixed `pt_pct`/`sl_pct` with partial config override or ATR logic.

**Step 3: Refine Time Limits**
Update `strategy_config.json` to have per-timeframe time limits, not a global "40 days".

**Step 4: Retrain & Compare**
*   Train New Model vs. Old Model.
*   Compare "Precision" (Win Rate of taken trades) and "Profit Factor" (Gross Profit / Gross Loss).

---

## 4. Current Configuration Notes (Reference)
*   **Features:** Volatility (20), RSI (14), MA Dist (50), ADX (14), Momentum (10).
*   **Strategy:** "WizardWave" (Dual EMA Cloud + Bid Zones + Reversals).
*   **Labeling:**
    *   Crypto: PT 8%, SL 5%.
    *   Trad: PT 3%, SL 4%.
*   **Model:** Random Forest (Balanced, Depth 5).

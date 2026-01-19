---
name: Prop Backtest Engine
description: Run rigorous simulations to check for Prop Firm Violations (Daily/Total Drawdown).
---

# Prop Backtest Engine Skill

This skill allows you to run high-fidelity backtests designed specifically for Prop Firm risk management standards. 

## Usage

### Parameters
- **Strategy Config**: `strategy_v2.json` (default)
- **Date Range**: Start and End dates for the simulation.

### How to Run
```bash
python .agent/skills/prop-backtest-engine/scripts/backtest_validator.py --config strategy_v2.json --start 2024-01-01 --end 2024-12-31
```

## Logic Description

1. **Simulation**: Execute the strategy using `backtesting.py` or `vectorbt`.
2. **Prop Firm Violation Monitoring**:
   - **Max Daily Drawdown**: Scans every 24-hour period (00:00 to 00:00). If Equity drops > 3.9% from the day's starting peak, it's a breach.
   - **Max Total Drawdown**: Monitoring the absolute peak-to-trough decline.
   - **Performance Filtering**:
     - `Profit Factor` must be > 1.3.
     - `Win Rate` and `Expected Value` are calculated per trade.
3. **Thresholds**:
   - `FAIL - PROP BREACH`: If daily drawdown exceeds 3.9% (buffer for 4% hard limit).
   - `FAIL - UNPROFITABLE`: If Profit Factor < 1.3 or Net PnL is negative.
   - `PASS`: Strategy meets all Prop Firm and Profitability requirements.

## Output
A detailed report (`backtest_report.pdf` or console log) showing the drawdown curve and the "Pass/Fail" status.

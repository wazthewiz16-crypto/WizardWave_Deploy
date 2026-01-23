---
name: Model Performance Reporter
description: Generates a comprehensive performance report for the live trading models over the last 180 days.
---

# Model Performance Reporter

This skill generates a detailed "Health Report" for your ML-driven trading strategy. It runs a simulation over the last 180 days using the *current* live models and strategy logic to tell you exactly how they are performing.

## Metrics Included
- **Financial**: Total PnL, Win Rate, Profit Factor.
- **Risk**: Max Drawdown, Consecutive Losses.
- **Timing**: Average Holding Time (Winners vs Losers).
- **ML Insights**: Average Confidence Score (Winners vs Losers) - verify if the model is confident about the right things.
- **Breakdowns**: Performance by Asset Class and Timeframe.

## Usage

Run the report generator:

```bash
python .agent/skills/model-performance-reporter/scripts/generate_report.py
```

## Output
The script will generate:
1.  **Console Summary**: A quick overview printed to the terminal.
2.  **`model_report_180d.md`**: A detailed Markdown report file in the project root, suitable for reading or sharing.

---
name: Walk Forward Optimization
description: Implements Walk Forward Optimization (WFO) to test trading reliability.
---

# Walk Forward Optimization (WFO) Skill

This skill allows you to perform Walk Forward Optimization to validate model performance over rolling time windows.

## Usage

Refers to the script `scripts/wfo_engine.py`.

### How to Run

```bash
python .agent/skills/walk_forward_optimization/scripts/wfo_engine.py
```

### Configuration

The script is hardcoded for the current user experiment:
- **Training Window**: 6 Months
- **Testing Window**: 1 Month
- **Start Date**: Jan 1, 2022
- **Iterations**: 2 (Test July 2022, Test August 2022)

### Dependencies
- `pandas`
- `yfinance`
- `sklearn`
- `strategy.py` (Rule-based Logic)
- `feature_engine.py` (ML Features)

## Logic Flow
1. Load historical data for assets.
2. Window 1:
   - Train: Jan 2022 - Jun 2022.
   - Test: July 2022.
3. Window 2:
   - Train: Feb 2022 - July 2022.
   - Test: Aug 2022.
4. Output cumulative PnL and Metrics.

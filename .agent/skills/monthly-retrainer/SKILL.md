---
name: Monthly Model Retrainer
description: Automates monthly retraining of ML models using a rolling window to adapt to current market volatility.
---

# Monthly Model Retrainer Skill

This skill ensures that the WizardWave ML models are never "set and forget". It implements a rolling window training strategy:
1.  **Window**: Training is performed on the last 24 months of data.
2.  **Frequency**: Run once a month.
3.  **Adaptation**: By dropping the oldest month and adding the newest, the model automatically adjusts its decision trees to current volatility (e.g., bull vs. bear cycles).

## Usage

Run the following command to refresh your models with the latest data:

```bash
python .agent/skills/monthly-retrainer/scripts/retrain_models.py
```

## How it Works
1.  **Data Fetching**: Pulls historical data for all assets defined in `strategy_config.json`.
2.  **Rolling Window**: Automatically determines the 24-month training range ending "today".
3.  **Target Labeling**: Uses the production "Triple Barrier" logic (9% TP / 3% SL for Crypto).
4.  **Training**: Re-trains the Random Forest models with production parameters (200 Trees, Depth 10).
5.  **Deployment**: Overwrites `model_1d.pkl` and `model_1h.pkl` in the project root.

## Scheduling (Automation)
To truly automate this, set up a cron job (Linux) or Task Scheduler (Windows) to run this script on the 1st of every month.

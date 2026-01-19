---
name: ML Model Challenger
description: Experiment with different ML architectures to beat the current Champion model.
---

# ML Model Challenger Skill

This skill automates the R&D process of testing new machine learning architectures against our current "Champion" model to improve signal quality.

## Usage

### Parameters
- **Challengers**: A list of models to test (e.g., `["XGBoost", "RandomForest", "LSTM", "LogisticRegression"]`).
- **Data Period**: Fixed range for training and validation.

### How to Run
```bash
python .agent/skills/model-challenger/scripts/champion_vs_challenger.py --models XGBoost RandomForest --train_start 2020-01-01 --train_end 2023-12-31 --val_start 2024-01-01
```

## Logic Description

1. **Uniform Data Preprocessing**: All models are trained on the exact same feature set and time period (`2020-2023`).
2. **Evaluation Metric (Precision Focus)**:
   - We prioritize **Precision** (Signal Accuracy) over Recall (Signal Frequency).
   - "Winning Signal" Accuracy: What percentage of "Buy" or "Sell" signals actually hit Target Profit before Stop Loss?
3. **Leaderboard Mechanism**:
   - Compare Challenger metrics vs. Current Champion.
   - Rank models by Precision and Profitability.
4. **Recommendation**:
   - Recommends a switch only if improvement is > 2% in Precision with stable Win Rates.

## Output
A leaderboard ranking models and a recommendation string: `"Switch to [Model X] - +5% Precision improvement."`

---
name: strategy-builder
description: Automates the creation of new trading strategies. Generates standardized Python strategy classes from plain English descriptions, complete with config registration and unit tests.
---

# Strategy Builder (The Strategy Forge)

This skill streamlines the process of converting a trade idea into executable code. It handles the boilerplate of class inheritance, method signatures, and configuration updates, allowing you to focus on the *logic*.

## 🛠️ The Toolkit

1.  **Generate (`generate_strategy.py`)**: Takes a strategy name and description, and outputs a `strategy_[name].py` file.
2.  **Test (`test_strategy.py`)**: Runs a quick backtest on the generated strategy using cached data (via Data Steward) to verify no syntax errors.

## 🚀 Usage

### 1. Generate a New Strategy
```bash
python .agent/skills/strategy-builder/scripts/generate_strategy.py "RSI_Divergence" "Buy when RSI < 30 and Price > EMA 200. Sell when RSI > 70."
```

### 2. Output
The tool will:
- Create `strategy_rsi_divergence.py`.
- Add "RSI_Divergence" to `strategy_config.json`.
- Run a sanity check.

## 🧪 Template

The generator uses a standard template compatible with `WizardWave` structure:

```python
class StrategyName:
    def __init__(self, params...):
        ...
        
    def apply(self, df):
        # 1. Indicators
        # 2. Entry Logic
        # 3. Exit Logic
        return df
```

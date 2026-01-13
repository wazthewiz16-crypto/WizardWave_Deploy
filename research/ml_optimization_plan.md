# ML Optimization Suite - Implementation Plan

## Objective
Enhance the accuracy and robustness of the WizardWave ML models by implementing advanced techniques:
1.  **Feature Engineering**: Add "Cloud Distance" to measure overextension.
2.  **Regime Filtering**: Train separate models for Bull/Bear market regimes.
3.  **Hyperparameter Tuning**: Optimize Random Forest parameters via Grid Search.
4.  **Retraining**: Update models using the most recent 6-12 months of high-relevance data.

## Status

### 1. Feature Engineering: Cloud Distance ☁️
- [ ] Modify `feature_engine.py`
- [ ] Calculate `dist_to_cloud = (close - cloud_top) / close` (if Long) or distance to bottom (if Short).
- [ ] Handle potential NaNs.

### 2. Regime Filtering 🌊
- [ ] Determine Regime: `Price > EMA 200` = Bull, `Price < EMA 200` = Bear.
- [ ] Update `pipeline.py` / Training script to split dataset.
- [ ] Train `model_bull` and `model_bear` separately?
    - *Decision*: To keep architecture simple for the App deploy, we might train ONE model but add `regime_bull` (0 or 1) as a heavily weighted feature, OR train separate physical models. 
    - *Alternative*: Train separate models and save them. App needs to load them dynamically.
    - *Refined Approach for this sprint*: Let's start by adding `regime_state` as a **Feature**. Validating if the Tree can split on it effectively (it should). If performance doesn't improve, we split the models. This avoids major refactoring of `app.py` logic for now.

### 3. Hyperparameter Tuning 🧠
- [ ] Create `tune_hyperparams.py`.
- [ ] Use `sklearn.model_selection.GridSearchCV`.
- [ ] Params to tune: `n_estimators`, `max_depth`, `min_samples_split`, `class_weight`.

### 4. Retraining w/ Recent Data 🔄
- [ ] Create `train_optimized_models.py`.
- [ ] Logic:
    1. Fetch last 365 Days.
    2. Apply Strategy.
    3. Calc Features (incl. Cloud Dist & Regime).
    4. Run Grid Search (optional flag).
    5. Train & Export `.pkl` files.

## Execution Log
- Branch Created: `feature/ml-optimization-suite`

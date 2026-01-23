import joblib
import os

models = [
    'model_1d.pkl',
    'model_1h.pkl',
    'wizard_wave_ml_model.pkl',
    'wizard_scalp_ml_model.pkl'
]

for m in models:
    if os.path.exists(m):
        try:
            clf = joblib.load(m)
            if isinstance(clf, dict) and 'model' in clf:
                clf = clf['model']
            
            # Check for feature names if available (scikit-learn >= 1.0)
            if hasattr(clf, 'feature_names_in_'):
                print(f"{m} (n={clf.n_features_in_}) features: {list(clf.feature_names_in_)}")
            else:
                print(f"{m} (n={clf.n_features_in_}) features: unknown")
        except Exception as e:
            print(f"{m}: Error {e}")

import joblib
import os

models = [
    'model_1d.pkl',
    'model_ichimoku.pkl',
    'north_star_ml_model.pkl',
    'wizard_scalp_ml_model.pkl'
]

for m in models:
    if os.path.exists(m):
        clf = joblib.load(m)
        if isinstance(clf, dict) and 'model' in clf:
            clf = clf['model']
        
        try:
            # Check for feature names if available (scikit-learn >= 1.0)
            if hasattr(clf, 'feature_names_in_'):
                print(f"{m} features: {list(clf.feature_names_in_)}")
            else:
                print(f"{m} features: unknown (n_features={clf.n_features_in_})")
        except:
            print(f"{m}: Could not determine features")

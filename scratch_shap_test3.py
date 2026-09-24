import pandas as pd
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings

# Create dummy data
df = pd.DataFrame({
    'cat': ['A', 'B', 'A', 'C', 'B', 'A'],
    'num': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'target': [0, 1, 0, 1, 1, 0]
})

X = df[['cat', 'num']]
y = df['target']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), ['cat']),
    ('num', 'passthrough', ['num'])
])
model = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', RandomForestClassifier(random_state=42))
])

model.fit(X, y)

def predict_fn(X_eval):
    # SHAP will pass a numpy array of objects if there are mixed types.
    if isinstance(X_eval, np.ndarray):
        X_eval = pd.DataFrame(X_eval, columns=X.columns)
    return model.predict_proba(X_eval)

try:
    print("Trying Explainer instead of PermutationExplainer directly...")
    # Sometimes shap.Explainer wraps things correctly
    explainer = shap.Explainer(predict_fn, X, algorithm='permutation')
    shap_values = explainer(X.iloc[:2])
    print("SUCCESS with shap.Explainer!")
except Exception as e:
    import traceback
    print("FAILED with shap.Explainer!")
    traceback.print_exc()

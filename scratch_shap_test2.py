import pandas as pd
import numpy as np
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

df = pd.DataFrame({
    'cat': ['A', 'B', 'A', 'C', 'B', 'A'],
    'num': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'target': [0, 1, 0, 1, 1, 0]
})

df['cat'] = df['cat'].astype('category')
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
    if isinstance(X_eval, np.ndarray):
        X_eval = pd.DataFrame(X_eval, columns=X.columns)
        X_eval['cat'] = X_eval['cat'].astype('category')
    return model.predict_proba(X_eval)

try:
    print("Trying PermutationExplainer with categorical types...")
    masker = shap.maskers.Independent(X, max_samples=100)
    explainer = shap.PermutationExplainer(predict_fn, masker)
    shap_values = explainer(X.iloc[:2])
    print("SUCCESS!")
except Exception as e:
    import traceback
    print("FAILED!")
    traceback.print_exc()

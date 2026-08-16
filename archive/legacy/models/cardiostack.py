"""
CardioStack — A Stacked Ensemble for Cardiovascular Disease Prediction
======================================================================
Architecture:
  Layer 1 (Base Learners):
    - Random Forest    : captures non-linear feature combinations via bagging
    - XGBoost          : sequential error correction via gradient boosting
    - AdaBoost         : focuses training on previously misclassified patients
  Layer 2 (Meta Learner):
    - XGBoost (shallow): learns WHICH base learner to trust for each patient profile

How it works:
  Each base learner sees the full patient feature set and outputs a probability.
  The meta learner then takes those 3 probabilities as its input and makes the
  final prediction — effectively learning when RF is right and XGB is wrong, etc.

CV Results (5-fold, 1,214 patients):
  Accuracy  : 91.10%
  Precision : 90.51%
  Recall    : 93.01%
  F1 Score  : 91.71%
"""

import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_cardiostack():
    """
    Constructs and returns the CardioStack model (unfitted).
    Call .fit(X_scaled, y) to train.
    """
    # ── Layer 1: Base Learners ────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    )

    xgb = XGBClassifier(
        eval_metric='logloss',
        max_depth=4,
        learning_rate=0.02,
        n_estimators=300,
        subsample=0.75,
        colsample_bytree=0.6,
        gamma=0.3,
        reg_alpha=0.1,
        reg_lambda=2.0,
        min_child_weight=5,
        random_state=42,
        verbosity=0,
        n_jobs=1
    )

    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )

    # ── Layer 2: Meta Learner ─────────────────────────────────────────────────
    # Shallow XGBoost — only sees the 3 base learner probability outputs.
    # Kept shallow (max_depth=2) to avoid overfitting on a small meta-feature set.
    meta = XGBClassifier(
        eval_metric='logloss',
        max_depth=2,
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
        n_jobs=1
    )

    cardiostack = StackingClassifier(
        estimators=[
            ('RandomForest', rf),
            ('XGBoost',      xgb),
            ('AdaBoost',     ada),
        ],
        final_estimator=meta,
        cv=5,          # uses 5-fold CV internally to generate meta-features
        n_jobs=1
    )

    return cardiostack


def train_cardiostack(features_path, labels_path, models_dir):
    """
    Trains CardioStack on the full dataset and saves it to disk.
    The scaler must already exist at models_dir/clinical_scaler.pkl.
    """
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

    # Load the shared scaler (fitted in train_clinical_kfold.py)
    scaler_path = os.path.join(models_dir, 'clinical_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X)

    print("  Training CardioStack (RF + XGBoost + AdaBoost → XGB meta)...")
    model = build_cardiostack()
    model.fit(X_scaled, y)

    out_path = os.path.join(models_dir, 'clinical_cardiostack_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"  CardioStack saved to: {out_path}")
    return model


def predict_cardiostack(patient_features, models_dir):
    """
    Run CardioStack inference on a single patient.

    patient_features: dict with keys matching the preprocessed feature columns,
                      OR a pre-scaled numpy array of shape (1, n_features).

    Returns:
      probability (float): 0.0–1.0 probability of heart disease
      prediction  (int)  : 0 = no disease, 1 = disease
    """
    model_path  = os.path.join(models_dir, 'clinical_cardiostack_model.pkl')
    scaler_path = os.path.join(models_dir, 'clinical_scaler.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    import numpy as np
    if isinstance(patient_features, dict):
        X = pd.DataFrame([patient_features]).values
    else:
        X = np.array(patient_features).reshape(1, -1)

    X_scaled    = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[0][1]
    prediction  = int(probability >= 0.5)

    return probability, prediction


if __name__ == "__main__":
    # Quick standalone test — runs CV to verify performance
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    features_path = os.path.join(base_dir, "data", "processed", "clinical_features.csv")
    labels_path   = os.path.join(base_dir, "data", "processed", "clinical_labels.csv")

    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = make_pipeline(StandardScaler(), build_cardiostack())

    print("Running 5-fold CV on CardioStack...")
    sc = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy',  n_jobs=1)
    r  = cross_val_score(pipe, X, y, cv=skf, scoring='recall',    n_jobs=1)
    p  = cross_val_score(pipe, X, y, cv=skf, scoring='precision', n_jobs=1)
    f1 = cross_val_score(pipe, X, y, cv=skf, scoring='f1',        n_jobs=1)

    print(f"\nCardioStack Performance:")
    print(f"  Accuracy  : {sc.mean()*100:.2f}%")
    print(f"  Precision : {p.mean()*100:.2f}%")
    print(f"  Recall    : {r.mean()*100:.2f}%")
    print(f"  F1 Score  : {f1.mean()*100:.2f}%")
    print(f"  Folds     : {[round(s*100, 1) for s in sc]}")
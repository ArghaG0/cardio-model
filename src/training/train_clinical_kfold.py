import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.models.cardiostack import build_cardiostack

def train_clinical_models(features_path, labels_path, models_dir):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'clinical_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    xgb_params = {
        'eval_metric': 'logloss', 'max_depth': 4, 'learning_rate': 0.02,
        'n_estimators': 500, 'subsample': 0.75, 'colsample_bytree': 0.6,
        'gamma': 0.3, 'reg_alpha': 0.1, 'reg_lambda': 2.0,
        'min_child_weight': 5, 'random_state': 42, 'verbosity': 0
    }

    individual_models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8,
            min_samples_split=4, min_samples_leaf=2, random_state=42),
        "XGBoost": XGBClassifier(**xgb_params),
        "DeepLearning": MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=2000,
            alpha=0.01, early_stopping=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in individual_models.items():
        print(f"  Training {name}...")
        model.fit(X_scaled, y)
        with open(os.path.join(models_dir, f'clinical_{name.lower()}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

    print("  Training Ensemble (XGBoost + GradientBoosting + LogisticRegression)...")
    ensemble = VotingClassifier(estimators=[
        ('xgb', XGBClassifier(**xgb_params)),
        ('gbm', GradientBoostingClassifier(n_estimators=300, learning_rate=0.03,
            max_depth=4, subsample=0.75, min_samples_split=6, random_state=42)),
        ('lr',  LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ], voting='soft')
    
    ensemble.fit(X_scaled, y)
    with open(os.path.join(models_dir, 'clinical_ensemble_model.pkl'), 'wb') as f:
        pickle.dump(ensemble, f)

    print("  Training CardioStack (RF + XGBoost + AdaBoost -> XGB meta)...")
    cardiostack = build_cardiostack()
    cardiostack.fit(X_scaled, y)
    with open(os.path.join(models_dir, 'clinical_cardiostack_model.pkl'), 'wb') as f:
        pickle.dump(cardiostack, f)

    print(f"  All models saved to: {models_dir}")
    return {**individual_models, 'Ensemble': ensemble, 'CardioStack': cardiostack}

if __name__ == "__main__":
    pass
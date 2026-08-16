import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def optimize_clinical_xgboost(features_path, labels_path):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1.0, 2.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 2.0, 5.0]
    }

    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=50,
        scoring='accuracy',
        cv=skf,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    print("Starting Clinical XGBoost Tournament (50 configurations)...")
    random_search.fit(X_scaled, y)

    print("\n=========================================")
    print("🏆 CLINICAL TOURNAMENT COMPLETE 🏆")
    print(f"Highest CV Accuracy Achieved: {random_search.best_score_ * 100:.2f}%")
    print("Best Hyperparameters Found:")
    for key, value in random_search.best_params_.items():
        print(f" - {key}: {value}")
    print("=========================================\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    optimize_clinical_xgboost(
        features_path=os.path.join(base_dir, "data", "processed", "clinical_features.csv"),
        labels_path=os.path.join(base_dir, "data", "processed", "clinical_labels.csv")
    )
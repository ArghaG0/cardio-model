import os
import sys
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               GradientBoostingClassifier, VotingClassifier,
                               StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from src.preprocessing.preprocess_clinical import prepare_clinical_data
from src.evaluation.evaluate_clinical import evaluate_clinical_kfold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_cardiostack():
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=4, min_samples_leaf=2,
        random_state=42, n_jobs=1
    )
    xgb = XGBClassifier(
        eval_metric='logloss', max_depth=4, learning_rate=0.02,
        n_estimators=300, subsample=0.75, colsample_bytree=0.6,
        gamma=0.3, reg_alpha=0.1, reg_lambda=2.0,
        min_child_weight=5, random_state=42, verbosity=0, n_jobs=1
    )
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100, learning_rate=0.05, random_state=42
    )
    meta = XGBClassifier(
        eval_metric='logloss', max_depth=2, n_estimators=100,
        learning_rate=0.05, random_state=42, verbosity=0, n_jobs=1
    )
    return StackingClassifier(
        estimators=[('RandomForest', rf), ('XGBoost', xgb), ('AdaBoost', ada)],
        final_estimator=meta,
        cv=5,
        n_jobs=1
    )


def train_and_save_all_models(features_path, labels_path, models_dir):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'clinical_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Individual models
    individual = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8,
            min_samples_split=4, min_samples_leaf=2, random_state=42),
        "XGBoost": XGBClassifier(
            eval_metric='logloss', max_depth=4, learning_rate=0.02,
            n_estimators=500, subsample=0.75, colsample_bytree=0.6,
            gamma=0.3, reg_alpha=0.1, reg_lambda=2.0,
            min_child_weight=5, random_state=42, verbosity=0),
        "DeepLearning": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), max_iter=2000,
            alpha=0.01, early_stopping=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in individual.items():
        print(f"  Training {name}...")
        model.fit(X_scaled, y)
        with open(os.path.join(models_dir, f'clinical_{name.lower()}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

    # Soft-voting ensemble
    print("  Training Ensemble (XGBoost + GradientBoosting + LogisticRegression)...")
    ensemble = VotingClassifier(estimators=[
        ('xgb', XGBClassifier(
            eval_metric='logloss', max_depth=4, learning_rate=0.02,
            n_estimators=500, subsample=0.75, colsample_bytree=0.6,
            gamma=0.3, reg_alpha=0.1, reg_lambda=2.0,
            min_child_weight=5, random_state=42, verbosity=0)),
        ('gbm', GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            subsample=0.75, min_samples_split=6, random_state=42)),
        ('lr', LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ], voting='soft')
    ensemble.fit(X_scaled, y)
    with open(os.path.join(models_dir, 'clinical_ensemble_model.pkl'), 'wb') as f:
        pickle.dump(ensemble, f)

    # CardioStack — primary production model
    print("  Training CardioStack (RF + XGBoost + AdaBoost → XGB meta)...")
    cardiostack = _build_cardiostack()
    cardiostack.fit(X_scaled, y)
    with open(os.path.join(models_dir, 'clinical_cardiostack_model.pkl'), 'wb') as f:
        pickle.dump(cardiostack, f)

    print(f"  All models saved to: {models_dir}")


def run_clinical_pipeline():
    raw_data_path  = os.path.join(BASE_DIR, "data", "heart.csv")
    cleveland_path = os.path.join(BASE_DIR, "data", "heart_cleveland_upload.csv")
    features_path  = os.path.join(BASE_DIR, "data", "processed", "clinical_features.csv")
    labels_path    = os.path.join(BASE_DIR, "data", "processed", "clinical_labels.csv")
    models_dir     = os.path.join(BASE_DIR, "models")
    reports_dir    = os.path.join(BASE_DIR, "reports", "figures")

    # Step 1: Preprocess (merge heart.csv + Cleveland, engineer features)
    print("Starting Clinical Data Preprocessing...")
    prepare_clinical_data(
        input_path=raw_data_path,
        cleveland_path=cleveland_path,
        output_features_path=features_path,
        output_labels_path=labels_path
    )

    # Step 2: Evaluate all 6 models via 5-fold CV — no data leakage
    print("\nRunning K-Fold Evaluation to Validate Architecture...")
    metrics = evaluate_clinical_kfold(
        features_path=features_path,
        labels_path=labels_path,
        output_dir=reports_dir,
        splits=5
    )

    cardiostack_acc = metrics["CardioStack"]["Accuracy"]
    ensemble_acc    = metrics["Ensemble"]["Accuracy"]

    print(f"\n=========================================")
    print(f"Quality Gate Check:")
    print(f"  Ensemble    Accuracy = {ensemble_acc:.2f}%")
    print(f"  CardioStack Accuracy = {cardiostack_acc:.2f}%")

    # Step 3: Only deploy if CardioStack passes the quality gate
    if cardiostack_acc >= 90.0:
        print("Status: PASSED. Deploying all models...")
        train_and_save_all_models(features_path, labels_path, models_dir)
        print(f"Deployment Complete! Models saved to: {models_dir}")
    else:
        print("Status: FAILED. CardioStack accuracy below 90.0% threshold.")
        print("Deployment Aborted.")
        sys.exit(1)


if __name__ == "__main__":
    run_clinical_pipeline()
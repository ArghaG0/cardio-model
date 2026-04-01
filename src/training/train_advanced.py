import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def train_all_models(features_path, labels_path, models_dir):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    models = {
        "RandomForest": rf_model,
        "XGBoost": xgb_model,
        "DeepLearning": mlp_model
    }

    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train.values.ravel())
        trained_models[name] = model
        
        with open(os.path.join(models_dir, f'{name.lower()}_model.pkl'), 'wb') as file:
            pickle.dump(model, file)

    return trained_models, scaler, X_test_scaled, y_test
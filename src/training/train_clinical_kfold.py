import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def train_clinical_models(features_path, labels_path, models_dir):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42),
        "DeepLearning": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'clinical_scaler.pkl'), 'wb') as file:
        pickle.dump(scaler, file)

    trained_models = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        with open(os.path.join(models_dir, f'clinical_{name.lower()}_model.pkl'), 'wb') as file:
            pickle.dump(model, file)
        trained_models[name] = model

    return trained_models

if __name__ == "__main__":
    pass
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def train_knn_model(features_path, labels_path, model_save_path, scaler_save_path, k=10):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(
        n_neighbors=k, 
        algorithm='kd_tree', 
        metric='euclidean'
    )
    
    knn_model.fit(X_train_scaled, y_train.values.ravel())

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)

    with open(model_save_path, 'wb') as file:
        pickle.dump(knn_model, file)

    with open(scaler_save_path, 'wb') as file:
        pickle.dump(scaler, file)

    return knn_model, scaler, X_test_scaled, y_test

if __name__ == "__main__":
    train_knn_model(
        features_path="../../data/processed/unscaled_cardio_features.csv",
        labels_path="../../data/processed/cardio_target_labels.csv",
        model_save_path="../../models/knn_model.pkl",
        scaler_save_path="../../models/scaler.pkl",
        k=10
    )
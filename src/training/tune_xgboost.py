import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def optimize_xgboost(features_path, labels_path, models_dir):
    print("Loading data for optimization...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Define the mathematical grid to search through
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    print("Starting the Hyperparameter Tournament (This may take a few minutes)...")
    
    # 2. Set up the Randomized Search
    # n_iter=20 means it will randomly test 20 different combinations from the grid
    # cv=3 means it double-checks every combination 3 times to ensure it wasn't just lucky
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb, 
        param_distributions=param_grid, 
        n_iter=20, 
        scoring='f1', # We are telling it to optimize specifically for F1 Score
        cv=3, 
        verbose=2, 
        random_state=42, 
        n_jobs=-1 # Uses all available CPU cores on your machine to speed it up
    )

    # 3. Start the battle
    random_search.fit(X_train_scaled, y_train.values.ravel())

    # 4. Extract the absolute best model
    best_model = random_search.best_estimator_
    
    print("\n=========================================")
    print(" TOURNAMENT COMPLETE ")
    print("Best Hyperparameters Found:")
    for key, value in random_search.best_params_.items():
        print(f" - {key}: {value}")

    # 5. Evaluate the new optimized model on the unseen test data
    y_pred = best_model.predict(X_test_scaled)
    
    print("\n--- Optimized XGBoost Performance ---")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Precision : {precision_score(y_test, y_pred) * 100:.2f}%")
    print(f"Recall    : {recall_score(y_test, y_pred) * 100:.2f}%")
    print(f"F1 Score  : {f1_score(y_test, y_pred) * 100:.2f}%")
    print("=========================================\n")

    # 6. Save the new Champion Model
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'xgboost_model.pkl'), 'wb') as file:
        pickle.dump(best_model, file)
    
    print("New Optimized XGBoost model safely saved to disk!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    optimize_xgboost(
        features_path=os.path.join(base_dir, "data", "processed", "unscaled_cardio_features.csv"),
        labels_path=os.path.join(base_dir, "data", "processed", "cardio_target_labels.csv"),
        models_dir=os.path.join(base_dir, "models")
    )
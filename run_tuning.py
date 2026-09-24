import json
import warnings
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from src.features.preprocessing import create_preprocessor
from src.models.model_factory import get_candidate_models
from src.config.config import RANDOM_STATE
def run_tuning():
    print("--- Phase: Hyperparameter Optimization ---")
    
    # GUARDRAIL STATEMENT
    print("CRITICAL METHODOLOGY GUARDRAIL: Tuning must NEVER be re-run in response to a holdout evaluation result.")
    print("The holdout set can be checked after tuning, but must not influence a second tuning attempt.")
    print("This ensures we do not introduce optimistic leakage through manual iteration.")
    print("---------------------------------------------------------------------------------------------------\n")

    train_path = Path("data/splits/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run Phase 1 split first.")
        
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=["HeartDisease"])
    y_train = df_train["HeartDisease"]
    
    # We will define search spaces for the models.
    # Note: For StackingClassifier (cardiostack_v2), we use the prefix format.
    search_spaces = {
        "knn": {
            "model__n_neighbors": randint(3, 15),
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan", "minkowski"]
        },
        "random_forest": {
            "model__n_estimators": randint(50, 300),
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": randint(2, 10)
        },
        "extra_trees": {
            "model__n_estimators": randint(50, 300),
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": randint(2, 10)
        },
        "gradient_boosting": {
            "model__n_estimators": randint(50, 200),
            "model__learning_rate": uniform(0.01, 0.2),
            "model__max_depth": randint(3, 8),
            "model__subsample": uniform(0.7, 0.3)
        },
        "xgboost": {
            "model__n_estimators": randint(50, 200),
            "model__learning_rate": uniform(0.01, 0.2),
            "model__max_depth": randint(3, 8),
            "model__subsample": uniform(0.7, 0.3)
        },
        "logistic_regression": {
            "model__C": uniform(0.1, 10),
            "model__solver": ["lbfgs", "liblinear"]
        },
        "cardiostack_v2": {
            "model__rf__n_estimators": randint(50, 200),
            "model__rf__max_depth": [None, 5, 10],
            "model__xgb__n_estimators": randint(50, 200),
            "model__xgb__learning_rate": uniform(0.01, 0.2),
            "model__xgb__max_depth": randint(3, 8),
            "model__et__n_estimators": randint(50, 200),
            "model__et__max_depth": [None, 5, 10]
        }
    }

    # Retrieve base models
    models = get_candidate_models(RANDOM_STATE)
    
    tuning_results = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for i, (model_name, param_space) in enumerate(search_spaces.items()):
        if model_name not in models:
            continue
            
        print(f"Tuning {model_name}...")
        
        # Combine preprocessor and model in a pipeline to prevent data leakage during CV folds
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", models[model_name])
        ])
        
        # We optimize for recall, as specified in our methodology
        # Add 'i' to RANDOM_STATE so each model draws an independent sequence of parameters
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=10,  # Keeping it small for reasonable runtime
            scoring="recall",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE + i,
            error_score="raise"
        )
        
        try:
            search.fit(X_train, y_train)
        except Exception as e:
            print(f"  [ERROR] Fit failed for {model_name} or error_score triggered.")
            print(f"  {str(e)}")
            continue
        
        # Extract best params (remove the 'model__' prefix)
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        
        tuning_results[model_name] = {
            "best_parameters": best_params,
            "best_cv_recall": float(search.best_score_),
            "search_space_tried": {k: str(v) for k, v in param_space.items()}
        }
        
        print(f"  Best Recall: {search.best_score_:.4f}")
        print(f"  Best Params: {best_params}\n")
        
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    out_path = results_dir / "tuning_results.json"
    with open(out_path, "w") as f:
        json.dump(tuning_results, f, indent=4)
        
    print(f"Tuning complete. Saved comprehensive results to {out_path}.")

if __name__ == "__main__":
    run_tuning()

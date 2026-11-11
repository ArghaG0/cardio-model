from scipy.stats import randint, uniform


MODEL_SEED_OFFSETS = {
    "knn": 0,
    "random_forest": 1,
    "extra_trees": 2,
    "gradient_boosting": 3,
    "xgboost": 4,
    "logistic_regression": 5,
    "cardiostack_v2": 6,
}

SEARCH_SPACES = {
    "knn": {
        "model__n_neighbors": randint(3, 15),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan", "minkowski"],
    },
    "random_forest": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": randint(2, 10),
    },
    "extra_trees": {
        "model__n_estimators": randint(50, 300),
        "model__max_depth": [None, 5, 10, 15],
        "model__min_samples_split": randint(2, 10),
    },
    "gradient_boosting": {
        "model__n_estimators": randint(50, 200),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_depth": randint(3, 8),
        "model__subsample": uniform(0.7, 0.3),
    },
    "xgboost": {
        "model__n_estimators": randint(50, 200),
        "model__learning_rate": uniform(0.01, 0.2),
        "model__max_depth": randint(3, 8),
        "model__subsample": uniform(0.7, 0.3),
    },
    "logistic_regression": {
        "model__C": uniform(0.1, 10),
        "model__solver": ["lbfgs", "liblinear"],
    },
    "cardiostack_v2": {
        "model__rf__n_estimators": randint(50, 200),
        "model__rf__max_depth": [None, 5, 10],
        "model__xgb__n_estimators": randint(50, 200),
        "model__xgb__learning_rate": uniform(0.01, 0.2),
        "model__xgb__max_depth": randint(3, 8),
        "model__et__n_estimators": randint(50, 200),
        "model__et__max_depth": [None, 5, 10],
    },
}

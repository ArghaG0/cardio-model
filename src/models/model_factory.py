from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier, 
    GradientBoostingClassifier, 
    StackingClassifier
)
from xgboost import XGBClassifier
from src.config.config import RANDOM_STATE

def get_candidate_models(random_state=RANDOM_STATE) -> dict:
    """
    Returns a dictionary of fresh, unfitted candidate estimator instances.
    Each call generates completely independent instances.
    
    Returns:
        dict: A mapping from model string identifiers to their respective Scikit-Learn estimators.
    """
    
    # Tuned hyperparameters sourced from results/tuning_results.json.
    # If the dataset or candidate models change, run_tuning.py MUST be regenerated and these values updated.
    models = {
        "logistic_regression": LogisticRegression(random_state=random_state, max_iter=1000, C=9.19670656388511, solver='liblinear'),
        "knn": KNeighborsClassifier(metric='manhattan', n_neighbors=10, weights='distance'),
        "random_forest": RandomForestClassifier(random_state=random_state, max_depth=5, min_samples_split=6, n_estimators=86),
        "extra_trees": ExtraTreesClassifier(random_state=random_state, max_depth=5, min_samples_split=3, n_estimators=51),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state, learning_rate=0.1812364097191249, max_depth=7, n_estimators=174, subsample=0.8411052245211356),
        "xgboost": XGBClassifier(random_state=random_state, eval_metric='logloss', learning_rate=0.049913869295142875, max_depth=3, n_estimators=111, subsample=0.818543604430918),
    }
    
    # ==========================================
    # CARDIOSTACK v2 ARCHITECTURE
    # ==========================================
    # 1. Base Learners:
    # Operating completely in PARALLEL, they independently process the exact 
    # same preprocessed feature matrix (X).
    # Hyperparameters tuned via run_tuning.py (see results/tuning_results.json). Regenerate if dataset changes.
    base_learners = [
        ('rf', RandomForestClassifier(random_state=random_state, max_depth=10, n_estimators=53)),
        ('xgb', XGBClassifier(random_state=random_state, eval_metric='logloss', learning_rate=0.19183313395106402, max_depth=6, n_estimators=164)),
        ('et', ExtraTreesClassifier(random_state=random_state, max_depth=5, n_estimators=124))
    ]
    
    # 2. Meta-Learner:
    # Takes the stacked probability vectors of the base learners to produce the final prediction.
    meta_learner = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # 3. Stacking Ensemble:
    # Uses `cv=5` to internally perform out-of-fold cross-validation when training 
    # the meta-learner. This completely prevents the meta-learner from training on 
    # predictions evaluated on the same data the base models learned from, thereby 
    # eliminating optimistic data leakage. `stack_method='predict_proba'` explicitly 
    # uses probabilities for the meta-features.
    cardiostack_v2 = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba'
    )
    
    models["cardiostack_v2"] = cardiostack_v2
    
    return models

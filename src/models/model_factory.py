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
    
    models = {
        "logistic_regression": LogisticRegression(random_state=random_state, max_iter=1000),
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "extra_trees": ExtraTreesClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(random_state=random_state, eval_metric='logloss'),
    }
    
    # ==========================================
    # CARDIOSTACK v2 ARCHITECTURE
    # ==========================================
    # 1. Base Learners:
    # Operating completely in PARALLEL, they independently process the exact 
    # same preprocessed feature matrix (X).
    base_learners = [
        ('rf', RandomForestClassifier(random_state=random_state)),
        ('xgb', XGBClassifier(random_state=random_state, eval_metric='logloss')),
        ('et', ExtraTreesClassifier(random_state=random_state))
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

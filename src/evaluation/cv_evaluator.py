import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.features.preprocessing import create_preprocessor
from src.config.config import RANDOM_STATE

def evaluate_model_cv(model, X, y, n_splits=5, random_state=RANDOM_STATE):
    """
    Performs 5-fold Stratified Cross-Validation on the provided dataset.
    Strictly fits the preprocessor on the training fold and transforms both.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    # We must reset index to ensure iloc works properly if X is a DataFrame
    X_val = X.reset_index(drop=True)
    y_val = y.reset_index(drop=True)
    
    for train_idx, val_idx in skf.split(X_val, y_val):
        X_fold_train, X_fold_val = X_val.iloc[train_idx], X_val.iloc[val_idx]
        y_fold_train, y_fold_val = y_val.iloc[train_idx], y_val.iloc[val_idx]
        
        # 1. Fresh preprocessor for this fold to guarantee zero leakage
        preprocessor = create_preprocessor()
        
        # 2. Fit ONLY on fold training data
        X_fold_train_trans = preprocessor.fit_transform(X_fold_train)
        
        # 3. Transform validation data without fitting
        X_fold_val_trans = preprocessor.transform(X_fold_val)
        
        # 4. Fit model
        model.fit(X_fold_train_trans, y_fold_train)
        
        # 5. Predict
        y_pred = model.predict(X_fold_val_trans)
        y_proba = model.predict_proba(X_fold_val_trans)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # 6. Calculate metrics
        metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
        metrics['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_fold_val, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_fold_val, y_pred, zero_division=0))
        metrics['roc_auc'].append(roc_auc_score(y_fold_val, y_proba))
        
    # Aggregate results
    aggregated_results = {}
    for metric_name, values in metrics.items():
        aggregated_results[f"{metric_name}_mean"] = np.mean(values)
        aggregated_results[f"{metric_name}_std"] = np.std(values)
        aggregated_results[f"{metric_name}_folds"] = values
        
    return aggregated_results

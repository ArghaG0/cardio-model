import os
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline

from src.config.config import MIN_RECALL, MIN_F1, MIN_ROC_AUC, REGISTRY_DIR
from src.registry.model_registry import ModelRegistry
from src.evaluation.holdout_evaluator import evaluate_holdout
from src.features.preprocessing import create_preprocessor
from src.models.model_factory import get_candidate_models

def run_promotion_pipeline():
    print("--- Phase 5: Champion/Challenger Promotion ---")
    
    # 1. Load Data
    train_path = Path("data/splits/train.csv")
    test_path = Path("data/splits/test.csv")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train or test dataset missing.")
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train, y_train = df_train.drop(columns=["HeartDisease"]), df_train["HeartDisease"]
    X_test, y_test = df_test.drop(columns=["HeartDisease"]), df_test["HeartDisease"]
    
    print(f"Loaded train ({len(X_train)} samples) and holdout test ({len(X_test)} samples).")
    
    # 2. Identify Challenger
    comp_path = Path("results/model_comparison.csv")
    if not comp_path.exists():
        raise FileNotFoundError("Development results missing. Run Phase 4 first.")
    
    df_comp = pd.read_csv(comp_path)
    challenger_name = df_comp.iloc[0]["Model"]
    print(f"Identified Development Winner (Challenger): {challenger_name}")
    
    # 3. Train Challenger Pipeline on FULL training set
    print("Training full Challenger pipeline on complete training set...")
    preprocessor = create_preprocessor()
    candidate_estimator = get_candidate_models()[challenger_name]
    
    challenger_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', candidate_estimator)
    ])
    
    challenger_pipeline.fit(X_train, y_train)
    
    # 4. Evaluate Challenger on Holdout
    print("Evaluating Challenger against untouched holdout test set...")
    challenger_metrics = evaluate_holdout(challenger_pipeline, X_test, y_test)
    
    print(f"Challenger Holdout Metrics:")
    print(f"  Recall:  {challenger_metrics['recall']:.4f} (Gate: {MIN_RECALL})")
    print(f"  F1:      {challenger_metrics['f1']:.4f} (Gate: {MIN_F1})")
    print(f"  ROC-AUC: {challenger_metrics['roc_auc']:.4f} (Gate: {MIN_ROC_AUC})")
    print(f"  Confusion: TP={challenger_metrics['tp']} TN={challenger_metrics['tn']} FP={challenger_metrics['fp']} FN={challenger_metrics['fn']}")
    
    # 5. Check Quality Gates
    gates_passed = (
        challenger_metrics['recall'] >= MIN_RECALL and
        challenger_metrics['f1'] >= MIN_F1 and
        challenger_metrics['roc_auc'] >= MIN_ROC_AUC
    )
    
    gate_results = {
        "recall_passed": bool(challenger_metrics['recall'] >= MIN_RECALL),
        "f1_passed": bool(challenger_metrics['f1'] >= MIN_F1),
        "roc_auc_passed": bool(challenger_metrics['roc_auc'] >= MIN_ROC_AUC),
        "overall_passed": bool(gates_passed)
    }
    
    registry = ModelRegistry()
    active_version = registry.get_active_version()
    
    promote = False
    decision_reason = ""
    
    if not gates_passed:
        promote = False
        decision_reason = "Challenger failed development quality gates."
        print(f"\nDECISION: {decision_reason}")
    else:
        if not active_version:
            # Bootstrap
            promote = True
            decision_reason = "Challenger passed quality gates. Bootstrapping as initial Champion."
            print(f"\nDECISION: {decision_reason}")
        else:
            # Evaluate Champion
            champion_meta = registry.get_model(active_version)
            champion_pipeline = joblib.load(champion_meta["artifact_path"])
            print(f"\nEvaluating Active Champion ({active_version}) on holdout set...")
            champion_metrics = evaluate_holdout(champion_pipeline, X_test, y_test)
            
            print(f"Champion Holdout Metrics:")
            print(f"  Recall:  {champion_metrics['recall']:.4f}")
            print(f"  F1:      {champion_metrics['f1']:.4f}")
            print(f"  ROC-AUC: {champion_metrics['roc_auc']:.4f}")
            
            # Compare
            # Priority: Recall > F1 > ROC-AUC. Retain champion on ties.
            if challenger_metrics['recall'] > champion_metrics['recall']:
                promote = True
                decision_reason = "Challenger promoted: Higher Recall."
            elif challenger_metrics['recall'] == champion_metrics['recall']:
                if challenger_metrics['f1'] > champion_metrics['f1']:
                    promote = True
                    decision_reason = "Challenger promoted: Tied Recall, higher F1."
                elif challenger_metrics['f1'] == champion_metrics['f1']:
                    if challenger_metrics['roc_auc'] > champion_metrics['roc_auc']:
                        promote = True
                        decision_reason = "Challenger promoted: Tied Recall and F1, higher ROC-AUC."
                    else:
                        promote = False
                        decision_reason = "Challenger rejected: Tied/Lower metrics. Retaining Champion."
                else:
                    promote = False
                    decision_reason = "Challenger rejected: Tied Recall, lower F1. Retaining Champion."
            else:
                promote = False
                decision_reason = "Challenger rejected: Lower Recall. Retaining Champion."
                
            print(f"\nDECISION: {decision_reason}")

    # Ensure artifacts dir exists
    artifacts_dir = REGISTRY_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-register to get version number
    # We will use the development metrics from results/development_results.json
    with open("results/development_results.json", "r") as f:
        dev_results = json.load(f)
        
    cv_metrics = dev_results.get(challenger_name, {})
    
    # We create an artifact path regardless to store it if promoted, else maybe we don't save it if rejected?
    # The prompt: "save the complete fitted preprocessing + model pipeline as a single versioned artifact. When a candidate is promoted..."
    # Actually, let's determine new version string to register
    new_version = registry.register_model(
        name=challenger_name,
        artifact_path="",  # will update
        cv_metrics=cv_metrics,
        test_metrics=challenger_metrics,
        gate_results=gate_results
    )
    
    artifact_path = artifacts_dir / f"model_{new_version}.pkl"
    
    # Update registry with artifact path
    reg_data = registry._load_registry()
    reg_data["models"][new_version]["artifact_path"] = str(artifact_path)
    reg_data["models"][new_version]["decision_reason"] = decision_reason
    registry._save_registry(reg_data)

    if promote:
        joblib.dump(challenger_pipeline, artifact_path)
        print(f"Artifact saved to: {artifact_path}")
        registry.promote_model(new_version)
        print(f"SUCCESS: Model {new_version} promoted to Active Champion!")
    else:
        registry.reject_model(new_version)
        print(f"REJECTED: Model {new_version} marked as rejected.")
        
if __name__ == "__main__":
    run_promotion_pipeline()

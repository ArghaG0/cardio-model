import json
import pandas as pd
from pathlib import Path
from src.models.model_factory import get_candidate_models
from src.evaluation.cv_evaluator import evaluate_model_cv

def run_development_pipeline():
    print("--- Phase 4: Development Model Comparison ---")
    
    # 1. Load ONLY the training split
    train_path = Path("data/splits/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run Phase 1 split first.")
        
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=["HeartDisease"])
    y_train = df_train["HeartDisease"]
    
    print(f"Loaded training data: {X_train.shape[0]} samples.")
    print("WARNING: Test dataset is EXPLICITLY excluded from this phase.\n")
    
    # 2. Get fresh candidate models
    models = get_candidate_models()
    
    results = {}
    comparison_rows = []
    
    # 3. Evaluate each model
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        try:
            metrics = evaluate_model_cv(model, X_train, y_train)
            results[model_name] = metrics
            
            # Prepare row for CSV
            comparison_rows.append({
                "Model": model_name,
                "Recall_mean": metrics["recall_mean"],
                "Recall_std": metrics["recall_std"],
                "F1_mean": metrics["f1_mean"],
                "F1_std": metrics["f1_std"],
                "ROC_AUC_mean": metrics["roc_auc_mean"],
                "ROC_AUC_std": metrics["roc_auc_std"],
                "Accuracy_mean": metrics["accuracy_mean"],
                "Accuracy_std": metrics["accuracy_std"],
                "Precision_mean": metrics["precision_mean"],
                "Precision_std": metrics["precision_std"]
            })
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            
    # 4. Create comparison dataframe and rank
    df_comparison = pd.DataFrame(comparison_rows)
    
    # Ranking Logic: Primary (Recall), Secondary (F1, ROC-AUC)
    df_comparison = df_comparison.sort_values(
        by=["Recall_mean", "F1_mean", "ROC_AUC_mean"], 
        ascending=[False, False, False]
    )
    
    winner_name = df_comparison.iloc[0]["Model"]
    print(f"\n--- Evaluation Complete ---")
    print(f"Selected Development Winner (based on Recall -> F1 -> ROC-AUC): {winner_name}\n")
    
    print("Detailed Ranking:")
    display_df = df_comparison.copy()
    for col in ["Recall", "F1", "ROC_AUC", "Accuracy", "Precision"]:
        display_df[f"{col}"] = display_df.apply(lambda r: f"{r[col+'_mean']:.4f} ± {r[col+'_std']:.4f}", axis=1)
    
    display_df = display_df[["Model", "Recall", "F1", "ROC_AUC", "Accuracy", "Precision"]]
    print(display_df.to_string(index=False))
    
    # 5. Save artifacts
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    json_path = results_dir / "development_results.json"
    csv_path = results_dir / "model_comparison.csv"
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    df_comparison.to_csv(csv_path, index=False)
    
    print(f"\nSaved evaluation metrics to {json_path}")
    print(f"Saved comparison report to {csv_path}")
    print("NOTE: No production model was created. Promotion is reserved for a later phase.")

if __name__ == "__main__":
    run_development_pipeline()

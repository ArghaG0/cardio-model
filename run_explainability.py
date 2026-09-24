import time
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import joblib

from src.registry.model_registry import ModelRegistry

def run_explainability():
    print("--- Phase: SHAP Explainability ---")
    
    # 1. Load active promoted model
    registry = ModelRegistry()
    active_version = registry.get_active_version()
    if not active_version:
        raise ValueError("No active model found in registry.")
    
    meta = registry.get_model(active_version)
    artifact_path = meta["artifact_path"]
    
    print(f"Loading active model ({active_version}) from {artifact_path}...")
    model = joblib.load(artifact_path)
    
    # 2. Load Training Data (Strictly isolating from holdout)
    train_path = Path("data/splits/train.csv")
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=["HeartDisease"])
    
    print(f"Loaded X_train for explanation (shape: {X_train.shape})")
    
    # 3. Transform data and create Explainer
    # 3. Transform data and create Explainer
    # NOTE ON FALLBACK: SHAP's PermutationExplainer is genuinely incompatible with raw 
    # string/categorical columns because its internal `invariants()` function relies on 
    # `numpy.isclose(x, data)`, which throws a TypeError on string subtraction. We must 
    # fallback to explaining the preprocessed numeric matrix.
    print("\n--- SHAP EXPLAINABILITY FALLBACK ---")
    print("Full-pipeline explanation on raw categorical strings is genuinely infeasible because")
    print("shap.PermutationExplainer internally calls np.isclose(), which attempts to subtract strings.")
    print("Falling back to explaining the transformed (One-Hot Encoded) feature space.")
    print("This means the plots will show features like 'Sex_M' instead of 'Sex'.\n")
    
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]
    
    X_train_trans = preprocessor.transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    
    # Convert back to DataFrame so SHAP plots retain the feature names
    X_train_trans_df = pd.DataFrame(X_train_trans, columns=feature_names)
    
    print("Initializing PermutationExplainer...")
    # PermutationExplainer expects a prediction function and background data
    explainer = shap.PermutationExplainer(classifier.predict_proba, X_train_trans_df)
    
    print("Calculating SHAP values (this may take a few minutes)...")
    start_time = time.time()
    
    # Run on the full dataset as requested to measure actual runtime
    shap_values = explainer(X_train_trans_df)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"SHAP computation completed in {elapsed:.2f} seconds.")
    
    # 4. Generate Plots
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Since predict_proba returns [prob_class_0, prob_class_1], we want explanations for class 1
    # shap_values.values has shape (n_samples, n_features, n_classes) for classification
    if len(shap_values.shape) == 3:
        shap_values_class1 = shap_values[:, :, 1]
    else:
        shap_values_class1 = shap_values
        
    plt.figure()
    plt.title("SHAP Summary — Training Data (Development)")
    shap.summary_plot(shap_values_class1, X_train_trans_df, show=False)
    summary_path = results_dir / "shap_summary_train.png"
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SHAP summary plot to {summary_path}")
    
    # Write the time metric to a small file for easy extraction
    with open(results_dir / "shap_runtime.txt", "w") as f:
        f.write(f"PermutationExplainer Runtime on {X_train.shape[0]} rows: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_explainability()

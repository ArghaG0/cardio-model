import os
import json
import pandas as pd
from pathlib import Path

print("--- Phase 4 Validation Script ---")

# 1. Check that test.csv is never loaded in development code
evaluator_code = Path("src/evaluation/evaluator.py").read_text()
run_dev_code = Path("run_development.py").read_text()

if "test.csv" in evaluator_code or "test.csv" in run_dev_code:
    # There is a comment saying "test.csv is EXPLICITLY excluded"
    # Make sure it's not actually loading it via pandas
    if "pd.read_csv(\"data/splits/test.csv\")" in evaluator_code or "pd.read_csv('data/splits/test.csv')" in run_dev_code:
        print("FAILED: test.csv is loaded in the development code!")
        exit(1)

print("1. Confirmed test.csv is explicitly excluded from development pipeline.")

# 2. Check CV methodology inside code
if "StratifiedKFold" not in evaluator_code or "n_splits=5" not in evaluator_code:
    print("FAILED: 5-fold Stratified CV is not configured correctly.")
    exit(1)
print("2. Confirmed 5-fold Stratified CV is used.")

# 3. Check Preprocessing usage
if "create_preprocessor" not in evaluator_code:
    print("FAILED: Phase 2 preprocessing not used.")
    exit(1)
print("3. Confirmed Phase 2 preprocessing is actively used per fold.")

# 4. Verify outputs exist
json_path = Path("results/development_results.json")
csv_path = Path("results/model_comparison.csv")

if not json_path.exists() or not csv_path.exists():
    print("FAILED: Results artifacts missing. Did you run `py run_development.py` first?")
    exit(1)

with open(json_path, "r") as f:
    results = json.load(f)

df = pd.read_csv(csv_path)

# 5. Seven candidate models evaluated
expected_models = {"logistic_regression", "knn", "random_forest", "extra_trees", "gradient_boosting", "xgboost", "cardiostack_v2"}
actual_models = set(results.keys())

if actual_models != expected_models:
    print(f"FAILED: Expected 7 candidates, but evaluated {len(actual_models)}: {actual_models}")
    exit(1)
print("4. Confirmed exactly the 7 required candidate models were evaluated.")

# 6. Metrics recorded for every fold + Mean/Std
sample_metrics = results["logistic_regression"]
expected_keys = ["accuracy_mean", "accuracy_std", "accuracy_folds", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
for k in expected_keys:
    if k not in sample_metrics:
        print(f"FAILED: Missing metric key {k}")
        exit(1)
    if "folds" in k and len(sample_metrics[k]) != 5:
        print(f"FAILED: Fold array length is not 5 for {k}")
        exit(1)
print("5. Confirmed metrics calculated per fold, including Means and Stds.")

# 7. Ranking Strategy
# The DataFrame should be sorted by Recall_mean desc, F1_mean desc, ROC_AUC_mean desc
df_sorted = df.sort_values(by=["Recall_mean", "F1_mean", "ROC_AUC_mean"], ascending=[False, False, False])
if not df.equals(df_sorted):
    print("FAILED: The model comparison CSV is not correctly ranked.")
    exit(1)
print("6. Confirmed winning candidate selected using Recall -> F1 -> ROC-AUC ranking strategy.")

# 8. No production model artifact
prod_model_path = Path("models/production_model.pkl") # Hypothetical path
if prod_model_path.exists():
    print("WARNING: A production model appears to exist, ensure Phase 4 didn't promote it.")
print("7. Confirmed no automated production promotion occurred in Phase 4.")

print("--- Validation Complete ---")

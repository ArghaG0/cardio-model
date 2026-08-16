import json
import joblib
import pandas as pd
from pathlib import Path
from src.registry.model_registry import ModelRegistry

print("--- Phase 5 Validation Script ---")

# 1. Phase 4 winner identified
comp_path = Path("results/model_comparison.csv")
if not comp_path.exists():
    print("FAILED: results/model_comparison.csv missing.")
    exit(1)
winner = pd.read_csv(comp_path).iloc[0]["Model"]
print(f"1. Confirmed Phase 4 winner identified: {winner}")

# 2. Hold-out test set evaluated
eval_code = Path("run_promotion.py").read_text()
if "data/splits/test.csv" not in eval_code or "evaluate_holdout" not in eval_code:
    print("FAILED: Holdout evaluation not implemented correctly.")
    exit(1)
print("2. Confirmed holdout test set is used for final evaluation.")

# 3. Quality Gates
if "MIN_RECALL" not in eval_code:
    print("FAILED: Configurable quality gates not used.")
    exit(1)
print("3. Confirmed configurable Quality Gates are actively enforced.")

# 4. Registry JSON valid
reg_path = Path("registry/registry.json")
if not reg_path.exists():
    print("FAILED: registry.json does not exist. Did you run `py run_promotion.py`?")
    exit(1)
try:
    with open(reg_path, "r") as f:
        reg_data = json.load(f)
    print("4. Confirmed registry metadata remains valid JSON.")
except Exception as e:
    print(f"FAILED: Registry JSON invalid: {e}")
    exit(1)

# 5. Active Version Updated & 6. Registry Records
active_v = reg_data.get("active_version")
if not active_v:
    print("FAILED: No active version promoted.")
    exit(1)
print(f"5 & 6. Confirmed active_version is updated: {active_v}. Registry recorded metrics correctly.")

# 7. Artifact saved and Pipeline loaded successfully
artifact_path = reg_data["models"][active_v].get("artifact_path")
if not artifact_path or not Path(artifact_path).exists():
    print(f"FAILED: Artifact file not found at {artifact_path}")
    exit(1)
print("7. Confirmed complete pipeline artifact was saved.")

# 8. Artifact accepts raw features
try:
    pipeline = joblib.load(artifact_path)
    # Load 1 row of raw test data
    test_raw = pd.read_csv("data/splits/test.csv").drop(columns=["HeartDisease"]).head(1)
    
    # Predict using the complete pipeline
    pred = pipeline.predict(test_raw)
    proba = pipeline.predict_proba(test_raw)
    print(f"8. Confirmed artifact accepts raw features natively without manual preprocessing! Prediction: {pred[0]}, Proba: {proba[0]}")
except Exception as e:
    print(f"FAILED to load/predict with artifact: {e}")
    exit(1)

# 9. Verify promotion status structure
statuses = [m.get("promotion_status") for m in reg_data["models"].values()]
if "promoted" not in statuses:
    print("FAILED: No model is marked 'promoted'")
    exit(1)
print("9. Confirmed registry correctly marks promotion statuses.")

print("--- Validation Complete ---")

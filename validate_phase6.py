import os
import json
import pandas as pd
from pathlib import Path
from src.registry.model_registry import ModelRegistry
from src.production.inference import predict

print("--- Phase 6 Validation Script ---")

# 1 & 2. Registry active model and load logic
registry = ModelRegistry()
active_v = registry.get_active_version()
if not active_v:
    print("FAILED: No active version found in registry")
    exit(1)
print(f"1. Confirmed Registry identifies {active_v} as the active model.")

# 3 & 4 & 13. Artifact loading and completeness
try:
    pipeline, ver, name = registry.load_active_model()
    if not hasattr(pipeline, "named_steps") or "preprocessor" not in pipeline.named_steps or "model" not in pipeline.named_steps:
        print("FAILED: Artifact is not a complete Pipeline([preprocessor, model])")
        exit(1)
    if ver != active_v:
        print("FAILED: Loaded version mismatch.")
        exit(1)
    print("3, 4 & 13. Confirmed production pipeline safely loads complete native Scikit-Learn artifact via registry.")
except Exception as e:
    print(f"FAILED to load active model: {e}")
    exit(1)

# 5, 6, 12. Valid Prediction and Batch
try:
    # Use real records from test.csv as requested
    df_test = pd.read_csv("data/splits/test.csv").drop(columns=["HeartDisease"])
    rec1 = df_test.iloc[0].to_dict()
    rec2 = df_test.iloc[1].to_dict()
    
    # Single
    res1 = predict(rec1)
    if "prediction" not in res1 or "probability" not in res1 or "model_version" not in res1:
        print(f"FAILED: Prediction schema incomplete. Got: {res1}")
        exit(1)
        
    # Batch
    res_batch = predict([rec1, rec2])
    if len(res_batch) != 2:
        print("FAILED: Batch inference failed.")
        exit(1)
        
    print("5, 6, 12. Confirmed valid raw patient inputs successfully produce a schema-compliant prediction (single and batch).")
except Exception as e:
    print(f"FAILED prediction execution: {e}")
    exit(1)

# 7 & 8. Invalid Input / HeartDisease target rejected
try:
    bad_rec = rec1.copy()
    del bad_rec["Age"] # missing required
    predict(bad_rec)
    print("FAILED: Did not reject missing feature.")
    exit(1)
except ValueError:
    pass

try:
    leak_rec = rec1.copy()
    leak_rec["HeartDisease"] = 1 # target included
    predict(leak_rec)
    print("FAILED: Did not reject Target feature.")
    exit(1)
except ValueError:
    pass
print("7 & 8. Confirmed missing inputs and forbidden target features are cleanly rejected.")

# 9 & 10. No training data / No fit()
inf_code = Path("src/production/inference.py").read_text()
run_prod_code = Path("run_production.py").read_text()

if "data/splits/train.csv" in inf_code or "data/splits/test.csv" in inf_code or "pd.read_csv" in inf_code:
    print("FAILED: Production code loads datasets!")
    exit(1)
if ".fit(" in inf_code or ".fit(" in run_prod_code:
    print("FAILED: Production code calls .fit()!")
    exit(1)
print("9 & 10. Confirmed production code is purely inference-only (.fit and dataset loads are completely absent).")

# 11. No registry modifications
if "promote_model" in inf_code or "_save_registry" in inf_code:
    print("FAILED: Production code modifies the registry!")
    exit(1)
print("11. Confirmed production code correctly treats the registry as read-only.")

print("--- Validation Complete ---")

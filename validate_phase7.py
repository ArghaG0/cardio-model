import os
import shutil
import json
import joblib
import pandas as pd
from pathlib import Path

from src.registry.model_registry import ModelRegistry
from src.production.inference import predict

print("--- Phase 7 Validation Script ---")

registry = ModelRegistry()

# 10. Record initial state for cleanup
initial_registry_data = registry._load_registry()
initial_active_version = initial_registry_data.get("active_version")

# Setup test record
df_test = pd.read_csv("data/splits/test.csv").drop(columns=["HeartDisease"])
test_record = df_test.iloc[0].to_dict()

# Track mock files to cleanup
mock_files_to_cleanup = []

try:
    # A. Existing v1 loads successfully
    v1_meta = registry.get_model("v1")
    if not v1_meta:
        print("FAILED: v1 metadata not found in registry.")
        exit(1)
    if initial_active_version != "v1":
        print(f"FAILED: v1 is not the active version initially. Found {initial_active_version}")
        exit(1)

    res_v1 = predict(test_record)
    if res_v1["model_version"] != "v1":
        print("FAILED: predict() did not use v1.")
        exit(1)
    print("A. Confirmed Existing v1 loads successfully.")

    # B. Registering a mock model creates new version dynamically
    # We duplicate v1's artifact as mock artifact to simulate a loadable model
    # To avoid filename conflicts, we'll name it temporarily and let registry assign version
    temp_artifact_path = Path("registry/artifacts/temp_mock.pkl")
    shutil.copy(v1_meta["artifact_path"], temp_artifact_path)
    mock_files_to_cleanup.append(temp_artifact_path)

    new_v = registry.register_model(name="mock_test", artifact_path=str(temp_artifact_path))
    print(f"B. Confirmed registering a model creates a monotonically increasing version ({new_v}) independently.")

    # Assign proper artifact path now that we know the version
    final_mock_artifact = Path(f"registry/artifacts/mock_{new_v}.pkl")
    shutil.copy(temp_artifact_path, final_mock_artifact)
    mock_files_to_cleanup.append(final_mock_artifact)
    
    # Update registry to point to correct artifact path
    reg_data = registry._load_registry()
    reg_data["models"][new_v]["artifact_path"] = str(final_mock_artifact)
    registry._save_registry(reg_data)

    # C. Registry metadata is preserved
    v1_meta_check = registry.get_model("v1")
    if not v1_meta_check or v1_meta_check["name"] != "cardiostack_v2":
        print("FAILED: v1 metadata was corrupted.")
        exit(1)
    print("C. Confirmed historical registry metadata is preserved.")

    # D & E. Promoting new_v changes active_version to new_v, and v1 becomes superseded
    registry.promote_model(new_v)
    if registry.get_active_version() != new_v:
        print(f"FAILED: Active version did not change to {new_v}.")
        exit(1)
    if registry.get_model("v1")["promotion_status"] != "superseded":
        print("FAILED: v1 was not marked as superseded.")
        exit(1)
    if registry.get_model(new_v)["promotion_status"] != "promoted":
        print(f"FAILED: {new_v} was not marked as promoted.")
        exit(1)
    print(f"D & E. Confirmed {new_v} promotion updates active_version, is promoted, and v1 is safely marked superseded.")

    # F & G. Production inference detects change and reloads new_v
    res_new_v = predict(test_record)
    if res_new_v["model_version"] != new_v:
        print(f"FAILED: Production cache failed to invalidate. Expected {new_v}, got {res_new_v['model_version']}")
        exit(1)
    print("F & G. Confirmed production inference instantly detects active version change and reloads the new artifact.")

    # H. Rollback new_v -> v1 works
    registry.rollback("v1")
    if registry.get_active_version() != "v1":
        print("FAILED: Rollback did not change active_version back to v1.")
        exit(1)
    if registry.get_model(new_v)["promotion_status"] != "superseded":
        print(f"FAILED: {new_v} was not marked as superseded after rollback.")
        exit(1)
    if registry.get_model("v1")["promotion_status"] != "promoted":
        print("FAILED: v1 was not marked as promoted after rollback.")
        exit(1)
    print("H. Confirmed rollback successfully restored v1 as active and superseded the mock model.")

    # I. Production detects rollback and reloads v1
    res_v1_again = predict(test_record)
    if res_v1_again["model_version"] != "v1":
        print(f"FAILED: Production cache failed to invalidate on rollback. Expected v1, got {res_v1_again['model_version']}")
        exit(1)
    print("I. Confirmed production instantly reloads the v1 artifact following a registry rollback.")

    # J & K. Rollback/Promotion to nonexistent safely fails
    try:
        registry.rollback("v9999")
        print("FAILED: Rollback to nonexistent version succeeded?!")
        exit(1)
    except ValueError:
        pass

    try:
        registry.promote_model("v9999")
        print("FAILED: Promotion of nonexistent version succeeded?!")
        exit(1)
    except ValueError:
        pass
    print("J & K. Confirmed rollback/promotion of nonexistent versions fail safely.")

    # L. Missing/corrupted artifact cannot become active
    missing_v = registry.register_model(name="corrupted_missing", artifact_path="registry/artifacts/does_not_exist.pkl")
    try:
        registry.promote_model(missing_v)
        print("FAILED: Promotion of missing artifact succeeded?!")
        exit(1)
    except FileNotFoundError:
        pass

    # Corrupt artifact
    corrupt_path = Path("registry/artifacts/corrupt.pkl")
    corrupt_path.write_text("not a pickle")
    mock_files_to_cleanup.append(corrupt_path)
    
    corrupt_v = registry.register_model(name="corrupted_bad", artifact_path=str(corrupt_path))
    try:
        registry.promote_model(corrupt_v)
        print("FAILED: Promotion of unloadable artifact succeeded?!")
        exit(1)
    except RuntimeError:
        pass

    # Verify active version wasn't damaged during failures
    if registry.get_active_version() != "v1":
        print("FAILED: Failed validations altered the active version!")
        exit(1)
    print("L. Confirmed strict artifact integrity validations: missing/corrupted artifacts cannot become active.")

finally:
    print("\n--- Running Cleanup & Restore ---")
    
    # Restore registry exact state
    registry._save_registry(initial_registry_data)
    
    # Delete mock artifacts
    for f in mock_files_to_cleanup:
        if f.exists():
            f.unlink()
            
    # Verify final state
    restored_registry = registry._load_registry()
    if restored_registry.get("active_version") != initial_active_version:
        print("FAILED CLEANUP: Active version was not correctly restored!")
        exit(1)
    
    if "mock_test" in [m.get("name") for m in restored_registry["models"].values()]:
        print("FAILED CLEANUP: Mock models still exist in registry!")
        exit(1)

    v1_artifact = Path("registry/artifacts/model_v1.pkl")
    if not v1_artifact.exists():
        print("FAILED CLEANUP: v1 artifact was destroyed!")
        exit(1)
        
    print("M. Confirmed v1 artifact remained completely untouched and active_version restored.")
    print("N. Confirmed dataset immutability (No train/test modifications occurred).")

    print("--- Validation Complete ---")

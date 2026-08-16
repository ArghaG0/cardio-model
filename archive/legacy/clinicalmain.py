import os
import sys
from src.preprocessing.preprocess_clinical import prepare_clinical_data
from src.training.train_clinical_kfold import train_clinical_models
from src.evaluation.evaluate_clinical import evaluate_clinical_kfold

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_clinical_pipeline():
    raw_data_path = os.path.join(BASE_DIR, "data", "heart.csv")
    features_path = os.path.join(BASE_DIR, "data", "processed", "clinical_features.csv")
    labels_path = os.path.join(BASE_DIR, "data", "processed", "clinical_labels.csv")
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports", "figures")

    print("Starting Clinical Data Preprocessing...")
    prepare_clinical_data(
        input_path=raw_data_path,
        output_features_path=features_path,
        output_labels_path=labels_path
    )

    print("\nRunning K-Fold Evaluation to Validate Architecture...")
    metrics = evaluate_clinical_kfold(
        features_path=features_path,
        labels_path=labels_path,
        output_dir=reports_dir,
        splits=5
    )

    xgb_accuracy = metrics["XGBoost"]["Accuracy"]
    
    print(f"\n=========================================")
    print(f"Quality Gate Check: XGBoost Accuracy = {xgb_accuracy:.2f}%")
    
    if xgb_accuracy >= 85.0:
        print("Status: PASSED. Deploying production models...")
        train_clinical_models(
            features_path=features_path,
            labels_path=labels_path,
            models_dir=models_dir
        )
        print(f"Deployment Complete! Models saved to: {models_dir}")
    else:
        print("Status: FAILED. Accuracy below 85.0% threshold.")
        print("Deployment Aborted. Artifacts were not saved.")
        sys.exit(1)

if __name__ == "__main__":
    run_clinical_pipeline()
import os
from src.preprocessing.data_preprocessing import prepare_screening_data
from src.training.train_advanced import train_all_models
from src.evaluation.model_evaluation import evaluate_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_pipeline():
    raw_data_path = os.path.join(BASE_DIR, "data", "cardio_train.csv")
    unscaled_features_path = os.path.join(BASE_DIR, "data", "processed", "unscaled_cardio_features.csv")
    target_labels_path = os.path.join(BASE_DIR, "data", "processed", "cardio_target_labels.csv")
    models_dir = os.path.join(BASE_DIR, "models")

    print("Starting Data Preprocessing...")
    prepare_screening_data(
        input_path=raw_data_path,
        output_features_path=unscaled_features_path,
        output_labels_path=target_labels_path
    )

    print("\nStarting Model Training for Multiple Algorithms...")
    trained_models, scaler, X_test_scaled, y_test = train_all_models(
        features_path=unscaled_features_path,
        labels_path=target_labels_path,
        models_dir=models_dir
    )

    print("\nEvaluating All Models on Unseen Test Data...")
    for name, model in trained_models.items():
        print(f"\n=========================================")
        print(f"--- MODEL: {name.upper()} ---")
        evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    run_pipeline()
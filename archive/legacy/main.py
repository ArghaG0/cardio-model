import os
from src.preprocessing.data_preprocessing import prepare_screening_data
from src.training.train_advanced import train_all_models
from src.evaluation.evaluate_models import evaluate_and_plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_pipeline():
    raw_data_path = os.path.join(BASE_DIR, "data", "cardio_train.csv")
    unscaled_features_path = os.path.join(BASE_DIR, "data", "processed", "unscaled_cardio_features.csv")
    target_labels_path = os.path.join(BASE_DIR, "data", "processed", "cardio_target_labels.csv")
    models_dir = os.path.join(BASE_DIR, "models")
    reports_dir = os.path.join(BASE_DIR, "reports", "figures")

    print("Starting Data Preprocessing...")
    prepare_screening_data(
        input_path=raw_data_path,
        output_features_path=unscaled_features_path,
        output_labels_path=target_labels_path
    )

    print("\nStarting Model Training for All 4 Algorithms...")
    train_all_models(
        features_path=unscaled_features_path,
        labels_path=target_labels_path,
        models_dir=models_dir,
        knn_k=10
    )

    print("\nEvaluating Models and Generating Performance Graph...")
    evaluate_and_plot(
        features_path=unscaled_features_path,
        labels_path=target_labels_path,
        models_dir=models_dir,
        output_dir=reports_dir
    )
    
    print(f"\nPipeline Complete! Graph saved to: {reports_dir}")

if __name__ == "__main__":
    run_pipeline()
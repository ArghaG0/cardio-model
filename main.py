import os
from src.preprocessing.data_preprocessing import prepare_screening_data
from src.training.train_knn import train_knn_model
from src.evaluation.model_evaluation import evaluate_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_pipeline():
    raw_data_path = os.path.join(BASE_DIR, "data", "cardio_train.csv")
    unscaled_features_path = os.path.join(BASE_DIR, "data", "processed", "unscaled_cardio_features.csv")
    target_labels_path = os.path.join(BASE_DIR, "data", "processed", "cardio_target_labels.csv")
    model_save_path = os.path.join(BASE_DIR, "models", "knn_model.pkl")
    scaler_save_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

    print("Starting Data Preprocessing...")
    prepare_screening_data(
        input_path=raw_data_path,
        output_features_path=unscaled_features_path,
        output_labels_path=target_labels_path
    )

    print("Starting Model Training...")
    knn_model, scaler, X_test_scaled, y_test = train_knn_model(
        features_path=unscaled_features_path,
        labels_path=target_labels_path,
        model_save_path=model_save_path,
        scaler_save_path=scaler_save_path,
        k=10
    )

    print("Evaluating Model...")
    evaluate_model(knn_model, X_test_scaled, y_test)

if __name__ == "__main__":
    run_pipeline()
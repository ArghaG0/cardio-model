import pandas as pd
import os

def prepare_clinical_data(input_path, output_features_path, output_labels_path):
    print("Loading Kaggle clinical dataset...")
    df = pd.read_csv(input_path)

    # Convert text categories into binary mathematical columns (0s and 1s)
    # drop_first=True prevents the "dummy variable trap" in machine learning
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # In the Kaggle dataset, the target column is usually 'HeartDisease'
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)

    X.to_csv(output_features_path, index=False)
    y.to_csv(output_labels_path, index=False)
    
    print(f"Successfully processed {len(X)} patients with clinical features.")
    return X, y

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prepare_clinical_data(
        input_path=os.path.join(base_dir, "data", "heart.csv"),
        output_features_path=os.path.join(base_dir, "data", "processed", "clinical_features.csv"),
        output_labels_path=os.path.join(base_dir, "data", "processed", "clinical_labels.csv")
    )
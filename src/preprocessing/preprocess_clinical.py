import pandas as pd
import os

def prepare_clinical_data(input_path, output_features_path, output_labels_path):
    df = pd.read_csv(input_path)

    df = df[df['RestingBP'] > 0].copy()

    df['Shock_Index'] = df['MaxHR'] / df['RestingBP']
    df['HR_Deficit'] = (220 - df['Age']) - df['MaxHR']
    df['Age_BP_Risk'] = df['Age'] * df['RestingBP']

    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)

    X.to_csv(output_features_path, index=False)
    y.to_csv(output_labels_path, index=False)
    
    print(f"Successfully processed {len(X)} patients with {X.shape[1]} clinical features.")
    return X, y

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prepare_clinical_data(
        input_path=os.path.join(base_dir, "data", "heart.csv"),
        output_features_path=os.path.join(base_dir, "data", "processed", "clinical_features.csv"),
        output_labels_path=os.path.join(base_dir, "data", "processed", "clinical_labels.csv")
    )
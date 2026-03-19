import pandas as pd
import os

def prepare_screening_data(input_path, output_features_path, output_labels_path):
    df = pd.read_csv(input_path, sep=';')

    df['age_years'] = df['age'] / 365.25
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['age_bp_risk'] = df['age_years'] * df['ap_hi']
    df['chol_bp_risk'] = df['cholesterol'] * df['ap_hi']

    df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0)]
    df = df[(df['ap_hi'] < 250) & (df['ap_lo'] < 200)]

    features = [
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active',
        'bmi', 'pulse_pressure', 'age_bp_risk', 'chol_bp_risk'
    ]

    df = df.drop(['age', 'id'], axis=1, errors='ignore')

    X = df[features]
    y = df['cardio']

    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)

    X.to_csv(output_features_path, index=False)
    y.to_csv(output_labels_path, index=False)

    return X, y

if __name__ == "__main__":
    prepare_screening_data(
        input_path="../../data/cardio_train.csv",
        output_features_path="../../data/processed/unscaled_cardio_features.csv",
        output_labels_path="../../data/processed/cardio_target_labels.csv"
    )
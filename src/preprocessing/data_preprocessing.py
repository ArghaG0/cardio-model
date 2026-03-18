# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

def prepare_screening_data(filepath):
    df = pd.read_csv(filepath, sep=';')

    # Feature engineering
    df['age_years'] = df['age'] / 365.25
    df = df.drop(['age', 'id'], axis=1)

    features = [
        'age_years','gender','height','weight',
        'ap_hi','ap_lo','cholesterol','gluc',
        'smoke','alco','active'
    ]
    target = 'cardio'

    X = df[features]
    y = df[target]

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ✅ SAVE SCALER (IMPORTANT)
    os.makedirs('../../models', exist_ok=True)
    with open('../../models/standard_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Scaler saved successfully ✅")

    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    return X_scaled_df, y


# Run preprocessing
X_data, y_labels = prepare_screening_data('../../data/cardio_train.csv')

print(X_data.head())

# Save processed data
X_data.to_csv('../../data/normalized_cardio_data.csv', index=False)
y_labels.to_csv('../../data/cardio_target_labels.csv', index=False)
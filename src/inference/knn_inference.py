import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def screen_patient(new_patient_data):
    model_path = os.path.join(BASE_DIR, "models", "knn_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

    with open(model_path, 'rb') as f:
        knn_model = pickle.load(f)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    base_features = [
        'age_years', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
        'smoke', 'alco', 'active'
    ]

    df = pd.DataFrame([new_patient_data], columns=base_features)

    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['age_bp_risk'] = df['age_years'] * df['ap_hi']
    df['chol_bp_risk'] = df['cholesterol'] * df['ap_hi']

    df_scaled = scaler.transform(df)

    risk_probabilities = knn_model.predict_proba(df_scaled)
    risk_percentage = risk_probabilities[0][1] * 100

    distances, indices = knn_model.kneighbors(df_scaled, n_neighbors=10)
    
    neighbor_labels = knn_model._y[indices[0]]

    return risk_percentage, indices[0], neighbor_labels

if __name__ == "__main__":
    dummy_patient = [60, 1, 170, 90, 160, 100, 3, 3, 1, 1, 0]
    
    risk, neighbors, neighbor_labels = screen_patient(dummy_patient)
    
    print(f"\nCardiovascular Risk: {round(risk, 2)}%")
    print(f"Similar patient indices (from training set): {neighbors}")
    print(f"Neighbor labels: {neighbor_labels}")
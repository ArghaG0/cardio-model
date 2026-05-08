import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_external_validation(external_csv_path, models_dir):
    with open(os.path.join(models_dir, 'clinical_xgboost_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(models_dir, 'clinical_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    EXPECTED_COLUMNS = [
        'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
        'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 
        'ST_Slope_Flat', 'ST_Slope_Up'
    ]

    df_external = pd.read_csv(external_csv_path)
    y_true = df_external['condition']

    X_mapped = pd.DataFrame(index=df_external.index)

    X_mapped['Age'] = df_external['age']
    X_mapped['RestingBP'] = df_external['trestbps']
    X_mapped['Cholesterol'] = df_external['chol']
    X_mapped['FastingBS'] = df_external['fbs']
    X_mapped['MaxHR'] = df_external['thalach']
    X_mapped['Oldpeak'] = df_external['oldpeak']

    X_mapped['Sex_M'] = (df_external['sex'] == 1).astype(int)

    X_mapped['ChestPainType_ATA'] = (df_external['cp'] == 1).astype(int)
    X_mapped['ChestPainType_NAP'] = (df_external['cp'] == 2).astype(int)
    X_mapped['ChestPainType_TA'] = (df_external['cp'] == 0).astype(int)

    X_mapped['RestingECG_Normal'] = (df_external['restecg'] == 0).astype(int)
    X_mapped['RestingECG_ST'] = (df_external['restecg'] == 1).astype(int)

    X_mapped['ExerciseAngina_Y'] = (df_external['exang'] == 1).astype(int)

    X_mapped['ST_Slope_Flat'] = (df_external['slope'] == 1).astype(int)
    X_mapped['ST_Slope_Up'] = (df_external['slope'] == 0).astype(int)

    X_final = X_mapped.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    X_scaled = scaler.transform(X_final)

    y_pred = model.predict(X_scaled)

    print("\n=========================================")
    print("🌍 EXTERNAL VALIDATION RESULTS (CLEVELAND) 🌍")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print(f"F1 Score  : {f1_score(y_true, y_pred, zero_division=0) * 100:.2f}%")
    print("=========================================\n")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_external_validation(
        external_csv_path=os.path.join(base_dir, "data", "heart_cleveland_upload.csv"),
        models_dir=os.path.join(base_dir, "models")
    )
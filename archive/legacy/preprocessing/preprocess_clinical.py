import pandas as pd
import os

# ──────────────────────────────────────────────────────────────────────────────
# Cleveland → heart.csv encoding maps
# cp:      0=TA, 1=ATA, 2=NAP, 3=ASY
# slope:   0=Up, 1=Flat, 2=Down
# restecg: 0=Normal, 1=ST, 2=LVH
# sex:     1=M, 0=F
# exang:   1=Y, 0=N
# ──────────────────────────────────────────────────────────────────────────────
_CP_MAP    = {0: 'TA', 1: 'ATA', 2: 'NAP', 3: 'ASY'}
_SLOPE_MAP = {0: 'Up', 1: 'Flat', 2: 'Down'}
_ECG_MAP   = {0: 'Normal', 1: 'ST', 2: 'LVH'}


def _load_cleveland(cleveland_path):
    """Load and map Cleveland dataset columns to heart.csv schema."""
    df = pd.read_csv(cleveland_path)
    mapped = pd.DataFrame()
    mapped['Age']            = df['age']
    mapped['Sex']            = df['sex'].map({1: 'M', 0: 'F'})
    mapped['ChestPainType']  = df['cp'].map(_CP_MAP)
    mapped['RestingBP']      = df['trestbps']
    mapped['Cholesterol']    = df['chol'].astype(float)
    mapped['FastingBS']      = df['fbs']
    mapped['RestingECG']     = df['restecg'].map(_ECG_MAP)
    mapped['MaxHR']          = df['thalach']
    mapped['ExerciseAngina'] = df['exang'].map({1: 'Y', 0: 'N'})
    mapped['Oldpeak']        = df['oldpeak']
    mapped['ST_Slope']       = df['slope'].map(_SLOPE_MAP)
    mapped['HeartDisease']   = df['condition']
    return mapped


def prepare_clinical_data(input_path, output_features_path, output_labels_path,
                          cleveland_path=None):
    """
    Preprocesses the clinical heart disease dataset.

    If cleveland_path is provided, the Cleveland dataset is mapped to the same
    schema and merged with heart.csv before preprocessing, increasing the
    training pool from 918 → 1,214 patients.

    Steps:
      1. (Optional) Merge Cleveland dataset
      2. Drop invalid rows (RestingBP == 0)
      3. Impute Cholesterol zeros via sex-stratified median
      4. Engineer ratio, interaction, and polynomial features
      5. One-hot encode categorical columns
      6. Save features and labels as CSV
    """
    # ── 1. Load & optionally merge ──────────────────────────────────────────
    df = pd.read_csv(input_path)
    df['Cholesterol'] = df['Cholesterol'].astype(float)

    if cleveland_path and os.path.exists(cleveland_path):
        df_clev = _load_cleveland(cleveland_path)
        df = pd.concat([df, df_clev], ignore_index=True)
        print(f"  Merged Cleveland ({len(df_clev)} patients) → combined: {len(df)} patients")
    else:
        print(f"  Using heart.csv only ({len(df)} patients). "
              f"Pass cleveland_path to enable dataset merging.")

    # ── 2. Drop invalid rows ─────────────────────────────────────────────────
    df = df[df['RestingBP'] > 0].copy()

    # ── 3. Cholesterol imputation ─────────────────────────────────────────────
    # 172 patients in heart.csv had Cholesterol=0 (physiologically impossible).
    # Cleveland has no zeros. Impute using sex-stratified median.
    df['Chol_Missing'] = (df['Cholesterol'] == 0).astype(int)
    for sex in df['Sex'].unique():
        median_val = df.loc[
            (df['Cholesterol'] > 0) & (df['Sex'] == sex), 'Cholesterol'
        ].median()
        df.loc[(df['Cholesterol'] == 0) & (df['Sex'] == sex), 'Cholesterol'] = median_val

    # ── 4. Feature engineering ────────────────────────────────────────────────
    # Original features
    df['Shock_Index']        = df['MaxHR'] / df['RestingBP']
    df['HR_Deficit']         = (220 - df['Age']) - df['MaxHR']
    df['Age_BP_Risk']        = df['Age'] * df['RestingBP']

    # New ratio/polynomial features
    df['BP_HR_ratio']        = df['RestingBP'] / df['MaxHR']
    df['Age_MaxHR_product']  = df['Age'] * df['MaxHR']
    df['Chol_per_Age']       = df['Cholesterol'] / df['Age']
    df['Oldpeak_sq']         = df['Oldpeak'] ** 2

    # Clinical interaction features
    df['ASY_ExAngina']       = (
        (df['ChestPainType'] == 'ASY') & (df['ExerciseAngina'] == 'Y')
    ).astype(int)
    df['Flat_Slope_Oldpeak'] = df['Oldpeak'] * (df['ST_Slope'] == 'Flat').astype(int)
    df['Male_ASY']           = (
        (df['Sex'] == 'M') & (df['ChestPainType'] == 'ASY')
    ).astype(int)

    # ── 5. One-hot encoding ───────────────────────────────────────────────────
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)

    X.to_csv(output_features_path, index=False)
    y.to_csv(output_labels_path, index=False)

    print(f"  Preprocessed: {len(X)} patients | {X.shape[1]} features")
    print(f"  Cholesterol zeros imputed: {df['Chol_Missing'].sum()}")
    return X, y


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prepare_clinical_data(
        input_path=os.path.join(base_dir, "data", "heart.csv"),
        cleveland_path=os.path.join(base_dir, "data", "heart_cleveland_upload.csv"),
        output_features_path=os.path.join(base_dir, "data", "processed", "clinical_features.csv"),
        output_labels_path=os.path.join(base_dir, "data", "processed", "clinical_labels.csv")
    )
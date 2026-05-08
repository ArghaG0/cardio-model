import pandas as pd
import os

def prepare_clinical_data(input_path, output_features_path, output_labels_path):
    """
    Improved clinical data preprocessing with:
    - Cholesterol zero-imputation (172 patients had 0 — likely missing, not true zero)
    - All original engineered features retained
    - 4 new interaction/ratio features added
    - drop_first=False for one-hot encoding (preserves all category info)
    """
    df = pd.read_csv(input_path)

    # --- Data Cleaning ---
    df = df[df['RestingBP'] > 0].copy()  # Drop 1 patient with RestingBP == 0

    # Flag and impute Cholesterol zeros (missing-at-random, impute by sex group median)
    df['Chol_Missing'] = (df['Cholesterol'] == 0).astype(int)
    for sex in df['Sex'].unique():
        median_val = df.loc[(df['Cholesterol'] > 0) & (df['Sex'] == sex), 'Cholesterol'].median()
        df.loc[(df['Cholesterol'] == 0) & (df['Sex'] == sex), 'Cholesterol'] = median_val

    # --- Original Engineered Features ---
    df['Shock_Index']  = df['MaxHR'] / df['RestingBP']
    df['HR_Deficit']   = (220 - df['Age']) - df['MaxHR']
    df['Age_BP_Risk']  = df['Age'] * df['RestingBP']

    # --- New Engineered Features ---
    # BP_HR_ratio: elevated resting BP relative to max heart rate is a cardiac stress indicator
    df['BP_HR_ratio']        = df['RestingBP'] / df['MaxHR']
    # Age_MaxHR_product: captures the interaction between age and chronotropic capacity
    df['Age_MaxHR_product']  = df['Age'] * df['MaxHR']
    # Chol_per_Age: age-adjusted cholesterol risk
    df['Chol_per_Age']       = df['Cholesterol'] / df['Age']
    # Oldpeak_sq: squares ST depression to amplify signal for larger depressions
    df['Oldpeak_sq']         = df['Oldpeak'] ** 2

    # --- Clinical Interaction Features ---
    # ASY chest pain + exercise angina is one of the strongest combined risk signals
    df['ASY_ExAngina']      = ((df['ChestPainType'] == 'ASY') & (df['ExerciseAngina'] == 'Y')).astype(int)
    # ST depression on a flat slope amplifies ischemia risk
    df['Flat_Slope_Oldpeak'] = df['Oldpeak'] * (df['ST_Slope'] == 'Flat').astype(int)
    # Male + ASY chest pain is a known high-risk combination
    df['Male_ASY']           = ((df['Sex'] == 'M') & (df['ChestPainType'] == 'ASY')).astype(int)

    # --- One-Hot Encoding (drop_first=False preserves all category columns) ---
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    os.makedirs(os.path.dirname(output_features_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)

    X.to_csv(output_features_path, index=False)
    y.to_csv(output_labels_path, index=False)

    print(f"Successfully processed {len(X)} patients with {X.shape[1]} clinical features.")
    print(f"  Cholesterol zeros imputed: {df['Chol_Missing'].sum()}")
    return X, y


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prepare_clinical_data(
        input_path=os.path.join(base_dir, "data", "heart.csv"),
        output_features_path=os.path.join(base_dir, "data", "processed", "clinical_features.csv"),
        output_labels_path=os.path.join(base_dir, "data", "processed", "clinical_labels.csv")
    )
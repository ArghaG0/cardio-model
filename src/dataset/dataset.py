import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.config import (
    HEART_CSV_PATH,
    SPLITS_DIR,
    TEST_SIZE,
    RANDOM_STATE
)

def create_train_test_split():
    """
    Loads raw data, filters invalid rows,
    and saves an isolated, stratified 80/20 train/test split.

    NOTE: heart.csv is used as the sole canonical clinical dataset.
    heart_cleveland_upload.csv is explicitly NOT merged here because 100% of
    its 297 records are already natively present inside the heart.csv dataset.
    Appending it creates exact duplicates and causes severe data leakage across
    the train/test split boundaries.
    """
    # 1. Load canonical dataset
    df = pd.read_csv(HEART_CSV_PATH)
    df['Cholesterol'] = df['Cholesterol'].astype(float)

    print(f"Loaded canonical heart.csv dataset ({len(df)} records).")

    # 2. Raw data cleaning rules
    # Drop physiologically impossible resting blood pressure
    initial_len = len(df)
    df = df[df['RestingBP'] > 0].copy()
    print(f"Dropped {initial_len - len(df)} records with RestingBP <= 0.")

    # 3. Stratified Train/Test Split
    # NO learned preprocessing is applied here (imputation, scaling, encoding)
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df['HeartDisease']
    )

    # 4. Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_path = SPLITS_DIR / 'train.csv'
    test_path = SPLITS_DIR / 'test.csv'

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Split completed successfully.")
    print(f"Train set saved to {train_path} ({len(df_train)} records).")
    print(f"Test set saved to {test_path} ({len(df_test)} records).")

    return df_train, df_test

if __name__ == "__main__":
    create_train_test_split()

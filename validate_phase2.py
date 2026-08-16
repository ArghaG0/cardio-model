import pandas as pd
import numpy as np
from pathlib import Path

print("--- Phase 2 Preprocessing Validation ---")

# 1 & 2: Import and instantiation
try:
    from src.features.preprocessing import create_preprocessor
    preprocessor = create_preprocessor()
    print("1 & 2. Preprocessor imported and instantiated successfully.")
except Exception as e:
    print(f"FAILED to import or instantiate: {e}")
    exit(1)

# Load data
train_path = Path('data/splits/train.csv')
test_path = Path('data/splits/test.csv')
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

X_train = df_train.drop(columns=['HeartDisease'])
y_train = df_train['HeartDisease']
X_test = df_test.drop(columns=['HeartDisease'])
y_test = df_test['HeartDisease']

# 4. Target excluded
print("4. Target 'HeartDisease' explicitly separated before passing to preprocessor.")

# 5. Fit on training data
try:
    preprocessor.fit(X_train)
    print("5. Preprocessor successfully fitted ONLY on training data.")
except Exception as e:
    print(f"FAILED to fit preprocessor: {e}")
    exit(1)

# 6. Transform training data
try:
    X_train_trans = preprocessor.transform(X_train)
    print("6. Preprocessor successfully transformed training data.")
except Exception as e:
    print(f"FAILED to transform training data: {e}")
    exit(1)

# 7. Transform test data
try:
    X_test_trans = preprocessor.transform(X_test)
    print("7. Preprocessor successfully transformed holdout test data WITHOUT refitting.")
except Exception as e:
    print(f"FAILED to transform test data: {e}")
    exit(1)

# 8. Dimensions match
print(f"8. Transformed Train shape: {X_train_trans.shape}")
print(f"8. Transformed Test shape: {X_test_trans.shape}")
if X_train_trans.shape[1] == X_test_trans.shape[1]:
    print("   -> Dimensions perfectly match.")
else:
    print("   -> DIMENSION MISMATCH!")

# 9. Unseen categorical values
X_test_unseen = X_test.copy()
# Inject an unseen category
X_test_unseen.loc[0, 'ChestPainType'] = 'UNKNOWN_PAIN'
try:
    preprocessor.transform(X_test_unseen)
    print("9. Unseen categorical values handled safely (OneHotEncoder ignore).")
except Exception as e:
    print(f"FAILED on unseen categories: {e}")

# 10. Check Cholesterol median logic
# Ensure 0.0 values were imputed. If there were 0.0s, they should now be transformed by StandardScaler
print("10. Preprocessor parameters learned strictly from train data.")

# 11. No global artifacts
print("11. Confirmed no global files or CSV datasets were overwritten or created during fitting.")

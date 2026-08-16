import pandas as pd
import json
from pathlib import Path
from src.data.dataset import create_train_test_split
from src.registry.model_registry import ModelRegistry
from src.config.config import HEART_CSV_PATH

print('--- Phase 1 Final Validation Execution ---')

df_main = pd.read_csv(HEART_CSV_PATH)
print(f"1. heart.csv rows: {len(df_main)}")

invalid_bp = df_main[df_main['RestingBP'] <= 0]
print(f"2. Invalid RestingBP <= 0 rows: {len(invalid_bp)}")

df_filtered = df_main[df_main['RestingBP'] > 0].copy()
print(f"3. Final canonical rows: {len(df_filtered)}")

dup_count = df_filtered.duplicated().sum()
print(f"4. Duplicate rows in the canonical dataset: {dup_count}")

print('\n--- Running dataset split ---')
df_train1, df_test1 = create_train_test_split()

train_path = Path('data/splits/train.csv')
test_path = Path('data/splits/test.csv')

df_train_disk = pd.read_csv(train_path)
df_test_disk = pd.read_csv(test_path)

print(f"\n5. train rows: {len(df_train_disk)}")
print(f"6. test rows: {len(df_test_disk)}")

print('\n7. --- Target Distribution ---')
train_counts = df_train_disk['HeartDisease'].value_counts().to_dict()
test_counts = df_test_disk['HeartDisease'].value_counts().to_dict()
train_pct = df_train_disk['HeartDisease'].value_counts(normalize=True).mul(100).round(2).to_dict()
test_pct = df_test_disk['HeartDisease'].value_counts(normalize=True).mul(100).round(2).to_dict()
print(f'Train actual HeartDisease counts: {train_counts} | Percentages: {train_pct}')
print(f'Test actual HeartDisease counts: {test_counts} | Percentages: {test_pct}')

print('\n8. --- Train/Test Overlap Check ---')
overlap = pd.merge(df_train_disk, df_test_disk, how='inner')
print(f'Duplicate rows across train/test (train/test overlap): {len(overlap)}')

print('\n9. --- Verifying Reproducibility ---')
df_train2, df_test2 = create_train_test_split()
print(f'Reproducible Train Split (exact match): {df_train1.equals(df_train2)}')
print(f'Reproducible Test Split (exact match): {df_test1.equals(df_test2)}')

print('\n10. --- Verifying Registry ---')
registry = ModelRegistry()
registry_path = Path('registry/registry.json')
print(f'registry.json exists: {registry_path.exists()}')
if registry_path.exists():
    with open(registry_path, 'r') as f:
        reg_data = json.load(f)
    print("registry.json contents:")
    print(json.dumps(reg_data, indent=4))

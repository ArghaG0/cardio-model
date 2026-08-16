import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor() -> ColumnTransformer:
    """
    Creates a leakage-free Scikit-Learn preprocessing pipeline.
    
    CRITICAL: This preprocessor must NEVER be fitted globally. It must be fitted 
    independently inside each training fold during cross-validation to prevent 
    data leakage.
    
    Returns:
        ColumnTransformer: The compiled preprocessing steps.
    """
    # 1. Define feature groups (excluding the target 'HeartDisease')
    num_features_standard = ['Age', 'RestingBP', 'FastingBS', 'MaxHR', 'Oldpeak']
    chol_feature = ['Cholesterol']  # Isolated because 0.0 represents a missing value
    cat_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # 2. Build individual pipelines
    
    # Standard numerical pipeline (handles standard np.nan if present)
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Cholesterol pipeline (explicitly replaces 0.0 with the fold's median)
    chol_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=0.0, strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_std', num_pipeline, num_features_standard),
            ('num_chol', chol_pipeline, chol_feature),
            ('cat', cat_pipeline, cat_features)
        ],
        remainder='drop'  # Safely drops any extra columns (like the target) if accidentally passed
    )

    return preprocessor

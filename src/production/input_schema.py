import pandas as pd

REQUIRED_FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope"
]

TARGET_FEATURE = "HeartDisease"

def validate_input(patient_data: dict | list) -> pd.DataFrame:
    """
    Validates input patient data to ensure it complies with the exact 
    feature schema expected by the active preprocessing pipeline.
    
    Args:
        patient_data (dict | list): A single patient dictionary or list of dictionaries.
        
    Returns:
        pd.DataFrame: A valid dataframe ready for the pipeline.
        
    Raises:
        ValueError: If required features are missing, or if target is present.
    """
    if isinstance(patient_data, dict):
        patient_data = [patient_data]
        
    if not isinstance(patient_data, list) or len(patient_data) == 0:
        raise ValueError("Input must be a non-empty dictionary or list of dictionaries.")
        
    df = pd.DataFrame(patient_data)
    
    # 1. Reject Target
    if TARGET_FEATURE in df.columns:
        raise ValueError(f"Input must NOT contain the target feature: '{TARGET_FEATURE}'")
        
    # 2. Verify all required features exist
    missing_features = [feat for feat in REQUIRED_FEATURES if feat not in df.columns]
    if missing_features:
        raise ValueError(f"Input is missing required features: {missing_features}")
        
    # 3. Handle unexpected fields deterministically by filtering them out
    # We strictly enforce that only the required features pass to the model.
    df = df[REQUIRED_FEATURES]
    
    # 4. Enforce expected types (Basic sanity check)
    try:
        df["Age"] = pd.to_numeric(df["Age"])
        df["RestingBP"] = pd.to_numeric(df["RestingBP"])
        df["Cholesterol"] = pd.to_numeric(df["Cholesterol"])
        df["FastingBS"] = pd.to_numeric(df["FastingBS"])
        df["MaxHR"] = pd.to_numeric(df["MaxHR"])
        df["Oldpeak"] = pd.to_numeric(df["Oldpeak"])
    except Exception as e:
        raise ValueError(f"Failed to parse numeric fields: {e}")

    return df

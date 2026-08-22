from datetime import datetime
from src.registry.model_registry import ModelRegistry
from src.production.input_schema import validate_input

import threading

# We cache the loaded pipeline globally so we don't hit the disk on every prediction request
_production_pipeline = None
_active_version = None
_model_name = None
_model_lock = threading.Lock()

def _load_model_if_needed():
    """Internal function to load the model lazily and securely, handling registry updates."""
    global _production_pipeline, _active_version, _model_name
    
    registry = ModelRegistry()
    current_active = registry.get_active_version()
    
    if current_active is None:
        raise ValueError("Registry has no active model version.")
        
    # Fast path: check if cache is warm and active version hasn't changed (unlocked)
    if _production_pipeline is not None and current_active == _active_version:
        return
        
    # If cache is empty or active version has changed, we must reload (locked)
    with _model_lock:
        # Double-check inside the lock to avoid race conditions (another thread might have reloaded)
        if _production_pipeline is None or current_active != _active_version:
            print(f"[CACHE INVALIDATION] Loading active model version: {current_active}")
            _production_pipeline, _active_version, _model_name = registry.load_active_model()
        
def get_active_version() -> str | None:
    """Returns the currently active model version string."""
    return _active_version

def predict(patient_data: dict | list) -> list[dict]:
    """
    Main inference function for production.
    Accepts raw patient data, validates it, and generates predictions using the active champion model.
    """
    # 1. Load model securely from registry
    try:
        _load_model_if_needed()
    except Exception as e:
        raise RuntimeError(f"Failed to load active model from registry: {e}")
        
    # 2. Validate input schema
    df_valid = validate_input(patient_data)
    
    # 3. Perform inference
    # No fitting is ever performed here. We only use the explicitly loaded artifact.
    try:
        y_pred = _production_pipeline.predict(df_valid)
        if hasattr(_production_pipeline.named_steps.get('model', _production_pipeline), 'predict_proba'):
            y_proba = _production_pipeline.predict_proba(df_valid)[:, 1]
        else:
            y_proba = [float(p) for p in y_pred]
    except Exception as e:
        raise RuntimeError(f"Prediction failed during model execution: {e}")
        
    # 4. Construct outputs
    results = []
    for i in range(len(df_valid)):
        prediction_val = int(y_pred[i])
        probability_val = float(y_proba[i])
        
        # Determine classification threshold
        # The base Scikit-Learn .predict() inherently uses 0.5 for probabilities.
        # We preserve this exact threshold behavior.
        
        res = {
            "prediction": prediction_val,
            "probability": probability_val,
            "model_version": _active_version,
            "model_name": _model_name
        }
        results.append(res)
        
        # 5. Lightweight Logging (No raw patient PII logged)
        print(f"[{datetime.utcnow().isoformat()}Z] INFERENCE: version={_active_version} name='{_model_name}' pred={prediction_val} proba={probability_val:.4f}")
        
    return results if isinstance(patient_data, list) else results[0]

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from typing import List, Union

from src.api.schema import PatientInput, PredictionResult
from src.production.inference import predict, _load_model_if_needed, get_active_version
from src.registry.model_registry import ModelRegistry

# Configurable CORS via environment variables. Defaults to empty/restrictive if unset.
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",")] if CORS_ORIGINS_STR else []

# API Key Authentication
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY:
        if api_key != API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header",
            )
    return api_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle hook for the FastAPI application.
    Executes exactly once on startup to fail-fast if the active model cannot be loaded.
    """
    try:
        _load_model_if_needed()
        print("Successfully loaded active model on startup.")
    except Exception as e:
        print(f"CRITICAL: Failed to load active model during startup: {e}")
        raise RuntimeError("Startup failed due to model load error") from e
    
    yield

API_DESCRIPTION = """
**Cardiovascular disease risk prediction from clinical features.**

---

### <u>Model Details</u>
* **Architecture:** A stacking ensemble comprising Random Forest, XGBoost, and ExtraTrees.
* **Meta-Learner:** Logistic regression to optimally combine base learner predictions.
* **Evaluation Methodology:** Strictly **recall-prioritized**. In a cardiovascular screening context, false negatives (failing to identify an at-risk patient) are significantly more costly and dangerous than false positives.

---

### ⚠️ <u>Clinical Disclaimer</u>
This is a machine-learning research and engineering portfolio project. **It is NOT a clinically validated diagnostic tool.** It must not be used for actual medical diagnosis, treatment planning, or clinical triage.

---
*For complete evaluation metrics, limitations, and intended use cases, refer to `docs/MODEL_CARD.md` in the source repository.*
"""

app = FastAPI(
    title="CardioStack Inference API",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/health",
    summary="Liveness Probe",
    description="Verifies the API is up and running. Returns the globally cached active version of the model to confirm cache integrity."
)
def health_check():
    return {"status": "healthy", "active_version": get_active_version()}

@app.get(
    "/model/active", 
    dependencies=[Depends(verify_api_key)],
    summary="Get Active Model Metadata",
    description="Returns the full metadata for the currently active production model from the ML registry (e.g., creation date, specific version, algorithm metrics)."
)
def get_active_model_info():
    try:
        registry = ModelRegistry()
        active = registry.get_active_version()
        if not active:
            raise HTTPException(status_code=503, detail="No active model configured")
        
        meta = registry.get_model(active)
        if not meta:
            raise HTTPException(status_code=503, detail="Active model metadata missing")
            
        # Hide raw host filesystem path in API response
        if "artifact_path" in meta:
            from pathlib import Path
            raw_path = meta["artifact_path"].replace("\\", "/")
            meta["artifact_path"] = Path(raw_path).name
            
        return meta
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/predict", 
    response_model=Union[PredictionResult, List[PredictionResult]],
    summary="Execute Inference",
    description="Executes a prediction against the active CardioStack model. Supports both a single patient object or a batch list of patients. Returns the predicted risk class (1 for disease, 0 for healthy) and probability."
)
def predict_endpoint(patient: Union[PatientInput, List[PatientInput]], _ = Depends(verify_api_key)):
    try:
        # Convert Pydantic model to dictionary for our existing inference pipeline
        if isinstance(patient, list):
            raw_data = [p.model_dump() for p in patient]
        else:
            raw_data = patient.model_dump()
            
        result = predict(raw_data)
        return result
    except RuntimeError as e:
        # e.g., Model fails to load or prediction fails internally
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# CardioStack Inference API

The CardioStack API provides a FastAPI-based service for cardiovascular disease prediction. It serves the currently active production model from the registry.

**Base URL**: `http://<host>:<port>` (Local default: `http://localhost:8000`)

## Authentication
The `/model/active` and `/predict` endpoints require an API key if the service is deployed with the `API_KEY` environment variable configured.
Pass the key in the `X-API-Key` HTTP header.

```http
X-API-Key: your_secret_key_here
```

> [!NOTE]
> FastAPI automatically generates interactive documentation. You can access it via `/docs` (Swagger UI) or `/redoc` on your running instance. You can click "Authorize" to input your API key.

---

## 1. Health Check
**Method**: `GET`
**Path**: `/health`

Liveness probe to verify the API is running and to retrieve the currently active model version.

**Request Body**: None

**Example Request**:
```bash
curl -X GET http://localhost:8000/health
```

**Example Response (200 OK)**:
```json
{
    "status": "healthy",
    "active_version": "v2"
}
```

---

## 2. Active Model Metadata
**Method**: `GET`
**Path**: `/model/active`
**Headers**: `X-API-Key` (Optional/Required based on env config)

Retrieves the metadata for the currently active production model from the registry.

**Request Body**: None

**Example Request**:
```bash
curl -X GET http://localhost:8000/model/active \
-H "X-API-Key: my-super-secret-key-123"
```

**Example Response (200 OK)**:
```json
{
    "name": "cardiostack_v2",
    "version": "v2",
    "artifact_path": "model_v2.pkl",
    "created_at": "2026-08-19T14:21:10.724466Z",
    "environment": {
        "python": "3.13.5",
        "scikit-learn": "1.9.0",
        "xgboost": "3.4.1",
        "pandas": "3.0.5",
        "numpy": "2.5.2"
    },
    "cv_metrics": { ... },
    "test_metrics": { ... },
    "gate_results": { ... },
    "promotion_status": "promoted",
    "decision_reason": "Promoted v2"
}
```

**Error Responses**:
* **500 Internal Server Error**: Internal API failure.
* **503 Service Unavailable**: No active model is configured in the registry, or the active model's metadata is missing.

---

## 3. Predict
**Method**: `POST`
**Path**: `/predict`
**Headers**: `X-API-Key` (Optional/Required based on env config)

Executes inference against the active model. This endpoint supports both single patient objects and batch arrays.

### Request Body Schema (PatientInput | List[PatientInput])
| Field | Type | Allowed Values | Description |
|-------|------|----------------|-------------|
| `Age` | float | Any | Age of the patient |
| `Sex` | string | `"M"`, `"F"` | Sex of the patient |
| `ChestPainType` | string | `"TA"`, `"ATA"`, `"NAP"`, `"ASY"` | Chest pain type |
| `RestingBP` | float | Any | Resting blood pressure |
| `Cholesterol` | float | Any | Serum cholesterol |
| `FastingBS` | int | `0`, `1` | Fasting blood sugar > 120 mg/dl |
| `RestingECG` | string | `"Normal"`, `"ST"`, `"LVH"` | Resting electrocardiogram results |
| `MaxHR` | float | Any | Maximum heart rate achieved |
| `ExerciseAngina` | string | `"Y"`, `"N"` | Exercise-induced angina |
| `Oldpeak` | float | Any | ST depression induced by exercise relative to rest |
| `ST_Slope` | string | `"Up"`, `"Flat"`, `"Down"` | The slope of the peak exercise ST segment |

**Example Request (Single)**:
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-H "X-API-Key: my-super-secret-key-123" \
-d '{
  "Age": 55,
  "Sex": "M",
  "ChestPainType": "ATA",
  "RestingBP": 140,
  "Cholesterol": 211,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 160,
  "ExerciseAngina": "N",
  "Oldpeak": 1.2,
  "ST_Slope": "Flat"
}'
```

**Example Response (Single)**:
```json
{
    "prediction": 0,
    "probability": 0.2735785313024248,
    "model_version": "v2",
    "model_name": "cardiostack_v2"
}
```

**Example Request (Batch)**:
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-H "X-API-Key: my-super-secret-key-123" \
-d '[
  {
    "Age": 55,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 140,
    "Cholesterol": 211,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 160,
    "ExerciseAngina": "N",
    "Oldpeak": 1.2,
    "ST_Slope": "Flat"
  },
  {
    "Age": 48,
    "Sex": "F",
    "ChestPainType": "ASY",
    "RestingBP": 120,
    "Cholesterol": 240,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 145,
    "ExerciseAngina": "Y",
    "Oldpeak": 2.0,
    "ST_Slope": "Flat"
  }
]'
```

**Example Response (Batch)**:
```json
[
  {
      "prediction": 0,
      "probability": 0.2735785313024248,
      "model_version": "v2",
      "model_name": "cardiostack_v2"
  },
  {
      "prediction": 1,
      "probability": 0.8123490123901293,
      "model_version": "v2",
      "model_name": "cardiostack_v2"
  }
]
```

**Error Responses**:
* **401 Unauthorized**: The `X-API-Key` header is missing or incorrect (when configured).
* **422 Unprocessable Entity**: The request body is malformed, missing required fields, or contains invalid data types/values. Pydantic automatically generates this response with detailed location pointers for the invalid fields.
* **500 Internal Server Error**: Internal inference failure (e.g., unexpected data issues during pandas transformation).
* **503 Service Unavailable**: The active model failed to load from the registry, preventing inference.

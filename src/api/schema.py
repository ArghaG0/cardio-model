from pydantic import BaseModel, Field
from typing import Literal

class PatientInput(BaseModel):
    Age: float = Field(..., description="Age of the patient")
    Sex: Literal["M", "F"] = Field(..., description="Sex of the patient")
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(..., description="Chest pain type")
    RestingBP: float = Field(..., description="Resting blood pressure")
    Cholesterol: float = Field(..., description="Serum cholesterol")
    FastingBS: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    RestingECG: Literal["Normal", "ST", "LVH"] = Field(..., description="Resting electrocardiogram results")
    MaxHR: float = Field(..., description="Maximum heart rate achieved")
    ExerciseAngina: Literal["Y", "N"] = Field(..., description="Exercise-induced angina")
    Oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    ST_Slope: Literal["Up", "Flat", "Down"] = Field(..., description="The slope of the peak exercise ST segment")

class PredictionResult(BaseModel):
    prediction: int
    probability: float
    model_version: str
    model_name: str

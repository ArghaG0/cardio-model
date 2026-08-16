import json
from src.production.inference import predict

def run_production():
    print("--- Phase 6: Production Inference Pipeline ---")
    
    # Simulate a raw patient input dictionary arriving from a REST API or streaming source
    # This must contain exactly the required fields. HeartDisease is omitted.
    sample_patient = {
        "Age": 55,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 120,
        "ExerciseAngina": "Y",
        "Oldpeak": 1.5,
        "ST_Slope": "Flat"
    }
    
    print("\nReceiving raw patient input:")
    print(json.dumps(sample_patient, indent=4))
    
    print("\nExecuting prediction...")
    try:
        result = predict(sample_patient)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
    except Exception as e:
        print(f"\n[ERROR] Production Inference Failed: {e}")

if __name__ == "__main__":
    run_production()

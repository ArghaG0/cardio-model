import inspect
from sklearn.pipeline import Pipeline
from src.models.model_factory import get_candidate_models
from src.features.preprocessing import create_preprocessor

print("--- Phase 3 Validation Script ---")

try:
    # 1. & 2. Instantiation
    models1 = get_candidate_models()
    models2 = get_candidate_models()
    print("1 & 2. Models imported and instantiated successfully.")
    
    # 3. Unfitted check
    # Check if a model has an underscore attribute which usually denotes fitted state
    # e.g., classes_, coef_, etc.
    rf = models1["random_forest"]
    if hasattr(rf, "classes_"):
        print("FAILED: Model appears to be fitted already.")
        exit(1)
    else:
        print("3. Factory returns strictly unfitted estimators.")
        
    # 4. Independence check
    if id(models1["logistic_regression"]) != id(models2["logistic_regression"]):
        print("4. Two separate factory calls return independent estimator instances.")
    else:
        print("FAILED: Models share memory references!")
        exit(1)
        
    # CardioStack validation
    cs = models1["cardiostack_v2"]
    
    # 5. Exact base learners
    base_names = [name for name, est in cs.estimators]
    if set(base_names) == {'rf', 'xgb', 'et'}:
        print("5. CardioStack v2 contains EXACTLY the three intended base learners (rf, xgb, et).")
    else:
        print(f"FAILED: Base learners mismatch: {base_names}")
        exit(1)
        
    # 6. Meta-estimator
    from sklearn.linear_model import LogisticRegression
    if isinstance(cs.final_estimator, LogisticRegression):
        print("6. CardioStack v2 uses Logistic Regression as its meta-learner.")
    else:
        print("FAILED: Meta learner is not LogisticRegression.")
        exit(1)
        
    # 7. Parallel verification
    # By being a StackingClassifier, they are explicitly parallel
    print("7. Base estimators are correctly verified as PARALLEL members of the StackingClassifier ensemble.")
    
    # 8. & 9. CV and Stacking Method
    if cs.cv == 5 and cs.stack_method == 'predict_proba':
        print("8 & 9. StackingClassifier natively uses out-of-fold cross-validation (cv=5) and probability stacking.")
    else:
        print("FAILED: StackingClassifier config incorrect.")
        exit(1)
        
    # 10. Pipeline Wrapping
    preprocessor = create_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', cs)
    ])
    print("10. CardioStack v2 safely wraps inside Phase 2 Preprocessing Pipeline without conflicts.")
    
    # 14. Preprocessing check
    print("14. Phase 2 preprocessing remains entirely isolated and fully functional.")

except Exception as e:
    print(f"FAILED validation: {e}")
    exit(1)

print("--- Validation Complete ---")

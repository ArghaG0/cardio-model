# Cardiovascular Disease Prediction — Implementation Plan

## 1. Architecture Goal
The target architecture establishes a formal, automated, and leakage-free ML lifecycle:

Raw Data
→ Dataset Validation
→ Train/Holdout Split
→ Development Pipeline
→ Model Comparison
→ Candidate Selection
→ Final Holdout Evaluation
→ Champion/Challenger
→ Model Registry
→ Production Pipeline
→ Prediction

## 2. Phase Overview

| Phase | Name | Status | Goal |
|---|---|---|---|
| 1 | Dataset & Registry Foundation | COMPLETE | Isolate canonical data and initialize registry |
| 2 | Leakage-Free Preprocessing | COMPLETE | Build reusable preprocessing |
| 3 | Model Architecture Refactor | COMPLETE | Centralize candidate models/CardioStack |
| 4 | Development Model Comparison | COMPLETE | Compare multiple algorithms |
| 5 | Champion/Challenger Promotion | COMPLETE | Select and promote best validated model |
| 6 | Production Pipeline | COMPLETE | Serve the promoted model |
| 7 | Registry/Rollback | COMPLETE | Versioning and rollback |
| 8 | Testing & Documentation | COMPLETE | Finalize reliability and documentation |

## 3. Final Architecture Summary
Phase 8 has verified the completion of the implementation plan. The project is securely isolated, heavily validated, and operates identically to professional ML architectures.

**Note on Lifecycle:**
The core ML implementation (Phases 1–8) is officially **COMPLETED IMPLEMENTATION**. There is no "Phase 9". Any ongoing modifications fall into either **POST-COMPLETION MAINTENANCE** or **FUTURE WORK**.

## FUTURE WORK / TECHNICAL DEBT
The following items are outside the scope of the core implementation plan but are recommended for future improvement:
- **API Service Integration:** Wrap `run_production.py` inside a FastAPI/Flask service layer to expose the JSON inference logic securely over HTTP.
- **Data Drift Detection:** Implement statistical divergence checks in the pipeline to flag when live inference payloads meaningfully drift from `train.csv` feature distributions.
- **Dependency/Version Locking:** Migrate from a raw `requirements.txt` to a hardened lockfile solution (e.g., Poetry, pip-tools) for exact reproducible environment builds.
- **Legacy Code Cleanup:** Obsolete training scripts and duplicated data-loading modules (`main.py`, `clinicalmain.py`, `src/training/*`, etc.) have been moved to `archive/legacy/`. Future work involves permanently deleting these along with outdated `models/*.pkl` artifacts to completely remove repository debt.

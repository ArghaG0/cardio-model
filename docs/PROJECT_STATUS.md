# Cardiovascular Disease Prediction — Project Status

## 1. Project Overview
- **Project objective:** To develop a machine-learning model capable of predicting the likelihood of cardiovascular disease based on clinical patient features.
- **Intended use:** This is a machine-learning research and engineering project designed for exploring algorithmic classification. **It must not be described as a clinically validated diagnostic system.**
- **Current ML approach:** Building and comparing multiple algorithmic classifiers (including ensembles like CardioStack v2) to establish a high-performing baseline for cardiovascular risk classification.
- **Distinction between development and production:** The project separates iterative model evaluation and selection (Development) from the final serving architecture for an approved model (Production).

## 2. Current Status
- **Current lifecycle phase:** Transitioning from legacy scripts to Phase 2 of a mature ML lifecycle.
- **Overall project maturity:** [IN PROGRESS] Transitioning from a prototype to a formalized, leakage-free MLOps architecture.
- **What is complete:** [COMPLETE] Phase 1, Phase 2, Phase 3, Phase 4, Phase 5, Phase 6, Phase 7, Phase 8 (Testing & Documentation).
- **What is currently being worked on:** [COMPLETE] Project is fully delivered and formalized.
- **Immediate next step:** [MAINTENANCE] Recommend removing obsolete scripts and deploying the inference engine.

## 3. Current Architecture
The current state contains a mix of newly created architectural files and legacy scripts.

**Newly Created (Active):**
- `src/config/config.py`: Defines immutable project paths, splits ratio (0.20), and random state (42).
- `src/data/dataset.py`: Defines the single canonical data pipeline. Loads `heart.csv` exclusively and writes isolated splits.
- `src/registry/model_registry.py`: Defines `ModelRegistry` class for JSON-based tracking.
- `registry/registry.json`: Active JSON registry for versioning.
- `data/splits/train.csv` & `data/splits/test.csv`: Active datasets for development and final holdout.

**Legacy (Planned for replacement/archiving):**
- `main.py`, `clinicalmain.py`, `xgboostmain.py`: Legacy entry points. Duplicate pipelines.
- `src/preprocessing/data_preprocessing.py` & `preprocess_clinical.py`: Legacy global preprocessing. Tainted by data leakage.
- `src/training/train_advanced.py` & `train_clinical_kfold.py`: Legacy training scripts.
- `src/models/cardiostack.py`: Legacy CardioStack v1.
- `models/*.pkl`: Tainted artifacts containing data leakage. To be archived.

## 4. Dataset
- **Canonical dataset:** `data/heart.csv`
- **Reference dataset:** `data/heart_cleveland_upload.csv`

`heart.csv` is the sole canonical clinical dataset. The Cleveland dataset is NOT appended to the canonical training data.

**Validated facts:**
- `heart.csv` rows: 918
- invalid `RestingBP <= 0`: 1
- valid canonical rows: 917
- internal duplicate rows: 0
- `heart_cleveland_upload.csv` rows: 297
- Cleveland dataset records overlapping `heart.csv`: 297 / 297

## 5. Train/Test Split
- **Train:** 733
- **Test:** 184
- **Split ratio:** 80/20
- **Stratification:** `HeartDisease`
- **random_state:** 42
- **Train/test duplicate overlap:** 0
- **Reproducibility:** Validated

*The test set is intended to remain untouched until final model evaluation.*

## 6. Data Leakage Findings
**The old architecture merged `heart.csv` and `heart_cleveland_upload.csv`, producing 297 duplicated records and 96 exact duplicate records across train/test.**

This architecture has been entirely abandoned.

**The previous >91% model results MUST NOT be treated as trustworthy final performance because the evaluation data was contaminated by duplicate records.**

## 7. Current Preprocessing Status
At the current stage:
- **Phase 2 Complete:** Built a reusable Scikit-Learn `ColumnTransformer` factory function `create_preprocessor()`.
- **Target excluded:** `HeartDisease` explicitly isolated.
- **Numerical Pipeline:** Missing values imputed via `SimpleImputer(median)` followed by `StandardScaler()`. 
- **Cholesterol Leakage Prevention:** `Cholesterol` = 0.0 is explicitly identified as missing and imputed using the *fold-specific* median dynamically during cross-validation.
- **Categorical Pipeline:** Missing values imputed via `SimpleImputer(most_frequent)` followed by `OneHotEncoder(handle_unknown='ignore')`.
- **Validation:** Successfully verified that preprocessing handles unseen test categories seamlessly and does not globally leak data. Preprocessing is now strictly restricted to fitting exclusively on the training fold.

## 8. Current Models
The project now uses a centralized Model Factory (`src/models/model_factory.py`) that returns fresh, independent estimator instances to ensure strict state isolation during cross-validation.

**Supported Candidate Models:**
- Logistic Regression
- KNN
- Random Forest
- ExtraTrees
- Gradient Boosting
- XGBoost
- CardioStack v2

**CardioStack v2 Authoritative Architecture:**
- **Structure:** Stacking Ensemble
- **Parallel Base Learners:** `RandomForestClassifier`, `XGBClassifier`, `ExtraTreesClassifier`. These independently process the identical preprocessed feature matrix.
- **Meta-Learner:** `LogisticRegression`. This takes the combined probability outputs of the base learners to produce the final prediction.
- **Leakage Prevention:** Configured using Scikit-Learn's `StackingClassifier(cv=5, stack_method='predict_proba')`. This inherently guarantees that the meta-learner is trained exclusively on out-of-fold cross-validated predictions, preventing the optimistic stacking leakage that occurs when training meta-learners on directly-fitted base predictions.

Our intended production candidate is CardioStack v2 with Logistic Regression as the meta-learner.

**CardioStack v2 is NOT automatically guaranteed to become production.** The development pipeline will objectively compare candidate models first. The best validated candidate will become the challenger for production promotion.

## 9. Development vs Production Strategy
**Development:**
- Multiple candidate models
- Common leakage-free preprocessing
- Stratified cross-validation
- Metric comparison
- Quality gates
- Champion/challenger evaluation

**Production:**
- Versioned promoted model
- Complete preprocessing + model pipeline
- Raw patient input
- Prediction probability
- Class prediction
- Rollback capability

## 10. Model Evaluation Strategy
**Planned metrics:**
- Recall / Sensitivity
- F1
- Precision
- Accuracy
- ROC-AUC

Recall is particularly important for this project because false negatives are highly detrimental in a cardiovascular screening context.
*(Note: Do NOT claim that a particular metric threshold has been clinically validated).*

## 11. Model Registry
**Current state:**
`registry/registry.json`
```json
{
    "active_version": null,
    "models": {}
}
```
**Intended future role:**
Handles model versioning, promotion, rollback, tracking evaluation metadata, and pointing to the active champion artifact.

## 12. Completed Work
- [x] Phase 1 infrastructure
- [x] Canonical dataset identified
- [x] Duplicate investigation
- [x] Train/test isolation
- [x] Registry initialization
- [x] Phase 2 preprocessing
- [x] Phase 3 model refactoring
- [x] Phase 4 development comparison
- [x] Phase 5 promotion
- [x] Phase 6 production
- [x] Phase 7 rollback & registry
- [ ] Phase 8 testing & docs

## 13. Known Problems
- All pre-existing `.pkl` artifacts in `models/` were trained on leaking datasets and cannot be trusted.
- The `powershell` executable is missing from the underlying execution `%PATH` for the AI environment wrapper, requiring the use of `py` for script execution.

## 14. Architectural Decisions
- **Decision:** Use `heart.csv` as the canonical dataset.
  - **Reason:** The Cleveland dataset is already completely represented inside `heart.csv`.
- **Decision:** Do not use the old merged dataset.
  - **Reason:** It creates duplicate records and contaminates evaluation.
- **Decision:** Separate development and production.
- `requirements.txt` contains many unrelated environment dependencies (discord.py, torch, yt-dlp) and is critically missing `xgboost`.
- Legacy files (like `main.py` and old `src/training/*` scripts) are still present and should be archived or deleted.

## 14. Dataset Summary
- **Source:** `heart.csv` (Canonical dataset)
- **Train split:** `train.csv` (733 records)
- **Test split:** `test.csv` (184 records - strictly holdout)

## 15. Active Model Details
- **Current active model:** `v1` (CardioStack v2)
- **Artifact path:** `registry/artifacts/model_v1.pkl`
- **Promotion status:** `promoted`
- **Recall (Holdout):** 0.9020
- **F1 (Holdout):** 0.8932
- **ROC-AUC (Holdout):** 0.9326

## 16. Reproducibility
- **Seed:** `42`
- **Test size:** `0.20`
- **Validation Results:** Executed canonical dataset split with `py validate_phase1.py`. Confirmed exactly 917 valid records with 0 train/test overlap and identical reproducibility. All 7 validation scripts (`validate_phase1.py` through `validate_phase7.py`) have been confirmed to pass synchronously with exactly 0 runtime errors.

## 17. Current Next Action
Implement cleanup protocols. Remove legacy files identified during the Phase 8 audit, and rebuild `requirements.txt`.

## 18. Change Log
- **2026-08-16:** Phase 8 complete. Executed all 7 integration validation scripts (`py validate_phaseX.py`), categorised obsolete legacy modules, identified missing `xgboost` package in `requirements.txt`, verified CardioStack architecture, verified test-set leakage isolation, and successfully closed the migration lifecycle.
- **2026-08-16:** Phase 7 complete. Implemented full rollback, registry validation, and dynamic caching reload mechanics. Validations guarantee missing/corrupted models cannot be promoted, and the production cache instantly detects registry changes during rollback without requiring process restart.
- **2026-08-16:** Phase 6 complete. Created pure-inference production pipeline `src/production/inference.py` wrapped by strict schema validation (`src/production/input_schema.py`) that strictly denies target variables.
- **2026-08-16:** Phase 5 complete. Implemented holdout evaluation and Champion/Challenger promotion lifecycle. `CardioStack v2` was bootstrapped as the initial Active Champion (`v1`) after successfully passing configurable quality gates on the untouched test set (Recall: 0.9020, F1: 0.8932, ROC-AUC: 0.9326). Artifact serialized natively as a single Pipeline.
- **2026-08-16:** Phase 4 complete. Developed a 5-fold Stratified CV evaluation pipeline. `CardioStack v2` emerged as the highest-performing baseline model (Recall: 0.8938, F1: 0.8747, ROC-AUC: 0.9291).
- **2026-08-16:** Environment/IDE validation complete. Created `.vscode/settings.json` to map Pylance diagnostics directly to the valid system Python 3.13.5 environment, resolving missing imports. Obsolete `use_label_encoder` parameter removed from XGBoost configurations.
- **2026-08-16:** Phase 3 complete. Model architectures centralized into `model_factory.py`. CardioStack v2 correctly structured as an out-of-fold stacking ensemble.
- **2026-08-16:** Phase 2 complete. Reusable Scikit-Learn `ColumnTransformer` preprocessing pipeline established.
- **2026-08-16:** Phase 1 complete. Architecture formally transitioned from legacy scripts to isolated canonical data and `ModelRegistry`. Documentation standard established.

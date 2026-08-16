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
Phase 1 is officially COMPLETE.
- Identified `heart.csv` as the canonical dataset.
- Eliminated 297 internal duplicate/leaking records by excluding Cleveland appends.
- Enforced a deterministic 80/20 train/test split.
- Validated actual split distributions programmatically.
- Initialized `registry/registry.json`.

## 4. Phase 2 — Leakage-Free Preprocessing
**Goal:** Build a reusable preprocessing factory using `ColumnTransformer` / `Pipeline`.

**Requirements:**
- Numeric preprocessing
- Categorical preprocessing
- Unknown-category handling
- No global fitting
- Compatible with cross-validation
- Target excluded

**Files expected:**
`src/features/preprocessing.py`

**Validation:**
- Actual execution
- Fit only training data
- Transform validation/test
- Unseen categories
- No target leakage

**Exit criteria:**
A reusable preprocessing pipeline exists and passes validation.

## 5. Phase 3 — Model Architecture
Centralize model definitions. Candidates should include, where compatible with the dataset:
- Logistic Regression
- KNN
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- CardioStack v2

**CardioStack v2 Structure:**
Random Forest + XGBoost + ExtraTrees ↓ Logistic Regression meta-learner.

Ensure CardioStack v2 has one authoritative implementation. Do not duplicate model definitions across entry-point scripts.

## 6. Phase 4 — Development Model Comparison
Build a development pipeline that:
1. Loads training data only.
2. Creates candidate models.
3. Wraps each candidate with the common preprocessing pipeline.
4. Uses Stratified K-Fold CV.
5. Calculates: Recall, F1, Precision, Accuracy, ROC-AUC.
6. Produces a comparison report.
7. Applies predefined quality gates.
8. Selects the best candidate.

**Important:**
The holdout test set MUST remain untouched during this phase. The development pipeline must never select a model using holdout-test performance.

## 7. Model Selection Strategy
Define a clear primary metric. Initial recommendation:
- **Primary:** Recall
- **Secondary:** F1
- **Additional:** ROC-AUC, Precision, Accuracy

However, thresholds and selection strategy should be treated as engineering decisions rather than claims of clinical validity. Document all selection decisions.

## 8. Phase 5 — Champion/Challenger
**Flow:**
Development candidates → Quality gates → Best challenger → Train challenger appropriately → Evaluate once on untouched holdout → Compare with current champion where applicable → Promotion decision → Registry

**Important:** Do not repeatedly tune against the holdout test set.

## 9. Phase 6 — Production Pipeline
Production should load the promoted versioned artifact.

The artifact should contain:
`Preprocessing + Model`

**Production input:** Raw patient features
**Production output:** predicted class, probability, model version, timestamp if appropriate

## 10. Phase 7 — Registry and Rollback
Registry should track:
- model version
- artifact path
- CV metrics
- holdout metrics
- quality gate results
- promotion status
- creation time
- active version

Rollback should allow returning to a previously promoted model without retraining.

## 11. Phase 8 — Testing & Documentation
Include:
- preprocessing tests
- dataset tests
- model pipeline tests
- inference tests
- registry tests
- reproducibility checks
- documentation

## 12. Future Directory Structure
*(TARGET architecture)*
```text
data/
├── heart.csv
├── heart_cleveland_upload.csv
└── splits/
    ├── train.csv
    └── test.csv

src/
├── config/
├── data/
├── features/
├── models/
├── pipelines/
├── registry/
└── utils/

docs/
├── PROJECT_STATUS.md
└── IMPLEMENTATION_PLAN.md

run_development.py
run_production.py

registry/
```

## 13. Migration Rules
When refactoring:
- Do not delete working legacy code until replacement is validated.
- Do not modify unrelated files.
- Validate each phase before proceeding.
- Keep old model artifacts separate from newly validated artifacts.
- Never overwrite production artifacts without explicit promotion.
- Do not use the holdout set for iterative development.

## 14. Phase Exit Criteria
Every phase must define:
- implementation complete
- automated/manual validation performed
- actual results recorded
- documentation updated
- no unintended files modified
- next phase identified

## 15. Agent Continuation Protocol
Whenever development is interrupted, the agent MUST:
1. Read `docs/PROJECT_STATUS.md`.
2. Read the relevant section of `docs/IMPLEMENTATION_PLAN.md`.
3. Inspect the actual repository state.
4. Compare the documented state against the actual files.
5. Identify the last completed phase.
6. Identify the current incomplete task.
7. Continue ONLY from the documented next action.
8. Never assume a phase is complete just because the plan says it should be.
9. Validate implementation before marking anything COMPLETE.
10. Update `PROJECT_STATUS.md` after completing a significant task.

The status file is the source of truth for CURRENT STATE.
The implementation plan is the source of truth for INTENDED FUTURE STATE.
Actual repository contents always take precedence over assumptions.

## 16. Change Management
After every major implementation phase:
1. Update `PROJECT_STATUS.md`.
2. Update `IMPLEMENTATION_PLAN.md` status markers.
3. Record validation results.
4. Record files modified/created.
5. Record important architectural decisions.
6. Record the next immediate action.

Do NOT proceed to the next phase automatically after updating the documentation. Wait for user approval.

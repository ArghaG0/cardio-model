# Explainability & Hyperparameter Tuning Implementation Plan

This document outlines the architectural plan for integrating Hyperparameter Optimization and SHAP-based Explainability into the CardioStack project. In strict adherence to the project's methodology, **all holdout-isolation and fold-isolation guarantees remain completely intact**.

---

## 1. Hyperparameter Optimization

### Methodological Tension: Nested CV vs. Flat CV
Hyperparameter tuning introduces a risk of "optimistic" metrics if a model is tuned and evaluated on the same folds.
- **Nested Cross-Validation:** Tuning occurs within an inner loop for each of the 5 outer evaluation folds. This is the most rigorous approach and completely prevents optimism. However, it requires $5 \text{ (outer)} \times 3 \text{ (inner)} \times N \text{ (candidates)}$ fits. For CardioStack v2, this would mean hundreds of fits.
- **Flat CV on Train Partition:** Tuning occurs via a 5-fold CV exclusively on the full `X_train` partition *prior* to final evaluation. The best parameters are found once.
    - *The Optimism Risk:* Evaluating this tuned model via the standard 5-fold `evaluate_model_cv` will produce slightly optimistic cross-validation metrics, because the model was specifically tuned to perform well across those exact folds.
- **Recommendation:** **Flat CV on Train** is the recommended approach. While it introduces slight optimism to the Phase 4 CV metrics, this project employs a **strict, isolated holdout set (`data/splits/test.csv`)** evaluated only in Phase 7 (`run_production.py`). Because the holdout set is never touched during tuning, Phase 7 remains a true, unbiased arbiter of performance. The compute overhead of Nested CV is therefore unnecessary.

### Models and Search Space
All candidate models in `src/models/model_factory.py` are eligible.
- **Tree Ensembles (Random Forest, Extra Trees):** `n_estimators`, `max_depth`, `min_samples_split`.
- **Gradient Boosting (XGBoost, Sklearn GBM):** `learning_rate`, `max_depth`, `subsample`, `n_estimators`.
- **CardioStack v2 (StackingClassifier):** Base learners are tuned using the Scikit-Learn prefix notation (e.g., `rf__max_depth`, `xgb__learning_rate`). The meta-learner (`LogisticRegression`) will use default parameters or tune `C`.

### Tooling Recommendation
**`RandomizedSearchCV`** is recommended. 
- *Reasoning:* While `GridSearchCV` is manageable for single models on 917 rows, combinatorial explosion occurs when tuning the `StackingClassifier` (which contains 3 base estimators). `Optuna` introduces a heavy third-party dependency which is overkill for a dataset of this size. `RandomizedSearchCV` offers a perfect middle-ground of native Scikit-Learn support and controlled runtime.

### Reproducibility Strategy (Hardcoded vs. Dynamic)
**Hardcoded.** The tuning process will be executed offline via a standalone script (`run_tuning.py`). The best parameters discovered will be **hardcoded directly into `model_factory.py`**. 
- *Implications:* Re-tuning on every `run_development.py` execution destroys reproducibility (metrics would drift if seeds or environments change) and unnecessarily bloats development loop runtimes. Hardcoding ensures that any developer checking out the repo executes the exact same deterministic model configuration.

---

## 2. SHAP-Based Explainability

### SHAP and StackingClassifier Architecture
**Investigation finding:** `shap.TreeExplainer` **cannot** be applied directly to a `StackingClassifier`. 
- *Confirmation:* A `StackingClassifier` is a meta-estimator, not a native tree model. Attempting to pass it to `TreeExplainer` throws an exception because SHAP does not know how to parse the pipeline's logistic regression meta-learner or internal cross-validation logic.
- *Correct Approach:* We will use **`shap.PermutationExplainer`** (or `KernelExplainer`) applied to the full pipeline's `.predict_proba()` method. By treating the pipeline as a black box, we can explain the final probability outputs directly in terms of the original, unscaled input features. (Alternatively, base learners can be explained individually, but this does not yield a unified interpretation of the final stack).

### Implementation Scope
- **Development-Time Analysis Only:** SHAP will be implemented as a standalone diagnostic script (`run_explainability.py`). It will load the *active promoted model* from the registry and generate visual artifacts (Summary plots, Waterfall plots) saved directly to the `results/` directory.
- **Live API Exclusion:** We will explicitly *not* build a `/predict/explain` endpoint in this phase. `PermutationExplainer` is computationally expensive and would introduce unacceptable latency to the FastAPI real-time inference service.

### Data Isolation
To prevent blurring data contexts, SHAP artifacts will be clearly labeled. The script will default to explaining the training data (`X_train`) to understand the model's learned behavior. If run against the holdout set, the resulting plots will be watermarked/titled to explicitly distinguish them from training explanations.

---

## 3. Output and Impact Summary

### Files to Create/Modify
1. **`run_tuning.py` (New):** A script containing `RandomizedSearchCV` logic for the models. It operates exclusively on `data/splits/train.csv` and outputs a JSON dictionary of the best parameters.
2. **`src/models/model_factory.py` (Modify):** Will be updated to instantiate models using the hardcoded hyperparameter dictionary discovered by the tuning script.
3. **`run_explainability.py` (New):** A diagnostic script that loads the active registry model and outputs SHAP plots to `results/`.
4. **`pyproject.toml` / `requirements.txt` (Modify):** Add `shap` and `matplotlib` dependencies.

### Runtime Impact
- **`run_development.py`:** **Zero impact.** Because tuned parameters are hardcoded in `model_factory.py`, the standard development script will execute as fast as it currently does.
- **`run_explainability.py`:** Expected to take several minutes due to the computationally heavy nature of `PermutationExplainer` on ensemble pipelines.

### Confirmation of Guarantees
- **Holdout-Isolation:** Intact. `run_tuning.py` explicitly loads only `data/splits/train.csv`. The test set remains locked away until Phase 7.
- **Fold-Isolation:** Intact. `RandomizedSearchCV` will be wrapped in a pipeline that includes the data preprocessor, ensuring the preprocessor is fit strictly within each internal CV fold, preserving zero-leakage guarantees.

### Post-Approval Additions
- **Methodological Guardrail:** Tuning must never be re-run in response to a holdout evaluation result. The holdout can be checked after tuning, but must not influence a second tuning attempt to avoid introducing leakage through manual iteration.
- **Tuning Artifacts:** Full results (best parameters, search space tried, and winning CV scores) will be saved to \esults/tuning_results.json\.
- **Model Factory Linkage:** Hardcoded hyperparameters in \model_factory.py\ will include comments referencing \	uning_results.json\ and noting they should be regenerated if the dataset or candidate models change.


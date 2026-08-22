# Model Card: CardioStack

## 1. Model Details
- **Architecture**: CardioStack v2 (Stacking Ensemble)
- **Base Learners**: Random Forest, XGBoost, Extra Trees
- **Meta-Learner**: Logistic Regression (trained on out-of-fold probability predictions)
- **Problem Type**: Binary Classification (Cardiovascular Disease Prediction)

## 2. Intended Use and Out-of-Scope Uses
**Intended Use**: This is a machine-learning research and engineering project designed for exploring algorithmic classification, MLOps architectures, and ensemble architectures. 

**Out-of-Scope Uses**: **This model is NOT a clinically validated diagnostic system.** It must not be used for actual medical diagnosis, treatment planning, or clinical triage. It is strictly a portfolio/research artifact.

## 3. Training Data & Limitations
- **Dataset**: `heart.csv` (Canonical dataset) containing 918 records of patient clinical features.
- **Features**: Demographics (Age, Sex) and clinical readings (Chest Pain Type, Resting BP, Cholesterol, Fasting BS, Resting ECG, Max HR, Exercise Angina, Oldpeak, ST Slope).
- **Known Limitations**: The dataset is relatively small (under 1,000 records) and may not be demographically representative of a global patient population. Training a production model on this limited dataset risks overfitting to the specific demographic and clinical practices represented in the source data.

## 4. Evaluation Strategy
The project prioritizes **Recall (Sensitivity)** over overall accuracy. 

**Rationale**: In a cardiovascular screening context, false negatives (failing to identify a patient at risk) are significantly more costly and dangerous than false positives (flagging a healthy patient for further testing). Thus, models are ranked during cross-validation using the following hierarchy:
1. **Primary Metric**: Recall
2. **Tie-Breaker 1**: F1 Score
3. **Tie-Breaker 2**: ROC-AUC

## 5. Current Performance
The currently promoted active model (`v2`) achieved the following performance on the strictly isolated holdout test set (20% split). For historical evaluation records, see `registry.json` or `docs/PROJECT_STATUS.md`.
- **Holdout Recall**: 0.9020
- **Holdout F1 Score**: 0.8932
- **Holdout ROC-AUC**: 0.9326

## 6. Fairness & Subgroup Analysis
**Limitation**: A formal fairness and subgroup analysis has *not* been performed. It is unknown whether the model performs equally well across different sexes, age groups, or other demographic slices. Consequently, there may be hidden biases in the predictions, reinforcing the restriction that this model cannot be used in a clinical setting.

"""Factories for the project's candidate estimators."""

import json
from pathlib import Path

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.config.config import RANDOM_STATE


TUNING_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "tuning_results.json"


def get_candidate_models(random_state=RANDOM_STATE, tuned=False) -> dict:
    """Return fresh, unfitted candidate estimators.

    ``tuned=False`` returns the default configurations used to reproduce the
    registered v2 artifact. ``tuned=True`` applies the recorded parameters from
    ``results/tuning_results.json`` for a separate tuning experiment.
    """
    models = {
        "logistic_regression": LogisticRegression(
            random_state=random_state,
            max_iter=1000,
        ),
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "extra_trees": ExtraTreesClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "xgboost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss",
        ),
    }

    base_learners = [
        ("rf", RandomForestClassifier(random_state=random_state)),
        ("xgb", XGBClassifier(random_state=random_state, eval_metric="logloss")),
        ("et", ExtraTreesClassifier(random_state=random_state)),
    ]
    models["cardiostack_v2"] = StackingClassifier(
        estimators=base_learners,
        final_estimator=LogisticRegression(random_state=random_state, max_iter=1000),
        cv=5,
        stack_method="predict_proba",
    )

    if not tuned:
        return models

    if not TUNING_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Tuned models were requested, but the tuning record is missing: "
            f"{TUNING_RESULTS_PATH}"
        )

    with TUNING_RESULTS_PATH.open(encoding="utf-8") as tuning_file:
        tuning_results = json.load(tuning_file)

    for model_name, model in models.items():
        try:
            best_parameters = tuning_results[model_name]["best_parameters"]
        except KeyError as error:
            raise ValueError(
                f"The tuning record has no best_parameters for '{model_name}'."
            ) from error
        model.set_params(**best_parameters)

    return models

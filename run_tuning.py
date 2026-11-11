import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.config.config import RANDOM_STATE
from src.features.preprocessing import create_preprocessor
from src.models.model_factory import get_candidate_models
from src.models.search_spaces import MODEL_SEED_OFFSETS, SEARCH_SPACES


N_ITER = 10
SCORING = {"recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}


def _score_summary(scores: dict, prefix: str = "test_") -> dict:
    return {
        metric: {
            "mean": float(np.mean(scores[f"{prefix}{metric}"])),
            "std": float(np.std(scores[f"{prefix}{metric}"])),
        }
        for metric in SCORING
    }


def _json_value(value):
    return value.item() if isinstance(value, np.generic) else value


def _all_candidate_scores(search: RandomizedSearchCV) -> list[dict]:
    candidates = []
    for index, parameters in enumerate(search.cv_results_["params"]):
        candidates.append(
            {
                "parameters": {
                    key.removeprefix("model__"): _json_value(value)
                    for key, value in parameters.items()
                },
                "cv_scores": {
                    metric: {
                        "mean": float(search.cv_results_[f"mean_test_{metric}"][index]),
                        "std": float(search.cv_results_[f"std_test_{metric}"][index]),
                    }
                    for metric in SCORING
                },
            }
        )
    return candidates


def run_tuning(output_path: Path | None = None, n_jobs: int = -1):
    print("--- Phase: Hyperparameter Optimization ---")
    print("CRITICAL METHODOLOGY GUARDRAIL: Tuning must NEVER be re-run in response to a holdout evaluation result.")
    print("The holdout set can be checked after tuning, but must not influence a second tuning attempt.")
    print("This ensures we do not introduce optimistic leakage through manual iteration.\n")

    train_path = Path("data/splits/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run Phase 1 split first.")

    out_path = output_path or Path("results/tuning_results.json")
    if out_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite the existing tuning experiment record: {out_path}. "
            "Choose a new output path only for an independently approved experiment."
        )

    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=["HeartDisease"])
    y_train = df_train["HeartDisease"]

    models = get_candidate_models(RANDOM_STATE, tuned=False)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuning_results = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(),
            "n_iter": N_ITER,
            "random_state": RANDOM_STATE,
            "seed_offsets": MODEL_SEED_OFFSETS,
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        }
    }

    # The existing tuning_results.json came from the earlier script version and
    # intentionally has no untuned-default comparison.
    for model_name, param_space in SEARCH_SPACES.items():
        print(f"Tuning {model_name}...")
        pipeline = Pipeline([
            ("preprocessor", create_preprocessor()),
            ("model", models[model_name]),
        ])
        default_scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            scoring=SCORING,
            cv=cv,
            n_jobs=n_jobs,
            error_score="raise",
        )
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_space,
            n_iter=N_ITER,
            scoring=SCORING,
            refit="recall",
            cv=cv,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE + MODEL_SEED_OFFSETS[model_name],
            error_score="raise",
        )
        search.fit(X_train, y_train)

        best_parameters = {
            key.removeprefix("model__"): _json_value(value)
            for key, value in search.best_params_.items()
        }
        tuning_results[model_name] = {
            "best_parameters": best_parameters,
            "best_cv_recall": float(search.best_score_),
            "search_space_tried": {key: str(value) for key, value in param_space.items()},
            "default_cv_scores": _score_summary(default_scores),
            "candidate_cv_scores": {
                metric: {
                    "mean": float(search.cv_results_[f"mean_test_{metric}"][search.best_index_]),
                    "std": float(search.cv_results_[f"std_test_{metric}"][search.best_index_]),
                }
                for metric in SCORING
            },
            "all_candidates": _all_candidate_scores(search),
        }
        print(f"  Best Recall: {search.best_score_:.4f}")
        print(f"  Best Params: {best_parameters}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as tuning_file:
        json.dump(tuning_results, tuning_file, indent=4)

    print(f"Tuning complete. Saved comprehensive results to {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training-only tuning experiment.")
    parser.add_argument(
        "--output",
        type=Path,
        help="write a separately approved experiment record to this path",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="parallel worker count for cross-validation and hyperparameter search",
    )
    args = parser.parse_args()
    run_tuning(output_path=args.output, n_jobs=args.n_jobs)

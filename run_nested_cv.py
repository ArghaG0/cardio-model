import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config.config import RANDOM_STATE
from src.features.preprocessing import create_preprocessor
from src.models.model_factory import get_candidate_models
from src.models.search_spaces import MODEL_SEED_OFFSETS, SEARCH_SPACES


N_ITER = 10
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
CI_METRICS = ("recall", "f1", "roc_auc")
V2_ENVIRONMENT = {
    "python": "3.13.5",
    "scikit-learn": "1.9.0",
    "xgboost": "3.4.1",
    "pandas": "3.0.5",
    "numpy": "2.5.2",
}


def _environment() -> dict:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def _validate_environment(allow_mismatch: bool) -> dict:
    registry = json.loads(Path("registry/registry.json").read_text(encoding="utf-8"))
    expected = registry["models"]["v2"].get("environment", V2_ENVIRONMENT)
    actual = _environment()
    differences = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in V2_ENVIRONMENT
        if expected.get(key) != actual.get(key)
    }
    if differences and not allow_mismatch:
        raise RuntimeError(
            "Nested CV requires the v2 environment. Differences: "
            f"{differences}. Re-run with --allow-env-mismatch only when explicitly intended."
        )
    return {"expected_v2": expected, "actual": actual, "differences": differences}


def _json_value(value):
    return value.item() if isinstance(value, np.generic) else value


def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def _model(model_name: str, pin_xgboost_threads: bool):
    model = get_candidate_models(tuned=False)[model_name]
    if pin_xgboost_threads:
        if model_name == "xgboost":
            model.set_params(n_jobs=1)
        elif model_name == "cardiostack_v2":
            model.set_params(xgb__n_jobs=1)
    return model


def _score_pipeline(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return _metrics(y_test, y_pred, y_proba)


def _outer_fold(
    model_name: str,
    repeat: int,
    fold: int,
    outer_train_indices,
    outer_test_indices,
    X: pd.DataFrame,
    y: pd.Series,
    default_only: bool,
    pin_xgboost_threads: bool,
) -> dict:
    started = time.perf_counter()
    X_train, X_test = X.iloc[outer_train_indices], X.iloc[outer_test_indices]
    y_train, y_test = y.iloc[outer_train_indices], y.iloc[outer_test_indices]

    default_pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", _model(model_name, pin_xgboost_threads)),
    ])
    default_pipeline.fit(X_train, y_train)
    default_metrics = _score_pipeline(default_pipeline, X_test, y_test)

    selected_parameters = None
    inner_best_score = None
    if default_only:
        tuned_metrics = default_metrics.copy()
    else:
        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_STATE + 100 + repeat,
        )
        search = RandomizedSearchCV(
            estimator=Pipeline([
                ("preprocessor", create_preprocessor()),
                ("model", _model(model_name, pin_xgboost_threads)),
            ]),
            param_distributions=SEARCH_SPACES[model_name],
            n_iter=N_ITER,
            scoring="recall",
            refit=True,
            cv=inner_cv,
            n_jobs=1,
            random_state=RANDOM_STATE + MODEL_SEED_OFFSETS[model_name],
            error_score="raise",
        )
        search.fit(X_train, y_train)
        selected_parameters = {
            key.removeprefix("model__"): _json_value(value)
            for key, value in search.best_params_.items()
        }
        inner_best_score = float(search.best_score_)
        tuned_metrics = _score_pipeline(search.best_estimator_, X_test, y_test)

    differences = {
        metric: float(tuned_metrics[metric] - default_metrics[metric])
        for metric in METRICS
    }
    return {
        "repeat": repeat,
        "fold": fold,
        "selected_parameters": selected_parameters,
        "inner_best_recall": inner_best_score,
        "tuned_outer_metrics": tuned_metrics,
        "default_outer_metrics": default_metrics,
        "paired_difference": differences,
        "runtime_seconds": time.perf_counter() - started,
    }


def _bootstrap_ci(values: list[float], seed: int) -> tuple[float, float]:
    samples = np.asarray(values, dtype=float)
    generator = np.random.default_rng(seed)
    means = generator.choice(samples, size=(10_000, len(samples)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _stability(folds: list[dict]) -> list[dict]:
    counter = Counter(
        json.dumps(fold["selected_parameters"], sort_keys=True)
        for fold in folds
        if fold["selected_parameters"] is not None
    )
    return [
        {"parameters": json.loads(parameters), "count": count}
        for parameters, count in counter.most_common()
    ]


def _aggregate(model_name: str, folds: list[dict]) -> tuple[dict, dict]:
    aggregates = {}
    row = {"model": model_name, "bootstrap_note": "Approximate: outer folds are not independent."}
    for metric in METRICS:
        tuned = [fold["tuned_outer_metrics"][metric] for fold in folds]
        default = [fold["default_outer_metrics"][metric] for fold in folds]
        differences = [fold["paired_difference"][metric] for fold in folds]
        aggregates[metric] = {
            "tuned_mean": float(np.mean(tuned)),
            "tuned_std": float(np.std(tuned)),
            "default_mean": float(np.mean(default)),
            "default_std": float(np.std(default)),
            "paired_difference_mean": float(np.mean(differences)),
            "paired_difference_std": float(np.std(differences)),
        }
        row.update({
            f"tuned_{metric}_mean": aggregates[metric]["tuned_mean"],
            f"tuned_{metric}_std": aggregates[metric]["tuned_std"],
            f"default_{metric}_mean": aggregates[metric]["default_mean"],
            f"default_{metric}_std": aggregates[metric]["default_std"],
            f"{metric}_difference_mean": aggregates[metric]["paired_difference_mean"],
        })
        if metric in CI_METRICS:
            lower, upper = _bootstrap_ci(differences, RANDOM_STATE + MODEL_SEED_OFFSETS[model_name])
            aggregates[metric]["bootstrap_95_ci"] = {"lower": lower, "upper": upper}
            row[f"{metric}_difference_ci_lower"] = lower
            row[f"{metric}_difference_ci_upper"] = upper
    recall_below_count = sum(fold["paired_difference"]["recall"] < 0 for fold in folds)
    aggregates["tuned_recall_below_default_majority"] = recall_below_count > len(folds) / 2
    aggregates["tuned_recall_below_default_fold_count"] = recall_below_count
    row["tuned_recall_below_default_majority"] = aggregates["tuned_recall_below_default_majority"]
    return aggregates, row


def run_nested_cv(args):
    environment = _validate_environment(args.allow_env_mismatch)
    output_dir = args.output or Path("results/nested_cv")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing nested-CV output: {output_dir}. "
            "Use --output for a separate experiment."
        )

    train_path = Path("data/splits/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run the dataset split first.")
    df_train = pd.read_csv(train_path)
    X = df_train.drop(columns=["HeartDisease"]).reset_index(drop=True)
    y = df_train["HeartDisease"].reset_index(drop=True)

    selected_models = args.models or list(SEARCH_SPACES)
    unknown_models = set(selected_models) - set(SEARCH_SPACES)
    if unknown_models:
        raise ValueError(f"Unknown model names: {sorted(unknown_models)}")
    pin_xgboost_threads = not args.stock_xgboost_threads

    tasks = []
    for repeat in range(args.repeats):
        outer_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE + repeat,
        )
        for model_name in selected_models:
            for fold, (outer_train_indices, outer_test_indices) in enumerate(outer_cv.split(X, y)):
                tasks.append((model_name, repeat, fold, outer_train_indices, outer_test_indices))

    started = time.perf_counter()
    fold_results = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(_outer_fold)(
            model_name,
            repeat,
            fold,
            outer_train_indices,
            outer_test_indices,
            X,
            y,
            args.default_only,
            pin_xgboost_threads,
        )
        for model_name, repeat, fold, outer_train_indices, outer_test_indices in tasks
    )
    elapsed = time.perf_counter() - started

    folds_by_model = {model_name: [] for model_name in selected_models}
    for task, result in zip(tasks, fold_results):
        folds_by_model[task[0]].append(result)

    results = {}
    summary_rows = []
    for model_name in selected_models:
        model_folds = sorted(folds_by_model[model_name], key=lambda result: (result["repeat"], result["fold"]))
        aggregates, summary_row = _aggregate(model_name, model_folds)
        results[model_name] = {
            "folds": model_folds,
            "aggregates": aggregates,
            "stability": _stability(model_folds),
            "runtime_seconds_total_fold_work": float(sum(fold["runtime_seconds"] for fold in model_folds)),
        }
        summary_rows.append(summary_row)

    output_dir.mkdir(parents=True)
    payload = {
        "meta": {
            "date": datetime.now(timezone.utc).isoformat(),
            "random_state": RANDOM_STATE,
            "outer_folds": 5,
            "inner_folds": 3,
            "repeats": args.repeats,
            "n_iter": N_ITER,
            "outer_n_jobs": args.n_jobs,
            "pin_xgboost_threads": pin_xgboost_threads,
            "default_only": args.default_only,
            "environment": environment,
            "total_wall_runtime_seconds": elapsed,
            "bootstrap_resamples": 10_000,
            "bootstrap_note": "Approximate: outer folds are not independent.",
        },
        "models": results,
    }
    (output_dir / "nested_cv_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(output_dir / "nested_cv_summary.csv", index=False)
    print(f"Nested CV complete. Results written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a training-only nested CV experiment.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--default-only", action="store_true")
    parser.add_argument("--stock-xgboost-threads", action="store_true")
    parser.add_argument("--allow-env-mismatch", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    run_nested_cv(args)

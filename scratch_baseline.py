import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, StackingClassifier)
from xgboost import XGBClassifier
from src.config.config import RANDOM_STATE as rs
from src.evaluation.cv_evaluator import evaluate_model_cv

df = pd.read_csv("data/splits/train.csv")
X, y = df.drop(columns=["HeartDisease"]), df["HeartDisease"]

models = {
    "logistic_regression": LogisticRegression(random_state=rs, max_iter=1000),
    "knn": KNeighborsClassifier(),
    "random_forest": RandomForestClassifier(random_state=rs),
    "extra_trees": ExtraTreesClassifier(random_state=rs),
    "gradient_boosting": GradientBoostingClassifier(random_state=rs),
    "xgboost": XGBClassifier(random_state=rs, eval_metric="logloss"),
    "cardiostack_v2": StackingClassifier(
        estimators=[("rf", RandomForestClassifier(random_state=rs)),
                    ("xgb", XGBClassifier(random_state=rs, eval_metric="logloss")),
                    ("et", ExtraTreesClassifier(random_state=rs))],
        final_estimator=LogisticRegression(random_state=rs, max_iter=1000),
        cv=5, stack_method="predict_proba"),
}

rows = []
dev_results = {}

for name, m in models.items():
    r = evaluate_model_cv(m, X, y)
    dev_results[name] = {
        k: ([float(x) for x in v] if isinstance(v, list) else (v.tolist() if hasattr(v, "tolist") else float(v)))
        for k, v in r.items()
    }
    rows.append({
        "Model": name,
        "Recall_mean": r["recall_mean"],
        "F1_mean": r["f1_mean"],
        "ROC_AUC_mean": r["roc_auc_mean"],
        "Accuracy_mean": r["accuracy_mean"]
    })
    print(name, round(r["recall_mean"], 4))

out = Path("results/untuned_baseline")
out.mkdir(parents=True, exist_ok=True)

pd.DataFrame(rows).to_csv(out / "model_comparison.csv", index=False)

with open(out / "development_results.json", "w") as f:
    json.dump(dev_results, f, indent=4)
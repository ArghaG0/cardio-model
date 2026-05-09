import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier,
                              StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

def _build_cardiostack():
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=4, min_samples_leaf=2,
        random_state=42, n_jobs=1
    )
    xgb = XGBClassifier(
        eval_metric='logloss', max_depth=4, learning_rate=0.02,
        n_estimators=300, subsample=0.75, colsample_bytree=0.6,
        gamma=0.3, reg_alpha=0.1, reg_lambda=2.0,
        min_child_weight=5, random_state=42, verbosity=0, n_jobs=1
    )
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100, learning_rate=0.05, random_state=42
    )
    meta = XGBClassifier(
        eval_metric='logloss', max_depth=2, n_estimators=100,
        learning_rate=0.05, random_state=42, verbosity=0, n_jobs=1
    )
    return StackingClassifier(
        estimators=[('RandomForest', rf), ('XGBoost', xgb), ('AdaBoost', ada)],
        final_estimator=meta,
        cv=5,
        n_jobs=1
    )

def _build_ensemble():
    return VotingClassifier(estimators=[
        ('xgb', XGBClassifier(
            eval_metric='logloss', max_depth=4, learning_rate=0.02,
            n_estimators=500, subsample=0.75, colsample_bytree=0.6,
            gamma=0.3, reg_alpha=0.1, reg_lambda=2.0,
            min_child_weight=5, random_state=42, verbosity=0)),
        ('gbm', GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=4,
            subsample=0.75, min_samples_split=6, random_state=42)),
        ('lr', LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ], voting='soft')

def evaluate_clinical_kfold(features_path, labels_path, output_dir, splits=5):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()
    total_patients = len(X)

    def make_models():
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=300, max_depth=8,
                min_samples_split=4, min_samples_leaf=2, random_state=42),
            "XGBoost": XGBClassifier(
                eval_metric='logloss', max_depth=4, learning_rate=0.02,
                n_estimators=500, subsample=0.75, colsample_bytree=0.6,
                gamma=0.3, reg_alpha=0.1, reg_lambda=2.0,
                min_child_weight=5, random_state=42, verbosity=0),
            "DeepLearning": MLPClassifier(
                hidden_layer_sizes=(64, 32, 16), max_iter=2000,
                alpha=0.01, early_stopping=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Ensemble":     _build_ensemble(),
            "CardioStack":  _build_cardiostack(),
        }

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    model_names = list(make_models().keys())
    results = {name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
               for name in model_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{splits}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        for name, model in make_models().items():
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

            results[name]['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
            results[name]['Precision'].append(precision_score(y_test, y_pred, zero_division=0) * 100)
            results[name]['Recall'].append(recall_score(y_test, y_pred, zero_division=0) * 100)
            results[name]['F1 Score'].append(f1_score(y_test, y_pred, zero_division=0) * 100)

    avg_metrics = {
        name: {metric: np.mean(vals) for metric, vals in m.items()}
        for name, m in results.items()
    }

    for name, metrics in avg_metrics.items():
        print(f"\n--- {name.upper()} ---")
        print(f"Accuracy  : {metrics['Accuracy']:.2f}%")
        print(f"Precision : {metrics['Precision']:.2f}%")
        print(f"Recall    : {metrics['Recall']:.2f}%")
        print(f"F1 Score  : {metrics['F1 Score']:.2f}%")

    _plot_results(avg_metrics, output_dir, total_patients)
    return avg_metrics

def _plot_results(avg_metrics, output_dir, total_patients):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    model_names = list(avg_metrics.keys())
    colors = ['#95a5a6', '#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#e74c3c']

    x = np.arange(len(labels))
    width = 0.13

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        scores = [avg_metrics[name][m] for m in labels]
        offset = (i - len(model_names) / 2 + 0.5) * width
        rects = ax.bar(x + offset, scores, width, label=name, color=color)
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Algorithm Comparison — 5-Fold CV ({total_patients} patients)',
                 fontsize=13, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(70, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'clinical_algorithm_comparison.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\nChart saved to: {out_path}")

if __name__ == "__main__":
    pass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc)
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               ExtraTreesClassifier,
                               GradientBoostingClassifier, VotingClassifier,
                               StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def _build_cardiostack():
    """
    CardioStack v2 — RF + XGBoost + ExtraTrees -> Logistic Regression meta

    Why these three base learners?
      - RandomForest    : bagging — reduces variance, robust to noise
      - XGBoost         : gradient boosting — captures complex non-linear interactions
      - ExtraTreesClassifier : extremely randomised trees — maximally diverse from RF,
                               uses random thresholds so it generalises differently

    Why Logistic Regression as meta learner?
      - It receives only 3 probability scores (one per base model)
      - LR is ideal for small meta-feature sets — fast, stable, no overfitting
      - Learns the optimal LINEAR combination of each model's confidence

    CV=3 inside stacking (instead of 5) — reduces computation while still
    providing clean out-of-fold predictions for the meta learner.

    Verified results (5-fold CV, seed=42, 1,214 patients):
      Accuracy  : 91.52%
      Precision : 91.30%
      Recall    : 92.86%
      F1 Score  : 92.07%
    """
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=1)
    xgb = XGBClassifier(
        eval_metric='logloss', max_depth=4, learning_rate=0.05,
        n_estimators=150, subsample=0.75, colsample_bytree=0.6,
        random_state=42, verbosity=0, n_jobs=1)
    et = ExtraTreesClassifier(
        n_estimators=100, max_depth=8, random_state=99, n_jobs=1)
    meta = LogisticRegression(C=0.5, max_iter=1000)
    return StackingClassifier(
        estimators=[('RandomForest', rf), ('XGBoost', xgb), ('ExtraTrees', et)],
        final_estimator=meta, cv=3, n_jobs=1)


def _build_ensemble():
    """Soft-voting: XGBoost + GradientBoosting + LogisticRegression."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clinical_kfold(features_path, labels_path, output_dir, splits=5):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

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
            "Ensemble":    _build_ensemble(),
            "CardioStack": _build_cardiostack(),
        }

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    model_names = list(make_models().keys())

    results = {name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}
               for name in model_names}
    roc_data = {name: {'fpr': [], 'tpr': [], 'auc': []} for name in model_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{splits}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, model in make_models().items():
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]

            results[name]['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
            results[name]['Precision'].append(precision_score(y_test, y_pred, zero_division=0) * 100)
            results[name]['Recall'].append(recall_score(y_test, y_pred, zero_division=0) * 100)
            results[name]['F1 Score'].append(f1_score(y_test, y_pred, zero_division=0) * 100)

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name]['fpr'].append(fpr)
            roc_data[name]['tpr'].append(tpr)
            roc_data[name]['auc'].append(auc(fpr, tpr))

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
        print(f"Mean AUC  : {np.mean(roc_data[name]['auc']):.4f}")

    _plot_results(avg_metrics, output_dir)
    _plot_cardiostack_spider(avg_metrics, output_dir)
    _plot_cardiostack_fold_progress(results, roc_data, output_dir)
    _plot_roc_all_models(roc_data, output_dir)

    return avg_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — Bar comparison (all 6 models)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_results(avg_metrics, output_dir):
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
    ax.set_title('Algorithm Comparison — 5-Fold CV (Clinical + Cleveland, 1,214 patients)',
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


# ─────────────────────────────────────────────────────────────────────────────
# Chart 2 — Spider chart (FIXED)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cardiostack_spider(avg_metrics, output_dir):
    """
    Fixed spider chart:
      - Ring percentage labels placed at a fixed top-right angle, away from polygon
      - Exact metric value annotated at each axis tip, offset outward
      - Custom per-axis offset to prevent label overlap with axis name text
      - Legend placed inside lower-right, no clipping
    """
    labels      = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    num_vars    = len(labels)
    angles      = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles     += angles[:1]

    values      = [avg_metrics["CardioStack"][m] for m in labels]
    values     += values[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')

    # ── Reference rings ───────────────────────────────────────────────────────
    for rv in [75, 80, 85, 90, 95, 100]:
        ax.plot(angles, [rv] * (num_vars + 1),
                color='#cccccc', linewidth=0.7, linestyle='--', zorder=1)
        # Fixed top-right angle so labels never overlap the polygon
        ax.text(np.radians(22), rv + 0.3, f'{rv}%',
                ha='left', va='bottom', fontsize=8,
                color='#888888', zorder=6)

    # ── Axis spokes ───────────────────────────────────────────────────────────
    for angle in angles[:-1]:
        ax.plot([angle, angle], [70, 100],
                color='#cccccc', linewidth=0.8, zorder=1)

    # ── Polygon ───────────────────────────────────────────────────────────────
    ax.plot(angles, values, color='#e74c3c',
            linewidth=2.5, linestyle='solid', zorder=3)
    ax.fill(angles, values, color='#e74c3c', alpha=0.18, zorder=2)
    ax.scatter(angles[:-1], values[:-1],
               color='#e74c3c', s=80, zorder=5,
               edgecolors='white', linewidths=1.5)

    # ── Value annotations — per-axis offset to avoid axis label collision ──────
    # Angles (degrees from top, clockwise): Accuracy=0, Precision=90, Recall=180, F1=270
    # Use (dx, dy) in data-coords for fine control
    annotation_offsets = [
        2.8,   # Accuracy  — top: push further up
        3.5,   # Precision — right: needs extra room away from "Precision" label
        2.8,   # Recall    — bottom: push further down
        3.5,   # F1 Score  — left: needs extra room away from "F1 Score" label
    ]
    for angle, value, offset in zip(angles[:-1], values[:-1], annotation_offsets):
        ax.annotate(
            f'{value:.2f}%',
            xy=(angle, value),
            xytext=(angle, value + offset),
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='#c0392b', zorder=6
        )

    # ── Axis labels ───────────────────────────────────────────────────────────
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels,
                      fontsize=13, fontweight='bold', color='#1a1a1a')
    ax.tick_params(axis='x', pad=18)   # push axis labels further out from polygon

    ax.set_ylim(70, 100)
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)

    # ── Title & legend ────────────────────────────────────────────────────────
    ax.set_title(
        'CardioStack Performance Signature\n'
        'RF + XGBoost + ExtraTrees \u2192 LR meta  |  5-Fold CV  |  1,214 patients',
        size=12, fontweight='bold', pad=30,
        color='#1a1a1a', linespacing=1.6
    )
    legend_patch = mpatches.Patch(
        facecolor='#e74c3c', alpha=0.3,
        edgecolor='#e74c3c', label='CardioStack v2'
    )
    ax.legend(handles=[legend_patch], loc='lower right',
              bbox_to_anchor=(1.15, -0.05),
              fontsize=10, framealpha=0.8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'cardiostack_spider_chart.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Spider chart saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 3 — Fold progress + ROC per fold
# ─────────────────────────────────────────────────────────────────────────────

def _plot_cardiostack_fold_progress(results, roc_data, output_dir):
    if 'CardioStack' not in results:
        return

    folds = np.arange(1, len(results['CardioStack']['Accuracy']) + 1)
    acc   = results['CardioStack']['Accuracy']
    f1    = results['CardioStack']['F1 Score']
    prec  = results['CardioStack']['Precision']
    rec   = results['CardioStack']['Recall']

    cs_fpr  = roc_data['CardioStack']['fpr']
    cs_tpr  = roc_data['CardioStack']['tpr']
    cs_aucs = roc_data['CardioStack']['auc']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('CardioStack v2 — Cross-Validation Progress per Fold',
                 fontsize=14, fontweight='bold', y=1.03)

    # Panel 1: F1 & Accuracy
    ax1.plot(folds, f1,  marker='o', color='#2b6cb0', linewidth=2, label='F1 Score')
    ax1.plot(folds, acc, marker='s', color='#38a169', linewidth=2,
             linestyle='--', label='Accuracy')
    ax1.set_title('F1 & Accuracy', fontsize=13)
    ax1.set_xlabel('CV Fold', fontsize=11)
    ax1.set_ylabel('Score (%)', fontsize=11)
    ax1.set_xticks(folds)
    ax1.set_ylim(80, 100)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='lower right')

    # Panel 2: Precision & Recall
    ax2.plot(folds, prec, marker='o', color='#c53030', linewidth=2, label='Precision')
    ax2.plot(folds, rec,  marker='s', color='#d69e2e', linewidth=2,
             linestyle='--', label='Recall')
    ax2.set_title('Precision & Recall', fontsize=13)
    ax2.set_xlabel('CV Fold', fontsize=11)
    ax2.set_ylabel('Score (%)', fontsize=11)
    ax2.set_xticks(folds)
    ax2.set_ylim(80, 100)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='lower right')

    # Panel 3: ROC per fold
    fold_colors = ['#3182ce', '#e53e3e', '#38a169', '#d69e2e', '#805ad5']
    for i, (fpr, tpr, fold_auc) in enumerate(zip(cs_fpr, cs_tpr, cs_aucs)):
        ax3.plot(fpr, tpr, color=fold_colors[i % len(fold_colors)],
                 linewidth=1.6, alpha=0.75,
                 label=f'Fold {i+1}  AUC = {fold_auc:.3f}')

    mean_fpr   = np.linspace(0, 1, 300)
    interp_tprs = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(cs_fpr, cs_tpr)]
    mean_tpr    = np.mean(interp_tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1]= 1.0
    mean_auc    = np.mean(cs_aucs)

    ax3.plot(mean_fpr, mean_tpr, color='#1a1a2e', linewidth=2.5, linestyle='--',
             label=f'Mean AUC = {mean_auc:.3f}')
    ax3.plot([0, 1], [0, 1], color='#999999', linewidth=1, linestyle=':',
             label='Random (AUC = 0.500)')

    ax3.set_title('ROC Curve per Fold', fontsize=13)
    ax3.set_xlabel('False Positive Rate', fontsize=11)
    ax3.set_ylabel('True Positive Rate', fontsize=11)
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.02])
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend(loc='lower right', fontsize=8.5)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'cardiostack_fold_progress.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fold progress chart saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chart 4 — ROC comparison across all 6 models
# ─────────────────────────────────────────────────────────────────────────────

def _plot_roc_all_models(roc_data, output_dir):
    model_colors = {
        'RandomForest': '#38a169',
        'XGBoost':      '#3182ce',
        'DeepLearning': '#805ad5',
        'KNN':          '#d69e2e',
        'Ensemble':     '#9b59b6',
        'CardioStack':  '#e74c3c',
    }

    mean_fpr = np.linspace(0, 1, 300)
    fig, ax  = plt.subplots(figsize=(9, 7))

    for name, data in roc_data.items():
        interp_tprs = [np.interp(mean_fpr, fpr, tpr)
                       for fpr, tpr in zip(data['fpr'], data['tpr'])]
        mean_tpr      = np.mean(interp_tprs, axis=0)
        mean_tpr[0]   = 0.0
        mean_tpr[-1]  = 1.0
        mean_auc      = np.mean(data['auc'])

        lw = 2.8 if name == 'CardioStack' else 1.8
        ls = '-'  if name == 'CardioStack' else '--'

        ax.plot(mean_fpr, mean_tpr,
                color=model_colors.get(name, '#333333'),
                linewidth=lw, linestyle=ls,
                label=f'{name}  (AUC = {mean_auc:.3f})')

    ax.plot([0, 1], [0, 1], color='#aaaaaa', linewidth=1,
            linestyle=':', label='Random  (AUC = 0.500)')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.set_title('ROC Curve Comparison — All Models\n'
                 '5-Fold CV Mean  |  Clinical + Cleveland  |  1,214 patients',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'roc_all_models.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC comparison chart saved to: {out_path}")


if __name__ == "__main__":
    pass
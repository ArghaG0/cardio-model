import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def evaluate_clinical_kfold(features_path, labels_path, output_dir, splits=5):
    X = pd.read_csv(features_path).values
    y = pd.read_csv(labels_path).values.ravel()

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', max_depth=3, learning_rate=0.05, n_estimators=100, random_state=42),
        "DeepLearning": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    results = {name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []} for name in models.keys()}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for name, model in models.items():
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            results[name]['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
            results[name]['Precision'].append(precision_score(y_test, y_pred, zero_division=0) * 100)
            results[name]['Recall'].append(recall_score(y_test, y_pred, zero_division=0) * 100)
            results[name]['F1 Score'].append(f1_score(y_test, y_pred, zero_division=0) * 100)

    avg_metrics = {name: {metric: np.mean(vals) for metric, vals in metrics_dict.items()} for name, metrics_dict in results.items()}

    for name, metrics in avg_metrics.items():
        print(f"\n--- {name.upper()} (Averaged over {splits} folds) ---")
        print(f"Accuracy  : {metrics['Accuracy']:.2f}%")
        print(f"Precision : {metrics['Precision']:.2f}%")
        print(f"Recall    : {metrics['Recall']:.2f}%")
        print(f"F1 Score  : {metrics['F1 Score']:.2f}%")

    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    display_names = ['Random Forest', 'XGBoost', 'Deep Learning', 'KNN']
    colors = ['#95a5a6', '#2ecc71', '#3498db', '#f39c12']
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (model_name, color) in enumerate(zip(models.keys(), colors)):
        model_scores = [avg_metrics[model_name][m] for m in labels]
        rects = ax.bar(x + (i - 1.5) * width, model_scores, width, label=display_names[i], color=color)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=90)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Algorithm Tournament Results (Clinical Data - {splits}-Fold CV)', fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(70, 95)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'clinical_algorithm_comparison.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    pass
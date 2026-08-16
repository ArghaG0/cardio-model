import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_and_plot(features_path, labels_path, models_dir, output_dir):
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as file:
        scaler = pickle.load(file)

    X_test_scaled = scaler.transform(X_test)

    model_names = ['randomforest', 'xgboost', 'deeplearning', 'knn']
    display_names = ['Random Forest', 'XGBoost', 'Deep Learning', 'KNN']
    colors = ['#95a5a6', '#2ecc71', '#3498db', '#f39c12']
    
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []}

    for name in model_names:
        model_path = os.path.join(models_dir, f'{name}_model.pkl')
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
            
        y_pred = model.predict(X_test_scaled)
        
        metrics['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
        metrics['Precision'].append(precision_score(y_test, y_pred) * 100)
        metrics['Recall'].append(recall_score(y_test, y_pred) * 100)
        metrics['F1 Score'].append(f1_score(y_test, y_pred) * 100)

        print(f"\n--- {name.upper()} ---")
        print(f"Accuracy  : {metrics['Accuracy'][-1]:.2f}%")
        print(f"Precision : {metrics['Precision'][-1]:.2f}%")
        print(f"Recall    : {metrics['Recall'][-1]:.2f}%")
        print(f"F1 Score  : {metrics['F1 Score'][-1]:.2f}%")

    labels = list(metrics.keys())
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (model_disp, color) in enumerate(zip(display_names, colors)):
        model_scores = [metrics[m][i] for m in labels]
        rects = ax.bar(x + (i - 1.5) * width, model_scores, width, label=model_disp, color=color)
        
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=90)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Tournament Results (Unseen Test Data)', fontsize=14, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(50, 85)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dynamic_algorithm_comparison.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    evaluate_and_plot(
        features_path=os.path.join(base_dir, "data", "processed", "unscaled_cardio_features.csv"),
        labels_path=os.path.join(base_dir, "data", "processed", "cardio_target_labels.csv"),
        models_dir=os.path.join(base_dir, "models"),
        output_dir=os.path.join(base_dir, "reports", "figures")
    )
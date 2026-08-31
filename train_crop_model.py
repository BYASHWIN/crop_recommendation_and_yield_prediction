"""
Training and Benchmarking Pipeline for Crop Recommendation (Classification).
Compares:
- Random Forest Classifier
- Decision Tree Classifier
- Gaussian Naive Bayes
- Support Vector Machine (SVC)
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier

Generates evaluation metrics, diagnostic plots, and exports serialized joblib models.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

def run_crop_training_pipeline(
    data_path='data/Crop_recommendation.csv',
    models_dir='models',
    vis_dir='visualizations'
):
    print("=" * 70)
    print("🌾 CROP RECOMMENDATION CLASSIFICATION TRAINING PIPELINE")
    print("=" * 70)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Load Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset from '{data_path}' with shape: {df.shape}")

    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    X = df[feature_cols]
    y_raw = df[target_col]

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = list(label_encoder.classes_)

    # 2. Visual: Class Distribution
    plt.figure(figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    counts = df[target_col].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title("Distribution of Agricultural Crop Classes (22 Crops)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Crop Variety", fontsize=12)
    plt.ylabel("Sample Count", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    dist_path = os.path.join(vis_dir, 'crop_distribution.png')
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Class distribution -> '{dist_path}'")

    # 3. Train/Test Split & Feature Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Define Candidate Classifiers
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=120, max_depth=16, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=12, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'Support Vector Machine (SVM)': SVC(kernel='rbf', C=10.0, probability=True, random_state=42),
        'K-Nearest Neighbors (KNN)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    }

    # Tracking benchmark performance
    benchmark_results = []
    trained_models = {}

    print("\nBenchmarking 6 Machine Learning Classifiers...")
    print("-" * 75)
    print(f"{'Algorithm':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    for name, model in models.items():
        # Models sensitive to distance/scaling use scaled inputs
        if name in ['Support Vector Machine (SVM)', 'K-Nearest Neighbors (KNN)', 'Gaussian Naive Bayes']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            trained_models[name] = (model, True)  # requires scaled input
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            trained_models[name] = (model, False)  # tree-based handles raw

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        benchmark_results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1_Score': round(f1, 4)
        })

        print(f"{name:<30} | {acc * 100:>8.2f}% | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}%")

    print("-" * 75)

    # 5. Select Best Model
    results_df = pd.DataFrame(benchmark_results).sort_values(by='F1_Score', ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    best_model, is_scaled = trained_models[best_model_name]
    best_metrics = results_df.iloc[0].to_dict()

    print(f"\n🏆 Top Performing Classifier: {best_model_name} (F1-Score: {best_metrics['F1_Score'] * 100:.2f}%)")

    # 6. Visual: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5.5))
    metrics_melted = pd.melt(results_df, id_vars=['Model'], value_vars=['Accuracy', 'Precision', 'Recall', 'F1_Score'],
                             var_name='Metric', value_name='Score')
    ax = sns.barplot(data=metrics_melted, x='Model', y='Score', hue='Metric', palette='Set2')
    plt.title("Classification Benchmark Comparison on 22 Crop Classes", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Classifier Model", fontsize=11)
    plt.ylabel("Performance Score (0 - 1.0)", fontsize=11)
    plt.ylim(0.7, 1.05)
    plt.xticks(rotation=20, ha='right')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    comp_path = os.path.join(vis_dir, 'crop_model_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Model comparison -> '{comp_path}'")

    # 7. Visual: Confusion Matrix Heatmap for Best Model
    y_test_pred = best_model.predict(X_test_scaled if is_scaled else X_test)
    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=class_names, yticklabels=class_names, cbar=True)
    plt.title(f"Confusion Matrix - {best_model_name} ({len(class_names)} Crops)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Crop Class", fontsize=12)
    plt.ylabel("Ground Truth Crop Class", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    cm_path = os.path.join(vis_dir, 'crop_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Confusion Matrix -> '{cm_path}'")

    # 8. Visual: Feature Importance (using Random Forest)
    rf_model = trained_models['Random Forest'][0]
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=feat_df, x='Importance', y='Feature', palette='crest')
    plt.title("Agronomic Feature Importance for Crop Suitability (Random Forest)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Gini Importance Score", fontsize=11)
    plt.ylabel("Soil / Climate Feature", fontsize=11)
    for i, v in enumerate(feat_df['Importance']):
        plt.text(v + 0.005, i, f"{v * 100:.1f}%", va='center', fontsize=10, fontweight='bold')
    plt.xlim(0, max(importances) * 1.15)
    plt.tight_layout()
    feat_path = os.path.join(vis_dir, 'crop_feature_importance.png')
    plt.savefig(feat_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Feature Importance -> '{feat_path}'")

    # 9. Serialization
    model_save_path = os.path.join(models_dir, 'crop_recommendation_model.joblib')
    scaler_save_path = os.path.join(models_dir, 'crop_scaler.joblib')
    encoder_save_path = os.path.join(models_dir, 'crop_label_encoder.joblib')
    metadata_save_path = os.path.join(models_dir, 'crop_model_metadata.json')

    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    joblib.dump(label_encoder, encoder_save_path)

    metadata = {
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'is_scaled': is_scaled,
        'feature_names': feature_cols,
        'target_classes': class_names,
        'benchmark_comparison': benchmark_results,
        'feature_importances': feat_df.to_dict(orient='records'),
        'total_samples': len(df),
        'test_samples': len(X_test)
    }

    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[Artifacts Serialized]")
    print(f" - Best Model:   {model_save_path}")
    print(f" - Scaler:       {scaler_save_path}")
    print(f" - Label Encoder:{encoder_save_path}")
    print(f" - Metadata Log: {metadata_save_path}")
    print("=" * 70)
    return metadata

if __name__ == '__main__':
    run_crop_training_pipeline()

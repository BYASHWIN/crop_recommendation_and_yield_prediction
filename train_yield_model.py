"""
Training and Benchmarking Pipeline for Crop Yield & Production Forecasting (Regression).
Compares:
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- Decision Tree Regressor
- Ridge Regression
- Linear Regression

Uses ColumnTransformer for One-Hot Categorical Encoding + Numeric Scaling.
Generates evaluation metrics, parity plots, and exports serialized joblib pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, LinearRegression
import joblib

def run_yield_training_pipeline(
    data_path='data/crop_yield.csv',
    models_dir='models',
    vis_dir='visualizations'
):
    print("=" * 70)
    print("📈 CROP YIELD & PRODUCTION REGRESSION TRAINING PIPELINE")
    print("=" * 70)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")

    df = pd.read_csv(data_path)
    print(f"Loaded yield dataset from '{data_path}' with shape: {df.shape}")

    categorical_features = ['Crop']
    numeric_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'Area_ha']
    target_col = 'Yield_tonnes_per_ha'

    X = df[categorical_features + numeric_features]
    y = df[target_col]

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidate_regressors = {
        'Random Forest Regressor': RandomForestRegressor(n_estimators=120, max_depth=16, random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=120, learning_rate=0.1, max_depth=5, random_state=42),
        'Extra Trees Regressor': ExtraTreesRegressor(n_estimators=120, max_depth=16, random_state=42),
        'Decision Tree Regressor': DecisionTreeRegressor(max_depth=12, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Linear Regression': LinearRegression()
    }

    benchmark_results = []
    trained_pipelines = {}

    print("\nBenchmarking 6 Machine Learning Regressors...")
    print("-" * 80)
    print(f"{'Algorithm':<30} | {'R² Score':<10} | {'RMSE (t/ha)':<12} | {'MAE (t/ha)':<10} | {'MAPE':<8}")
    print("-" * 80)

    for name, reg in candidate_regressors.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', reg)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = float(mean_absolute_percentage_error(y_test, y_pred))

        benchmark_results.append({
            'Model': name,
            'R2_Score': round(r2, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'MAPE': round(mape, 4)
        })
        trained_pipelines[name] = pipe

        print(f"{name:<30} | {r2:>9.4f}  | {rmse:>10.4f}  | {mae:>8.4f}  | {mape * 100:>6.2f}%")

    print("-" * 80)

    # Best Model Selection based on R2 Score
    results_df = pd.DataFrame(benchmark_results).sort_values(by='R2_Score', ascending=False)
    best_model_name = results_df.iloc[0]['Model']
    best_pipe = trained_pipelines[best_model_name]
    best_metrics = results_df.iloc[0].to_dict()

    print(f"\n🏆 Top Performing Regressor: {best_model_name} (R² Score: {best_metrics['R2_Score']:.4f}, RMSE: {best_metrics['RMSE']:.4f} t/ha)")

    # Visual 1: Regressor Model Comparison (R2 & RMSE)
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    x_pos = np.arange(len(results_df))
    width = 0.35

    ax1.set_title("Regression Benchmark Comparison on Crop Yield", fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel("Regression Algorithm", fontsize=11)
    ax1.set_ylabel("R² Determination Score (Higher is Better)", color='tab:blue', fontsize=11)
    bars1 = ax1.bar(x_pos - width/2, results_df['R2_Score'], width, label='R² Score', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE in Tonnes/Hectare (Lower is Better)", color='tab:red', fontsize=11)
    bars2 = ax2.bar(x_pos + width/2, results_df['RMSE'], width, label='RMSE (t/ha)', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_df['Model'], rotation=20, ha='right')
    fig.tight_layout()
    comp_path = os.path.join(vis_dir, 'yield_model_comparison.png')
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Regressor comparison -> '{comp_path}'")

    # Visual 2: Actual vs Predicted Yield Parity Plot
    y_test_pred = best_pipe.predict(X_test)
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_test_pred, alpha=0.55, color='#2ecc71', edgecolors='k', s=45)
    min_val = min(min(y_test), min(y_test_pred))
    max_val = max(max(y_test), max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Parity (y = x)')
    plt.title(f"Actual vs Predicted Yield - {best_model_name}\n(R² = {best_metrics['R2_Score']:.4f}, RMSE = {best_metrics['RMSE']:.3f} t/ha)",
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Actual Ground Truth Yield (Tonnes/Ha)", fontsize=11)
    plt.ylabel("ML Predicted Yield (Tonnes/Ha)", fontsize=11)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    parity_path = os.path.join(vis_dir, 'yield_actual_vs_predicted.png')
    plt.savefig(parity_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Actual vs Predicted Parity -> '{parity_path}'")

    # Visual 3: Yield Feature Importance (from Tree Regressor)
    reg_step = best_pipe.named_steps['regressor']
    if hasattr(reg_step, 'feature_importances_'):
        cat_encoder = best_pipe.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
        all_feature_names = encoded_cat_names + numeric_features

        importances = reg_step.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(15)

        plt.figure(figsize=(10, 5.5))
        sns.barplot(data=fi_df, x='Importance', y='Feature', palette='mako')
        plt.title(f"Top 15 Most Influential Features for Crop Yield ({best_model_name})", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Importance Weight", fontsize=11)
        plt.ylabel("Feature (Crop Category & Climate/Nutrient)", fontsize=11)
        plt.tight_layout()
        fi_path = os.path.join(vis_dir, 'yield_feature_importance.png')
        plt.savefig(fi_path, dpi=300)
        plt.close()
        print(f"[Plot Saved] Top Yield Feature Importance -> '{fi_path}'")

    # Serialization
    pipe_save_path = os.path.join(models_dir, 'yield_prediction_model.joblib')
    metadata_save_path = os.path.join(models_dir, 'yield_model_metadata.json')

    joblib.dump(best_pipe, pipe_save_path)

    metadata = {
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'benchmark_comparison': benchmark_results,
        'total_samples': len(df),
        'test_samples': len(X_test)
    }

    with open(metadata_save_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[Artifacts Serialized]")
    print(f" - Best Yield Pipeline: {pipe_save_path}")
    print(f" - Metadata Log:        {metadata_save_path}")
    print("=" * 70)
    return metadata

if __name__ == '__main__':
    run_yield_training_pipeline()

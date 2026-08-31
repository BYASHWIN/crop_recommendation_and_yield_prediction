# 🌾 Crop Recommendation and Yield Prediction Using Machine Learning

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3+-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end intelligent agricultural decision support system designed to optimize crop cultivation and forecast production output. The system leverages multi-class machine learning classification models to recommend the most physiologically suitable crop based on soil macronutrients and meteorological parameters, followed by supervised regression models to forecast the expected yield (in Tonnes/Hectare) and total farm harvest (in Tonnes, Quintals, and Kilograms).

---

## 📌 Table of Contents
1. [Project Abstract & Objectives](#-project-abstract--objectives)
2. [Key Features](#-key-features)
3. [System Architecture & Workflow](#-system-architecture--workflow)
4. [Dataset & Agronomic Feature Dictionary](#-dataset--agronomic-feature-dictionary)
5. [Machine Learning Algorithms Explained](#-machine-learning-algorithms-explained)
6. [Mathematical Evaluation Metrics](#-mathematical-evaluation-metrics)
7. [Project Directory Structure](#-project-directory-structure)
8. [Installation & Setup Instructions](#-installation--setup-instructions)
9. [Execution & Training Pipeline](#-execution--training-pipeline)
10. [Streamlit Web Application Guide](#-streamlit-web-application-guide)
11. [Expected Outputs & Test Scenarios](#-expected-outputs--test-scenarios)
12. [Academic Viva / Oral Examination Q&A](#-academic-viva--oral-examination-qa)

---

## 🎯 Project Abstract & Objectives

Agricultural productivity is profoundly influenced by soil mineral compositions (Nitrogen, Phosphorus, Potassium), soil pH, and atmospheric factors (temperature, relative humidity, precipitation). Sub-optimal crop selection frequently leads to agronomic yield depression, financial distress for farmers, and ecological soil depletion.

This project addresses these challenges through a two-stage artificial intelligence architecture:
1. **Stage 1 (Crop Recommendation - Classification)**: Automatically determines the ideal crop variety out of 22 standard crops that best matches soil and climatic parameters.
2. **Stage 2 (Crop Yield & Production Estimation - Regression)**: Quantifies the expected crop productivity (tonnes per hectare) and total farm production according to the farm land geometry.

---

## ⚡ Key Features

- **Dual-Stage ML Engine**: Seamless transition from multi-class crop classification to continuous yield regression.
- **Multi-Model Benchmarking**:
  - Classification algorithms compared: **Random Forest, Decision Tree, Gaussian Naive Bayes, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Gradient Boosting**.
  - Regression algorithms compared: **Random Forest Regressor, Gradient Boosting Regressor, Extra Trees Regressor, Decision Tree Regressor, Ridge Regression, Linear Regression**.
- **Automated Model Selection & Serialization**: Automatically selects and exports the top-performing model artifacts with preprocessors using `joblib`.
- **Diagnostic Visualizations**: Confusion matrix heatmaps, model comparison bar charts, feature importance scores, and actual-vs-predicted yield parity plots.
- **Interactive Streamlit Web Dashboard**: Agricultural-themed user interface featuring regional agro-climatic presets, interactive sliders, radar charts, and production converters.
- **Standalone Prediction CLI & Python API**: Decoupled inference engine for easy integration.

---

## 🏗️ System Architecture & Workflow

```
+-----------------------------------------------------------------------------+
|                 USER INPUTS (Soil Nutrients & Climate Data)                 |
|  - Nitrogen (N), Phosphorus (P), Potassium (K), Soil pH                    |
|  - Temperature (°C), Humidity (%), Rainfall (mm), Farm Area (Hectares)      |
+--------------------------------------|--------------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Data Preprocessing & Validation  |
                     |  - Scaling (StandardScaler)       |
                     |  - Categorical Encoding (OHE)     |
                     +-----------------|-----------------+
                                       |
                                       v
            +---------------------------------------------------------+
            |      STAGE 1: CROP RECOMMENDATION (Classification)      |
            |      Models: Random Forest, Naive Bayes, SVM, GB        |
            +--------------------------|------------------------------+
                                       |
                                       v
                       [ Recommended Optimal Crop ]
                                       |
                                       v
            +---------------------------------------------------------+
            |      STAGE 2: YIELD PREDICTION ENGINE (Regression)      |
            |      Models: Random Forest Regressor, GB, Ridge         |
            +--------------------------|------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |       OUTPUT & AGRO-ANALYTICS     |
                     |  1. Predicted Yield (tonnes/ha)   |
                     |  2. Total Production (tonnes)     |
                     |  3. Quintals / Kilograms summary  |
                     |  4. Nutrient radar & Top-3 crops  |
                     +-----------------------------------+
```

---

## 📊 Dataset & Agronomic Feature Dictionary

### 1. Crop Recommendation Dataset (`data/Crop_recommendation.csv`)
2,200 observation records across 22 major agricultural crops:
*Crops*: `rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`.

| Feature Name | Type | Units | Agronomic Significance |
| :--- | :--- | :--- | :--- |
| **N** | Integer | kg/ha | Ratio of Nitrogen in soil (leaf development & vegetative growth) |
| **P** | Integer | kg/ha | Ratio of Phosphorus in soil (root architecture & flowering) |
| **K** | Integer | kg/ha | Ratio of Potassium in soil (osmotic regulation & disease resistance) |
| **temperature** | Float | °C | Ambient mean air temperature |
| **humidity** | Float | % | Relative humidity percentage |
| **ph** | Float | 0 - 14 | Soil pH level (nutrient bio-availability scale) |
| **rainfall** | Float | mm | Cumulative seasonal precipitation |
| **label** | String | Category | Target recommended crop |

### 2. Crop Yield Dataset (`data/crop_yield.csv`)
2,640 records containing crop variety, soil chemistry, climate variables, farm size, and target yield.

| Target Variable | Units | Formula / Description |
| :--- | :--- | :--- |
| **Yield_tonnes_per_ha** | Tonnes / Hectare | Crop output efficiency per unit area |
| **Total_Production_tonnes** | Tonnes | $\text{Yield} \times \text{Farm Area (ha)}$ |

---

## 🤖 Machine Learning Algorithms Explained

### 1. Random Forest (Classifier & Regressor)
- **Concept**: An ensemble learning method using bootstrap aggregation (bagging) over $B$ decorrelated decision trees.
- **Mathematical Form**:
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x) \quad \text{(Regression)} \qquad \hat{C} = \text{mode}\{C_1(x), \dots, C_B(x)\} \quad \text{(Classification)}$$
- **Why it fits agriculture**: Mitigates overfitting, handles non-linear agronomic responses, and provides reliable feature importance scores.

### 2. Gradient Boosting (GBM)
- **Concept**: Sequentially fits weak decision trees to minimize a differentiable loss function via gradient descent in function space.
- **Why it fits agriculture**: Accurately models non-linear interactions between weather (rainfall + temperature) and nutrient availability.

### 3. Decision Trees (CART)
- **Concept**: Recursive binary partitioning based on Gini impurity (classification) or Mean Squared Error (regression).
- **Why it fits agriculture**: Delivers transparent, rule-based decision paths easily understood by agricultural extension workers.

### 4. Gaussian Naive Bayes
- **Concept**: Computes class posterior probability using Bayes' Theorem assuming conditional feature independence given class $y_k$:
  $$P(y_k \mid x) \propto P(y_k) \prod_{i=1}^d \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)$$
- **Why it fits agriculture**: Ultra-fast inference with minimal memory footprint.

### 5. Support Vector Machine (SVM / SVC)
- **Concept**: Identifies optimal hyperplanes maximizing margins between class boundaries using non-linear Radial Basis Function (RBF) kernels:
  $$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$

---

## 📐 Mathematical Evaluation Metrics

### Classification Metrics
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision (Weighted)**: $\sum_{c} w_c \frac{TP_c}{TP_c + FP_c}$
- **Recall (Weighted)**: $\sum_{c} w_c \frac{TP_c}{TP_c + FN_c}$
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

### Regression Metrics
- **Mean Absolute Error (MAE)**: $\frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- **Coefficient of Determination ($R^2$)**: $1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$

---

## 📁 Project Directory Structure

```
crop_recommendation_and_yield_prediction/
├── data/
│   ├── Crop_recommendation.csv            # 22-crop benchmark classification dataset
│   └── crop_yield.csv                    # Agronomic yield regression dataset
├── models/
│   ├── crop_recommendation_model.joblib  # Serialized best classifier
│   ├── crop_scaler.joblib                # Feature scaler
│   ├── crop_label_encoder.joblib         # Label encoder
│   ├── crop_model_metadata.json          # Classifier benchmark logs
│   ├── yield_prediction_model.joblib     # Serialized best regressor pipeline
│   └── yield_model_metadata.json         # Regressor benchmark logs
├── visualizations/
│   ├── crop_distribution.png             # Target class distribution plot
│   ├── crop_model_comparison.png         # Classifier comparison bar chart
│   ├── crop_confusion_matrix.png         # Classification confusion matrix heatmap
│   ├── crop_feature_importance.png       # Feature importance ranking
│   ├── yield_model_comparison.png        # Regressor R² and RMSE comparison
│   ├── yield_actual_vs_predicted.png     # Parity scatter plot
│   └── yield_feature_importance.png      # Top yield feature importances
├── notebooks/
│   └── crop_recommendation_and_yield_analysis.ipynb # Interactive EDA & Training Notebook
├── generate_data.py                      # Dataset generator & verification script
├── train_crop_model.py                   # Classification training pipeline
├── train_yield_model.py                  # Regression training pipeline
├── predict.py                            # Standalone Python inference module & CLI
├── app.py                                # Full-featured Streamlit Web Application
├── requirements.txt                      # Project dependency specification
└── README.md                             # Academic project documentation
```

---

## ⚙️ Installation & Setup Instructions

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11 installed.
- PowerShell or Terminal.

### 2. Environment Setup
Navigate to the project directory:
```bash
cd C:\Users\Yashwin\.gemini\antigravity\scratch\crop_recommendation_and_yield_prediction
```

(Optional but recommended) Create a virtual environment:
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Training Pipeline

### Step 1: Generate Agronomic Datasets
```bash
python generate_data.py
```

### Step 2: Train Crop Recommendation Model (Classification)
```bash
python train_crop_model.py
```
*Outputs*: Prints comparative benchmark table, generates confusion matrix, classification report, feature importances, and exports `models/crop_recommendation_model.joblib`.

### Step 3: Train Crop Yield Prediction Model (Regression)
```bash
python train_yield_model.py
```
*Outputs*: Prints regression metrics ($R^2$, RMSE, MAE), produces actual vs predicted diagnostic charts, and exports `models/yield_prediction_model.joblib`.

### Step 4: Run Standalone CLI Prediction
```bash
python predict.py
```

---

## 🌐 Streamlit Web Application Guide

Launch the interactive web user interface:
```bash
streamlit run app.py
```

Once started, open your web browser at `http://localhost:8501`.

### Platform Modules:
1. **🌱 End-to-End Precision Predictor**:
   - Enter soil NPK, pH, Temperature, Humidity, Rainfall, and Farm Area (or select a quick regional preset).
   - Generates recommended crop, confidence %, yield per hectare, total harvest in tonnes, quintals, and kg.
2. **🌾 Crop Recommender**: Standalone classification testbench with top-5 crop alternatives.
3. **📈 Yield & Production Estimator**: Select any specific crop and simulate farm output.
4. **📊 Analytics & Model Benchmarks**: Inspect interactive tables, confusion matrix, feature importance, and EDA plots.
5. **📚 Academic & ML Documentation**: Theoretical algorithms guide and mathematical metric references.

---

## 🧪 Expected Outputs & Test Scenarios

### Test Scenario A: High Rainfall Tropical Environment
- **Inputs**: $N=85, P=48, K=40, \text{Temp}=24^\circ\text{C}, \text{Humidity}=84\%, \text{pH}=6.4, \text{Rainfall}=240\text{ mm}, \text{Area}=3.0\text{ ha}$
- **Expected Recommended Crop**: `RICE` ($>98\%$ confidence)
- **Predicted Yield**: $\approx 4.15\text{ Tonnes/Ha}$
- **Estimated Total Production**: $\approx 12.45\text{ Tonnes (124.5 Quintals)}$

### Test Scenario B: Cold Temperate Orchard Environment
- **Inputs**: $N=20, P=130, K=200, \text{Temp}=22^\circ\text{C}, \text{Humidity}=92\%, \text{pH}=6.0, \text{Rainfall}=110\text{ mm}, \text{Area}=2.0\text{ ha}$
- **Expected Recommended Crop**: `APPLE` or `GRAPES`
- **Predicted Yield**: $\approx 15.2\text{ Tonnes/Ha}$
- **Estimated Total Production**: $\approx 30.4\text{ Tonnes}$

---

## 🎓 Academic Viva / Oral Examination Q&A

**Q1: Why is Random Forest generally superior for crop recommendation?**
> **A:** Random Forest builds multiple decorrelated decision trees using bagging and random feature subspaces. In agricultural datasets where soil nutrients (N, P, K) exhibit complex non-linear thresholds and collinearity, ensemble averaging eliminates high variance and prevents overfitting.

**Q2: What is the significance of the $R^2$ score in yield regression?**
> **A:** The coefficient of determination ($R^2$) measures the proportion of variance in crop yield explained by soil and climatic features. An $R^2$ value of $0.95$ indicates that 95% of yield variance is captured by the model.

**Q3: How does the system handle categorical crop names during yield regression?**
> **A:** We utilize a scikit-learn `ColumnTransformer` with `OneHotEncoder(handle_unknown='ignore')` to convert nominal crop names into binary indicator columns, while standardizing numerical features using `StandardScaler`.

**Q4: How does this project support United Nations Sustainable Development Goals (SDGs)?**
> **A:** It directly contributes to **SDG 2 (Zero Hunger)** and **SDG 12 (Responsible Consumption and Production)** by reducing crop failure risks, preventing excessive fertilizer application, and optimizing agricultural land yields.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

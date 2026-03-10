# 🩺 Diabetes Health Indicators — Machine Learning Project

> A comprehensive machine learning pipeline for diabetes risk prediction and staging using clinical, lifestyle, and demographic health indicators.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [ML Tasks](#ml-tasks)
- [Modules](#modules)
- [Visualizations](#visualizations)
- [Streamlit App](#streamlit-app)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

This project builds an end-to-end machine learning solution to predict and classify diabetes risk using a rich set of health indicators. It tackles **three distinct ML problems** simultaneously:

| Task | Type | Target Variable |
|------|------|-----------------|
| Diabetes Diagnosis | Binary Classification | `diagnosed_diabetes` (0 / 1) |
| Diabetes Staging | Multiclass Classification | `diabetes_stage` (5 classes) |
| Risk Score Prediction | Regression | `diabetes_risk_score` (continuous) |

The project follows a clean, modular architecture — separating data preprocessing, model evaluation, and deployment into dedicated components — and is paired with an interactive **Streamlit web application** for real-time inference.

---

## 📁 Project Structure

```
diabetes-health-indicators-ml/
│
├── data/
│   ├── raw/
│   │   └── diabetes_dataset.csv          # Original raw dataset
│   └── processed/
│       ├── diabetes_processed.csv        # Cleaned & encoded dataset
│       └── label_encoders.joblib         # Fitted LabelEncoder objects
│
├── models/
│   ├── binary_model.joblib               # Tuned binary classification model (LR, C=10)
│   ├── multiclass_model.joblib           # Trained multiclass model (Decision Tree)
│   └── regression_model.joblib           # Trained regression model (Linear Regression)
│
├── notebooks/
│   ├── main_analysis.ipynb               # Full EDA, training & evaluation notebook
│   ├── binary_confusion_matrices.png
│   ├── binary_roc_curves.png
│   ├── boxplots.png
│   ├── class_balance.png
│   ├── feature_distributions.png
│   ├── multiclass_confusion_matrices.png
│   ├── regression_actual_vs_predicted.png
│   ├── risk_score_distribution.png
│   └── top_correlations.png
│
├── src/
│   ├── preprocessing.py                  # Data cleaning & feature engineering
│   └── evaluation.py                     # Metrics, plots & model comparison
│
├── streamlit_app/
│   └── app.py                            # Interactive prediction web app
│
├── requirements.txt
└── README.md
```

> **Note:** `regression_model.joblib` is saved by the notebook but was missing from the original README's project structure — it has been added above.

---

## 📊 Dataset

### Source
The dataset (`data/raw/diabetes_dataset.csv`) contains synthetic but clinically realistic patient records covering demographic, lifestyle, and biochemical features.

### Size
**100,000 rows × 31 columns** — no missing values.

### Features

**Demographic**
| Feature | Description |
|---------|-------------|
| `age` | Patient age in years (18–90) |
| `gender` | Gender (Female / Male) |
| `ethnicity` | Ethnicity (Asian / Black / Hispanic / Other / White) |
| `education_level` | Highest education attained |
| `income_level` | Income bracket (Low / Lower-Middle / Middle / Upper-Middle / High) |
| `employment_status` | Employment type (Employed / Unemployed / Retired / Self-Employed) |

**Lifestyle**
| Feature | Description |
|---------|-------------|
| `smoking_status` | Smoking behaviour (Never / Former / Current) |
| `alcohol_consumption_per_week` | Weekly alcohol units (0–10) |
| `physical_activity_minutes_per_week` | Exercise minutes/week (0–833) |
| `diet_score` | Diet quality score (0–10) |
| `sleep_hours_per_day` | Average sleep duration (3–10 hrs) |
| `screen_time_hours_per_day` | Daily screen time (0.5–16.8 hrs) |

**Clinical / Medical History**
| Feature | Description |
|---------|-------------|
| `family_history_diabetes` | Binary family history flag (0/1) — 22% positive |
| `hypertension_history` | Binary hypertension flag (0/1) — 25% positive |
| `cardiovascular_history` | Binary cardiovascular disease flag (0/1) — 8% positive |
| `bmi` | Body Mass Index (15.0–39.2) |
| `waist_to_hip_ratio` | Waist-to-hip ratio (0.67–1.06) |
| `systolic_bp` | Systolic blood pressure (90–179 mmHg) |
| `diastolic_bp` | Diastolic blood pressure (50–110 mmHg) |
| `heart_rate` | Resting heart rate (40–105 bpm) |

**Lab Results**
| Feature | Description |
|---------|-------------|
| `cholesterol_total` | Total cholesterol (100–318 mg/dL) |
| `hdl_cholesterol` | HDL cholesterol (20–98 mg/dL) |
| `ldl_cholesterol` | LDL cholesterol (50–263 mg/dL) |
| `triglycerides` | Triglyceride level (30–344 mg/dL) |
| `glucose_fasting` | Fasting blood glucose (60–172 mg/dL) |
| `glucose_postprandial` | Post-meal blood glucose (70–287 mg/dL) |
| `insulin_level` | Fasting insulin (2.0–32.2 µU/mL) |
| `hba1c` | Glycated haemoglobin (4.0–9.8%) |

**Target Variables**
| Feature | Type | Description |
|---------|------|-------------|
| `diagnosed_diabetes` | Binary | 0 = No Diabetes, 1 = Diagnosed |
| `diabetes_stage` | Multiclass | Gestational / No Diabetes / Pre-Diabetes / Type 1 / Type 2 |
| `diabetes_risk_score` | Continuous | Risk score ranging from 2.7 to 67.2 |

> ⚠️ **Important correction:** The original README described `diabetes_stage` as numeric classes 0–4 (No Risk → Diabetic). The actual dataset uses named string classes: **Gestational, No Diabetes, Pre-Diabetes, Type 1, Type 2**, which are label-encoded in preprocessing.

---

## 🤖 ML Tasks

### 1. Binary Classification — Diabetes Diagnosis
Predicts whether a patient has been diagnosed with diabetes (`0` or `1`).

- **Target:** `diagnosed_diabetes`
- **Class balance:** 59,998 positive (60%) / 40,002 negative (40%)
- **Models trained:** Logistic Regression, Decision Tree, KNN
- **Tuning:** GridSearchCV with 5-fold Stratified CV
- **Best model:** Tuned Logistic Regression (C=10) — saved to `models/binary_model.joblib`
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 2. Multiclass Classification — Diabetes Staging
Classifies a patient into one of five diabetes stages.

- **Target:** `diabetes_stage` (Gestational / No Diabetes / Pre-Diabetes / Type 1 / Type 2)
- **Models trained:** Decision Tree, Logistic Regression, KNN
- **Best model:** Decision Tree (Macro F1 = 0.514) — saved to `models/multiclass_model.joblib`
- **Evaluation:** Accuracy, Macro F1, per-class confusion matrix
- **Known limitation:** Gestational (278 samples) and Type 1 (122 samples) are severely underrepresented — minority class recall is poor. SMOTE or class weighting is recommended.

### 3. Regression — Risk Score Prediction
Predicts a continuous diabetes risk score.

- **Target:** `diabetes_risk_score` (range: 2.7–67.2, mean: 30.2)
- **Models trained:** Linear Regression, Decision Tree Regressor
- **Best model:** Linear Regression (R²=0.993) — saved to `models/regression_model.joblib`
- **Evaluation:** MAE, RMSE, R²

---

## 🧩 Modules

### `src/preprocessing.py`
Handles all data ingestion and preparation steps:
- Loads raw CSV data
- Median imputation for numerical columns; mode imputation for categoricals
- Encodes categorical variables using `LabelEncoder` (fitted encoders saved to `data/processed/label_encoders.joblib`)
- Applies `StandardScaler` — fitted on training data only to prevent leakage
- Outputs processed dataset to `data/processed/diabetes_processed.csv`

**Encoded categorical features:** `gender`, `ethnicity`, `education_level`, `income_level`, `employment_status`, `smoking_status`

---

### `src/evaluation.py`
Provides a complete suite of evaluation utilities:

```python
plot_roc_curve(model, X_test, y_test, title)
# → Plots ROC curve with AUC for binary classifiers

plot_confusion_matrix(model, X_test, y_test, labels, title)
# → Displays a colour-mapped confusion matrix

print_classification_metrics(model, X_test, y_test, task_name)
# → Prints full sklearn classification report

print_regression_metrics(model, X_test, y_test, task_name)
# → Prints MAE, RMSE, and R² scores

plot_feature_importance(model, feature_names, title, top_n)
# → Bar chart of top-N feature importances (tree or linear models)

compare_models(results_dict, metric)
# → Side-by-side bar chart comparing models on any metric
```

---

## 📈 Visualizations

All plots are generated in `notebooks/main_analysis.ipynb` and saved to the `notebooks/` directory.

| File | Description |
|------|-------------|
| `class_balance.png` | Class distribution for diagnosis and staging targets |
| `feature_distributions.png` | Histograms of 8 key clinical features |
| `boxplots.png` | Feature spread by diagnosis status (0 vs 1) |
| `top_correlations.png` | Top 12 feature correlations with `diagnosed_diabetes` |
| `risk_score_distribution.png` | Histogram of continuous risk score with mean marker |
| `binary_roc_curves.png` | ROC curves for all 3 binary classifiers |
| `binary_confusion_matrices.png` | Confusion matrices for binary classifiers |
| `multiclass_confusion_matrices.png` | Per-class confusion matrices for staging models |
| `regression_actual_vs_predicted.png` | Actual vs. predicted scatter for regression models |

---

## 🖥️ Streamlit App

An interactive prediction application is available in `streamlit_app/app.py`. It allows users to input patient health parameters and receive real-time predictions for:

- Diabetes diagnosis probability (binary)
- Diabetes stage classification (5-class)
- Continuous risk score

### Running the app

```bash
streamlit run streamlit_app/app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Clone the repository

```bash
git clone https://github.com/Bilalh429/diabetes-health-indicators-ml.git
cd diabetes-health-indicators-ml
```

### Install dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
jupyter
```

---

## 🚀 Usage

### 1. Run the Full Analysis Notebook

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

The notebook covers:
- Exploratory Data Analysis (EDA) — distributions, boxplots, correlation heatmap
- Feature correlation analysis (top predictor: HbA1c, r=0.679)
- Binary & multiclass classifier training and evaluation
- Linear regression for risk score prediction
- Hyperparameter tuning via GridSearchCV (5-fold Stratified CV)
- Saving best models with scalers and feature lists

### 2. Loading Saved Models

```python
import joblib

# Each bundle contains: model, scaler, and feature list
binary_bundle     = joblib.load("models/binary_model.joblib")
multiclass_bundle = joblib.load("models/multiclass_model.joblib")
regression_bundle = joblib.load("models/regression_model.joblib")

binary_model = binary_bundle["model"]
scaler       = binary_bundle["scaler"]
features     = binary_bundle["features"]

# Scale new data before predicting
X_scaled   = scaler.transform(X_new[features])
prediction = binary_model.predict(X_scaled)
```

### 3. Using Evaluation Utilities

```python
from src.evaluation import (
    plot_roc_curve,
    plot_confusion_matrix,
    print_classification_metrics,
    print_regression_metrics,
    compare_models
)

# Binary classification evaluation
plot_roc_curve(binary_model, X_test, y_binary_test, title="Diabetes Diagnosis ROC")
print_classification_metrics(binary_model, X_test, y_binary_test, task_name="Binary")

# Regression evaluation
print_regression_metrics(regression_model, X_test, y_risk_test, task_name="Risk Score")

# Compare models side by side
results = {
    "Logistic Regression": {"Accuracy": 0.8604},
    "Decision Tree":        {"Accuracy": 0.8613},
    "KNN":                  {"Accuracy": 0.8057},
}
compare_models(results, metric="Accuracy")
```

---

## 📉 Results

All results are from `notebooks/main_analysis.ipynb` with `random_state=42`, 80/20 stratified train/test split, and StandardScaler normalisation.

### Binary Classification — Diabetes Diagnosis

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression (baseline) | 0.8604 | 0.8850 | 0.9338 |
| Decision Tree (baseline) | 0.8613 | 0.8856 | 0.8530 |
| KNN (k=5) | 0.8057 | 0.8368 | 0.8726 |
| **LR (tuned, C=10)** ✓ | **0.8604** | **0.8850** | **0.9338** |
| **DT (tuned, depth=5)** ✓ | **0.9199** | **0.9285** | **0.9431** |

> ✅ **Saved model:** Tuned Logistic Regression (best ROC-AUC in CV). Tuned Decision Tree achieves the highest test accuracy (92.0%) and ROC-AUC (0.943).

### Multiclass Classification — Diabetes Staging

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **Decision Tree** ✓ | **0.8552** | **0.5141** |
| Logistic Regression | 0.8192 | 0.4658 |
| KNN (k=5) | 0.7557 | 0.4141 |

> ⚠️ Low Macro F1 is expected — Gestational (278) and Type 1 (122) classes are severely underrepresented. Accuracy is inflated by the dominant Type 2 class (59,774 samples).

### Regression — Risk Score Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Linear Regression** ✓ | **0.439** | **0.745** | **0.9933** |
| Decision Tree Regressor | 1.247 | 1.595 | 0.9692 |

> ✅ Linear Regression achieves near-perfect R²=0.993, suggesting the risk score is a linear combination of input features.

---

## 🗂️ Data Processing Pipeline

```
Raw CSV (100,000 rows × 31 columns)
   │
   ▼
Imputation (median for numeric, mode for categorical)
   │
   ▼
LabelEncoding (6 categorical columns + diabetes_stage)
   │
   ▼
Train / Test Split — 80/20 (stratified for classification)
   │
   ▼
StandardScaler (fit on train only)
   │
   ├──▶ Binary Classifier (LR / DT / KNN + GridSearchCV) ──▶ binary_model.joblib
   ├──▶ Multiclass Classifier (DT / LR / KNN)            ──▶ multiclass_model.joblib
   └──▶ Regression (Linear / Decision Tree)              ──▶ regression_model.joblib
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: description of change"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure new code follows the existing module structure.

---

## 📄 License

This project is open-source. See the repository for license details.

---

## 👤 Author

**Bilal H.**  
GitHub: [@Bilalh429](https://github.com/Bilalh429)

---

*Built with Python · scikit-learn · Streamlit · Matplotlib · Seaborn*
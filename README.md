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
| Diabetes Staging | Multiclass Classification | `diabetes_stage` (0 – 4) |
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
│   ├── binary_model.joblib               # Trained binary classification model
│   └── multiclass_model.joblib           # Trained multiclass classification model
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

---

## 📊 Dataset

### Source
The dataset (`data/raw/diabetes_dataset.csv`) contains synthetic but clinically realistic patient records covering demographic, lifestyle, and biochemical features.

### Features (31 columns)

**Demographic**
| Feature | Description |
|---------|-------------|
| `age` | Patient age in years |
| `gender` | Gender (Female / Male / Other) |
| `ethnicity` | Ethnicity (Asian / Black / Hispanic / Other / White) |
| `education_level` | Highest education attained |
| `income_level` | Income bracket (0–4) |
| `employment_status` | Employment type |

**Lifestyle**
| Feature | Description |
|---------|-------------|
| `smoking_status` | Smoking behaviour |
| `alcohol_consumption_per_week` | Weekly alcohol units |
| `physical_activity_minutes_per_week` | Exercise minutes/week |
| `diet_score` | Diet quality score |
| `sleep_hours_per_day` | Average sleep duration |
| `screen_time_hours_per_day` | Daily screen time |

**Clinical / Medical History**
| Feature | Description |
|---------|-------------|
| `family_history_diabetes` | Binary family history flag |
| `hypertension_history` | Binary hypertension flag |
| `cardiovascular_history` | Binary cardiovascular disease flag |
| `bmi` | Body Mass Index |
| `waist_to_hip_ratio` | Waist-to-hip ratio |
| `systolic_bp` | Systolic blood pressure (mmHg) |
| `diastolic_bp` | Diastolic blood pressure (mmHg) |
| `heart_rate` | Resting heart rate (bpm) |

**Lab Results**
| Feature | Description |
|---------|-------------|
| `cholesterol_total` | Total cholesterol (mg/dL) |
| `hdl_cholesterol` | HDL cholesterol (mg/dL) |
| `ldl_cholesterol` | LDL cholesterol (mg/dL) |
| `triglycerides` | Triglyceride level (mg/dL) |
| `glucose_fasting` | Fasting blood glucose (mg/dL) |
| `glucose_postprandial` | Post-meal blood glucose (mg/dL) |
| `insulin_level` | Fasting insulin (µU/mL) |
| `hba1c` | Glycated haemoglobin (%) |

**Target Variables**
| Feature | Description |
|---------|-------------|
| `diabetes_risk_score` | Continuous risk score (regression target) |
| `diabetes_stage` | Categorical staging: 0 = No Risk, 1 = Low, 2 = Moderate, 3 = High, 4 = Diabetic |
| `diagnosed_diabetes` | Binary diagnosis: 0 = No, 1 = Yes |

---

## 🤖 ML Tasks

### 1. Binary Classification — Diabetes Diagnosis
Predicts whether a patient has been diagnosed with diabetes (`0` or `1`).

- **Target:** `diagnosed_diabetes`
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Saved model:** `models/binary_model.joblib`

### 2. Multiclass Classification — Diabetes Staging
Classifies a patient into one of five diabetes risk stages (0–4).

- **Target:** `diabetes_stage`
- **Evaluation:** Classification report per class, Confusion Matrix
- **Saved model:** `models/multiclass_model.joblib`

### 3. Regression — Risk Score Prediction
Predicts a continuous diabetes risk score.

- **Target:** `diabetes_risk_score`
- **Evaluation:** MAE, MSE, RMSE, R²
- **Model:** `sklearn.linear_model.LinearRegression`

---

## 🧩 Modules

### `src/preprocessing.py`
Handles all data ingestion and preparation steps:
- Loads raw CSV data
- Encodes categorical variables using `LabelEncoder` (fitted encoders saved to `data/processed/label_encoders.joblib`)
- Feature engineering and cleaning
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
# → Prints MAE, MSE, RMSE, and R² scores

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
| `feature_distributions.png` | Histogram distributions of all numeric features |
| `boxplots.png` | Boxplots showing feature spread by diagnosis status |
| `top_correlations.png` | Top feature correlations with target variables |
| `class_balance.png` | Class distribution for staging and diagnosis targets |
| `risk_score_distribution.png` | Histogram of continuous risk score target |
| `binary_roc_curves.png` | ROC curves for binary classification models |
| `binary_confusion_matrices.png` | Confusion matrices for binary classifiers |
| `multiclass_confusion_matrices.png` | Per-class confusion matrices for staging models |
| `regression_actual_vs_predicted.png` | Actual vs. predicted scatter plot for regression |

---

## 🖥️ Streamlit App

An interactive prediction application is available in `streamlit_app/app.py`. It allows users to input patient health parameters and receive real-time predictions for:

- Diabetes diagnosis probability
- Diabetes stage classification
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

**Core dependencies include:**

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

### 1. Data Preprocessing

```python
from src.preprocessing import preprocess_data

X_train, X_test, y_train, y_test = preprocess_data("data/raw/diabetes_dataset.csv")
```

### 2. Training & Evaluation (Notebook)

Open and run the main analysis notebook:

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

The notebook walks through:
- Exploratory Data Analysis (EDA)
- Feature correlation analysis
- Training binary & multiclass classifiers
- Linear regression for risk score
- Full evaluation with plots

### 3. Loading Saved Models

```python
import joblib

binary_model = joblib.load("models/binary_model.joblib")
multiclass_model = joblib.load("models/multiclass_model.joblib")
label_encoders = joblib.load("data/processed/label_encoders.joblib")

# Predict
prediction = binary_model.predict(X_test)
```

### 4. Using Evaluation Utilities

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
    "Logistic Regression": {"Accuracy": 0.87},
    "Random Forest":        {"Accuracy": 0.91},
    "Gradient Boosting":    {"Accuracy": 0.93},
}
compare_models(results, metric="Accuracy")
```

---

## 📉 Results

> Results are generated from `notebooks/main_analysis.ipynb`. Below are representative performance indicators — exact figures depend on the train/test split and hyperparameter tuning performed during analysis.

### Binary Classification (Diabetes Diagnosis)
| Metric | Value |
|--------|-------|
| Accuracy | — |
| ROC-AUC | — |
| F1-Score | — |

### Multiclass Classification (Diabetes Staging)
| Metric | Value |
|--------|-------|
| Accuracy | — |
| Macro F1 | — |

### Regression (Risk Score)
| Metric | Value |
|--------|-------|
| MAE | — |
| RMSE | — |
| R² | — |

*Run `notebooks/main_analysis.ipynb` to populate exact results.*

---

## 🗂️ Data Processing Pipeline

```
Raw CSV
   │
   ▼
LabelEncoding (categorical columns)
   │
   ▼
Feature Selection / Engineering
   │
   ▼
Train / Test Split (80/20)
   │
   ├──▶ Binary Classifier  ──▶ binary_model.joblib
   ├──▶ Multiclass Classifier ──▶ multiclass_model.joblib
   └──▶ Linear Regression  ──▶ (inline prediction)
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: description of change"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure new code is covered by appropriate tests and follows the existing module structure.

---

## 📄 License

This project is open-source. See the repository for license details.

---

## 👤 Author

**Bilal H.**  
GitHub: [@Bilalh429](https://github.com/Bilalh429)

---

*Built with Python · scikit-learn · Streamlit · Matplotlib · Seaborn*
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

def inspect_data(df):
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Basic Stats ---")
    print(df.describe())
    return df

def preprocess(df, task='binary', scale=True):
    df = df.copy()
    df.dropna(inplace=True)

    # Encode categorical columns
    cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level',
                'employment_status', 'smoking_status']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Define targets
    target_cols = ['diagnosed_diabetes', 'diabetes_stage', 'diabetes_risk_score']
    feature_cols = [c for c in df.columns if c not in target_cols]

    X = df[feature_cols]

    if task == 'binary':
        y = df['diagnosed_diabetes']   # ← correct
        stratify = y
    elif task == 'multiclass':
        y = df['diabetes_stage']       # ← correct
        stratify = y
    else:
        y = df['diabetes_risk_score']  # ← correct
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
        X_test  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, f'models/scaler_{task}.joblib')

    print(f"\n[{task.upper()}] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Target distribution:\n{y_train.value_counts() if task != 'regression' else y_train.describe()}")
    return X_train, X_test, y_train, y_test, scaler, feature_cols
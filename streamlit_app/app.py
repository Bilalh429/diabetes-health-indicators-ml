"""
streamlit_app/app.py
====================
Diabetes Health Indicators — Interactive Prediction App

Tabs:
  1. Binary Classification  → diagnosed_diabetes (Yes / No)
  2. Multiclass Classification → diabetes_stage
  3. Regression             → diabetes_risk_score
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Health Indicator",
    page_icon="🩺",
    layout="wide",
)

MODELS_DIR = "models"

BINARY_PATH     = os.path.join(MODELS_DIR, "binary_model.joblib")
MULTI_PATH      = os.path.join(MODELS_DIR, "multiclass_model.joblib")
REGRESSION_PATH = os.path.join(MODELS_DIR, "regression_model.joblib")

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for key, path in [
        ("binary",     BINARY_PATH),
        ("multiclass", MULTI_PATH),
        ("regression", REGRESSION_PATH),
    ]:
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            models[key] = None
    return models


# ── Input widgets ─────────────────────────────────────────────────────────────
def render_input_form(prefix: str) -> dict:
    """Render patient input form and return a raw values dict."""
    st.markdown("#### 👤 Demographics")
    c1, c2, c3 = st.columns(3)
    age    = c1.slider("Age",    18, 100, 45, key=f"{prefix}_age")
    gender = c2.selectbox("Gender", ["Male", "Female", "Other"], key=f"{prefix}_gender")
    eth    = c3.selectbox("Ethnicity",
                          ["White", "Black", "Hispanic", "Asian", "Other"],
                          key=f"{prefix}_eth")

    c4, c5, c6 = st.columns(3)
    edu  = c4.selectbox("Education Level",
                        ["No Formal", "Highschool", "Some College", "Bachelor", "Graduate"],
                        key=f"{prefix}_edu")
    inc  = c5.selectbox("Income Level",
                        ["Low", "Lower-Middle", "Middle", "Upper-Middle", "High"],
                        key=f"{prefix}_inc")
    emp  = c6.selectbox("Employment Status",
                        ["Employed", "Unemployed", "Self-Employed", "Retired", "Student"],
                        key=f"{prefix}_emp")

    st.markdown("#### 🏃 Lifestyle")
    l1, l2, l3 = st.columns(3)
    smoke  = l1.selectbox("Smoking Status", ["Never", "Former", "Current"],
                          key=f"{prefix}_smoke")
    alcohol = l2.number_input("Alcohol (drinks/week)", 0, 50, 2, key=f"{prefix}_alc")
    activity = l3.number_input("Physical Activity (min/week)", 0, 1000, 150,
                               key=f"{prefix}_act")

    l4, l5, l6 = st.columns(3)
    diet        = l4.slider("Diet Score (0–10)", 0.0, 10.0, 5.0, step=0.1, key=f"{prefix}_diet")
    sleep       = l5.slider("Sleep Hours/Day", 3.0, 12.0, 7.0, step=0.5, key=f"{prefix}_sleep")
    screen_time = l6.slider("Screen Time Hours/Day", 0.0, 16.0, 5.0, step=0.5,
                            key=f"{prefix}_screen")

    st.markdown("#### 🏥 Medical History")
    m1, m2, m3 = st.columns(3)
    fam_hist   = m1.checkbox("Family History of Diabetes",    key=f"{prefix}_fam")
    hyp_hist   = m2.checkbox("Hypertension History",          key=f"{prefix}_hyp")
    cardio_hist = m3.checkbox("Cardiovascular History",       key=f"{prefix}_cardio")

    st.markdown("#### 📐 Physical Measurements")
    p1, p2, p3 = st.columns(3)
    bmi       = p1.number_input("BMI", 10.0, 60.0, 25.0, step=0.1, key=f"{prefix}_bmi")
    whr       = p2.number_input("Waist-to-Hip Ratio", 0.5, 1.5, 0.85, step=0.01,
                                key=f"{prefix}_whr")
    sys_bp    = p3.number_input("Systolic BP (mmHg)", 80, 200, 120, key=f"{prefix}_sbp")

    p4, p5, p6 = st.columns(3)
    dia_bp    = p4.number_input("Diastolic BP (mmHg)", 40, 130, 80, key=f"{prefix}_dbp")
    hr        = p5.number_input("Heart Rate (bpm)",    40, 180, 75, key=f"{prefix}_hr")

    st.markdown("#### 🔬 Lab Results")
    b1, b2, b3 = st.columns(3)
    chol_total = b1.number_input("Total Cholesterol (mg/dL)", 50, 400, 180, key=f"{prefix}_chol")
    hdl        = b2.number_input("HDL Cholesterol (mg/dL)",   10, 150, 55,  key=f"{prefix}_hdl")
    ldl        = b3.number_input("LDL Cholesterol (mg/dL)",   10, 300, 100, key=f"{prefix}_ldl")

    b4, b5, b6 = st.columns(3)
    trig       = b4.number_input("Triglycerides (mg/dL)", 20, 600, 120, key=f"{prefix}_trig")
    gluc_fast  = b5.number_input("Fasting Glucose (mg/dL)", 50, 400, 95, key=f"{prefix}_gf")
    gluc_post  = b6.number_input("Postprandial Glucose (mg/dL)", 50, 500, 130,
                                 key=f"{prefix}_gp")

    b7, b8 = st.columns(2)
    insulin = b7.number_input("Insulin Level (µU/mL)", 1.0, 200.0, 10.0, step=0.1,
                              key=f"{prefix}_ins")
    hba1c   = b8.number_input("HbA1c (%)", 3.0, 15.0, 5.5, step=0.1, key=f"{prefix}_hba1c")

    return dict(
        age=age, gender=gender, ethnicity=eth,
        education_level=edu, income_level=inc, employment_status=emp,
        smoking_status=smoke, alcohol_consumption_per_week=alcohol,
        physical_activity_minutes_per_week=activity, diet_score=diet,
        sleep_hours_per_day=sleep, screen_time_hours_per_day=screen_time,
        family_history_diabetes=int(fam_hist),
        hypertension_history=int(hyp_hist),
        cardiovascular_history=int(cardio_hist),
        bmi=bmi, waist_to_hip_ratio=whr,
        systolic_bp=sys_bp, diastolic_bp=dia_bp, heart_rate=hr,
        cholesterol_total=chol_total, hdl_cholesterol=hdl,
        ldl_cholesterol=ldl, triglycerides=trig,
        glucose_fasting=gluc_fast, glucose_postprandial=gluc_post,
        insulin_level=insulin, hba1c=hba1c,
    )


# ── Encode & align ────────────────────────────────────────────────────────────
GENDER_MAP     = {"Male": 1, "Female": 0, "Other": 2}
ETH_MAP        = {"Asian": 0, "Black": 1, "Hispanic": 2, "Other": 3, "White": 4}
EDU_MAP        = {"No Formal": 0, "Highschool": 1, "Some College": 2, "Bachelor": 3, "Graduate": 4}
INC_MAP        = {"Low": 0, "Lower-Middle": 1, "Middle": 2, "Upper-Middle": 3, "High": 4}
EMP_MAP        = {"Employed": 0, "Retired": 1, "Self-Employed": 2, "Student": 3, "Unemployed": 4}
SMOKE_MAP      = {"Current": 0, "Former": 1, "Never": 2}


def encode_inputs(raw: dict, features: list) -> np.ndarray:
    """Encode categoricals and align to model's feature order."""
    encoded = raw.copy()
    encoded["gender"]           = GENDER_MAP.get(raw["gender"], 0)
    encoded["ethnicity"]        = ETH_MAP.get(raw["ethnicity"], 0)
    encoded["education_level"]  = EDU_MAP.get(raw["education_level"], 0)
    encoded["income_level"]     = INC_MAP.get(raw["income_level"], 0)
    encoded["employment_status"]= EMP_MAP.get(raw["employment_status"], 0)
    encoded["smoking_status"]   = SMOKE_MAP.get(raw["smoking_status"], 0)

    row = pd.DataFrame([encoded])
    # Keep only features the model knows; fill any missing with 0
    for col in features:
        if col not in row.columns:
            row[col] = 0
    return row[features].values


# ── Tabs ──────────────────────────────────────────────────────────────────────
def main():
    st.title("🩺 Diabetes Health Indicator — ML Prediction App")
    st.markdown(
        "Enter patient details to get predictions across three model types. "
        "This tool is for **educational purposes only** and is not a medical diagnostic device."
    )

    models = load_models()
    tab1, tab2, tab3 = st.tabs([
        "🔵 Binary Classification",
        "🟢 Multiclass Classification",
        "🟠 Regression",
    ])

    # ── Tab 1: Binary ─────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Binary Classification — Diagnosed Diabetes (Yes / No)")
        if models["binary"] is None:
            st.warning("⚠️ Binary model not found. Run `python src/train_binary.py` first.")
        else:
            raw = render_input_form("bin")
            if st.button("🔮 Predict Diabetes", key="btn_bin", use_container_width=True):
                bundle  = models["binary"]
                X_input = encode_inputs(raw, bundle["features"])
                X_scaled = bundle["scaler"].transform(X_input)
                pred     = bundle["model"].predict(X_scaled)[0]
                proba    = bundle["model"].predict_proba(X_scaled)[0] \
                           if hasattr(bundle["model"], "predict_proba") else None

                st.divider()
                if pred == 1:
                    st.error("🔴 Prediction: **DIABETIC**")
                else:
                    st.success("🟢 Prediction: **NOT DIABETIC**")

                if proba is not None:
                    col_a, col_b = st.columns(2)
                    col_a.metric("Probability — No Diabetes", f"{proba[0]*100:.1f}%")
                    col_b.metric("Probability — Diabetic",    f"{proba[1]*100:.1f}%")

    # ── Tab 2: Multiclass ─────────────────────────────────────────────────────
    with tab2:
        st.subheader("Multiclass Classification — Diabetes Stage")
        if models["multiclass"] is None:
            st.warning("⚠️ Multiclass model not found. Run `python src/train_multiclass.py` first.")
        else:
            raw = render_input_form("multi")
            if st.button("🔮 Predict Stage", key="btn_multi", use_container_width=True):
                bundle   = models["multiclass"]
                X_input  = encode_inputs(raw, bundle["features"])
                X_scaled = bundle["scaler"].transform(X_input)
                pred_idx = bundle["model"].predict(X_scaled)[0]
                classes  = bundle.get("classes", ["No Diabetes", "Pre-Diabetes", "Type 2"])
                pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

                st.divider()
                color_map = {
                    "No Diabetes":  ("success", "🟢"),
                    "Pre-Diabetes": ("warning", "🟡"),
                    "Type 2":       ("error",   "🔴"),
                }
                style, icon = color_map.get(pred_label, ("info", "🔵"))
                getattr(st, style)(f"{icon} Predicted Stage: **{pred_label}**")

                if hasattr(bundle["model"], "predict_proba"):
                    proba = bundle["model"].predict_proba(X_scaled)[0]
                    st.markdown("##### Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Stage":       classes,
                        "Probability": [f"{p*100:.1f}%" for p in proba],
                    })
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Regression ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("Regression — Diabetes Risk Score (continuous)")
        st.caption("Risk score is a continuous index; higher = greater risk.")
        if models["regression"] is None:
            st.warning("⚠️ Regression model not found. Run `python src/train_regression.py` first.")
        else:
            raw = render_input_form("reg")
            if st.button("🔮 Predict Risk Score", key="btn_reg", use_container_width=True):
                bundle   = models["regression"]
                X_input  = encode_inputs(raw, bundle["features"])
                X_scaled = bundle["scaler"].transform(X_input)
                score    = bundle["model"].predict(X_scaled)[0]

                st.divider()
                st.metric("🎯 Predicted Diabetes Risk Score", f"{score:.2f}")

                # Risk band
                if score < 20:
                    st.success("🟢 Risk Band: **Low**")
                elif score < 40:
                    st.warning("🟡 Risk Band: **Moderate**")
                else:
                    st.error("🔴 Risk Band: **High** — Please consult a healthcare professional")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ℹ️ About")
        st.markdown(
            "This app uses trained ML models to predict diabetes status from "
            "clinical and lifestyle features.\n\n"
            "**Models trained on:** 100,000 patient records\n\n"
            "**Tasks covered:**\n"
            "- Binary classification\n"
            "- Multiclass classification\n"
            "- Risk score regression\n\n"
            "---\n"
            "⚠️ *Not a substitute for medical advice.*"
        )

        st.markdown("## 🔧 Model Status")
        for key, label in [("binary", "Binary"), ("multiclass", "Multiclass"), ("regression", "Regression")]:
            status = "✅ Loaded" if models[key] else "❌ Not found"
            st.markdown(f"**{label}:** {status}")


if __name__ == "__main__":
    main()

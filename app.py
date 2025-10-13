import streamlit as st
import numpy as np
import joblib
import pandas as pd
import json, os
from src.data import FEATURES

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")

st.title("🩺 Diabetes Prediction (PIMA)")
st.write("Loads `artifacts/best_model.pkl` and decision threshold from `artifacts/test_metrics.json`.")

@st.cache_resource
def load_model():
    return joblib.load("artifacts/best_model.pkl")

@st.cache_resource                        
def load_threshold(default: float = 0.5) -> float:
    path = "artifacts/test_metrics.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # u tvom fajlu je na vrhu "threshold"
        return float(data.get("threshold", default))
    return default

with st.sidebar:
    st.header("Input features")
    vals = {}
    for f in FEATURES:
        if f in {"Pregnancies","Age"}:
            vals[f] = st.number_input(f, min_value=0.0, step=1.0, value=0.0)
        elif f in {"Glucose","BloodPressure","SkinThickness","Insulin"}:
            vals[f] = st.number_input(f, min_value=0.0, step=1.0, value=0.0)
        elif f in {"BMI","DiabetesPedigreeFunction"}:
            vals[f] = st.number_input(f, min_value=0.0, step=0.1, value=0.0)
    use_trained_thr = st.checkbox("Use trained decision threshold", value=True) 
    go = st.button("Predict")

if go:
    model = load_model()
    thr = load_threshold() if use_trained_thr else 0.5 
    X = pd.DataFrame([vals])[FEATURES]
    prob = float(model.predict_proba(X)[:,1][0])
    pred = int(prob >= thr)
    st.subheader("Result")
    st.metric("Probability of diabetes", f"{prob:.3f}")
    st.write(f"Prediction: **{pred}** (1 = positive)")
    if not use_trained_thr:
        st.info("Using default threshold 0.5; you can switch on the trained threshold in the sidebar.")
        
        
# Outlieri: IQR → winsorizacija ili RobustScaler gpt
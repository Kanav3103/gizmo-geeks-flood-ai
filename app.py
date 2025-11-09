#https://gizmo-geeks-flood-ai-pew2g7fgsivnjbznvnnrmc.streamlit.app

# Link, for backend, KANAV ONLY : https://github.com/Kanav3103/gizmo-geeks-flood-ai/blob/main/app.py

# ==============================
# 🌊 Flood Prediction AI Dashboard (Real Data + Random Forest)
# ==============================
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import folium
from streamlit_folium import st_folium
import os
from sklearn.calibration import CalibratedClassifierCV

# =====================================
# 🔸 LOAD REAL DATA & TRAIN MODEL
# =====================================
@st.cache_resource
def load_and_train_model():
    if not os.path.exists("flood_risk_dataset_india.csv"):
        st.error("❌ Dataset 'flood_risk_dataset_india.csv' not found in project directory.")
        return None

    df = pd.read_csv("flood_risk_dataset_india.csv")
    df.columns = df.columns.str.strip()  # remove hidden spaces

    # Use your real columns
    X = df[["Rainfall (mm)", "Temperature (°C)", "Humidity (%)"]]
    y = df["Flood Occurred"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

   # Calibrate probabilities for better risk prediction
    calibrated_model = CalibratedClassifierCV(estimator=model, cv=5)
    calibrated_model.fit(X_train, y_train)

    # Evaluate
    y_pred = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]

    print("✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
    print("🌊 ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))

    # Save and return calibrated model
    joblib.dump(calibrated_model, "flood_model.pk

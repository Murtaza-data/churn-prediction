"""
Streamlit frontend for the Customer Churn Prediction API.

Collects raw Telco customer details, calls the FastAPI /predict endpoint,
and shows the churn verdict, probability, and risk level.

Set the backend URL via the API_URL env var / Streamlit secret, or edit below.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Prediction")
st.caption("Tuned XGBoost · Telco dataset · decision threshold 0.55")

with st.sidebar:
    st.header("About")
    st.write(
        "Predicts whether a telecom customer is likely to **churn**. "
        "The model optimises recall / ROC-AUC (not raw accuracy), since "
        "accuracy is misleading on imbalanced churn data."
    )
    st.metric("Test ROC-AUC", "0.848")
    st.metric("Test Recall", "0.687")
    st.divider()
    api_url = st.text_input("Backend API URL", value=API_URL)

st.subheader("Customer details")

col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
with col2:
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
with col3:
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

c1, c2 = st.columns(2)
with c1:
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
with c2:
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=840.0, step=10.0)

payload = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

if st.button("Predict churn", type="primary", use_container_width=True):
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        prob = result["churn_probability"]
        risk = result["risk"]

        if result["churn"]:
            st.error(f"⚠️ Likely to CHURN — risk: {risk}")
        else:
            st.success(f"✅ Likely to STAY — risk: {risk}")

        st.progress(min(prob, 1.0), text=f"Churn probability: {prob:.1%}")
        st.caption(f"Decision threshold: {result['threshold']:.2f}")
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not reach the API at {api_url}. Details: {e}")

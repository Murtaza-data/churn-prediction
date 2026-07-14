"""
Streamlit frontend for the Customer Churn Prediction API.
Draws the 19 input fields, POSTs them to the FastAPI /predict endpoint,
and shows the churn verdict + probability.

Set API_URL (env var / Streamlit secret) to the deployed Render backend URL.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.title("📉 Customer Churn Prediction")
st.caption("Tuned XGBoost · Telco dataset · threshold 0.55")

api_url = st.sidebar.text_input("API URL", value=API_URL)

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
with col2:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, value=70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, value=840.0)

payload = {
    "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
    "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
    "MultipleLines": MultipleLines, "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
    "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
    "Contract": Contract, "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

if st.button("Predict churn", type="primary"):
    try:
        r = requests.post(f"{api_url}/predict", json=payload, timeout=30)
        r.raise_for_status()
        result = r.json()
        prob = result["churn_probability"]
        if result["churn"]:
            st.error(f"⚠️ Likely to CHURN — probability {prob:.1%}")
        else:
            st.success(f"✅ Likely to STAY — probability {prob:.1%}")
        st.progress(min(prob, 1.0))
        st.caption(f"Decision threshold: {result['threshold']}")
    except Exception as e:
        st.warning(f"Could not reach the API at {api_url}. Details: {e}")

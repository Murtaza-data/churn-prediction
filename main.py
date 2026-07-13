"""
FastAPI backend for the Telco Customer Churn model.

Endpoints:
    GET  /            -> service info
    GET  /health      -> {"status": "ok"}
    POST /predict     -> churn prediction for one customer

Model: tuned XGBoost (best_xgb2), threshold 0.55, no scaler.
"""

import os
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from preprocessing import preprocess

THRESHOLD = 0.55
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a Telco customer will churn. Tuned XGBoost, threshold 0.55.",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    """Raw Telco customer fields (pre-engineering)."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.7,
                "TotalCharges": 257.1,
            }
        }
    }


class PredictionOutput(BaseModel):
    churn: bool
    churn_probability: float
    threshold: float
    risk: Literal["Low", "Medium", "High"]


@app.get("/")
def root():
    return {
        "service": "Customer Churn Prediction API",
        "model": "Tuned XGBoost (best_xgb2)",
        "threshold": THRESHOLD,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    row = preprocess(customer.model_dump(), feature_columns)
    proba = float(model.predict_proba(row)[0][1])
    churn = proba >= THRESHOLD

    if proba < 0.35:
        risk = "Low"
    elif proba < THRESHOLD:
        risk = "Medium"
    else:
        risk = "High"

    return PredictionOutput(
        churn=churn,
        churn_probability=round(proba, 4),
        threshold=THRESHOLD,
        risk=risk,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

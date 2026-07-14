"""
FastAPI backend for the Telco Customer Churn model.

Same logic you built and tested in Colab, organized as a file:
    load model + checklist (once)  ->  /predict: preprocess (19->39) -> model -> threshold 0.55
"""

import os

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from preprocessing import preprocess

THRESHOLD = 0.55
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Loaded once, when the API starts up.
model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

app = FastAPI(title="Customer Churn Prediction API")


# The raw 19 fields a request must send.
class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"service": "Customer Churn Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerInput):
    row = preprocess(customer.model_dump(), feature_columns)   # 19 -> 39
    prob = float(model.predict_proba(row)[0][1])               # probability
    return {
        "churn": bool(prob >= THRESHOLD),                      # apply 0.55
        "churn_probability": round(prob, 4),
        "threshold": THRESHOLD,
    }


if __name__ == "__main__":
    import uvicorn

    # Render injects $PORT; default to 8000 locally.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

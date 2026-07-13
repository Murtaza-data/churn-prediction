"""
Preprocessing for the Telco Customer Churn model.

This MUST replicate the exact pipeline used to train the model in Colab:
    clean -> feature engineering -> one-hot encode -> align to feature_columns.

The final model is a tuned XGBoost (best_xgb2) trained on UNSCALED data,
so there is NO scaler. Decision threshold = 0.55.

NOTE: one-hot alignment is handled robustly by reindexing to the saved
feature_columns.pkl, so drop_first behaviour is reproduced automatically
regardless of which category a single request happens to contain.
"""

import pandas as pd

# The 8 add-on / phone services counted by `num_services`.
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Raw categorical columns that get one-hot encoded (same set as training).
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the 6 engineered features.

    >>> VERIFY THESE FORMULAS AGAINST THE COLAB NOTEBOOK CELL <<<
    Getting a formula wrong here silently shifts predictions. These are the
    standard/most-likely definitions given the saved column names; confirm the
    exact expressions from the notebook and adjust if needed.
    """
    df = df.copy()

    # Count of subscribed services (each counted when == "Yes").
    df["num_services"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

    # Interaction: tenure * monthly charge.
    df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]

    # Average monthly spend over the customer's lifetime (guard tenure == 0).
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # How current monthly charge compares to historical average spend.
    df["charges_ratio"] = df["MonthlyCharges"] / df["avg_monthly_spend"].replace(0, 1)

    # New-customer flag (first year).
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)

    # Tenure buckets (months) -> matches tenure_group_* dummy columns.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
    )

    return df


def preprocess(raw: dict, feature_columns: list) -> pd.DataFrame:
    """Turn one raw customer dict into a single model-ready feature row.

    Args:
        raw: raw Telco fields (see main.py CustomerInput schema).
        feature_columns: list loaded from feature_columns.pkl (exact order).

    Returns:
        A 1-row DataFrame with columns == feature_columns, ready for the model.
    """
    df = pd.DataFrame([raw])

    # --- clean ---
    # TotalCharges arrives as a number here, but guard against blanks/strings
    # exactly like the training clean step (blank -> 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)

    # --- feature engineering ---
    df = add_engineered_features(df)

    # --- one-hot encode ---
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # --- align to training columns (reproduces drop_first + fills missing) ---
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df

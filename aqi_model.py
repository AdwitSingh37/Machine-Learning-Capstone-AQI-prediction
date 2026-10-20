"""Shared, reproducible preprocessing and prediction for the AQI demo."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "aqi_pipeline.joblib"
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
DATE_FEATURES = ["Year", "Month", "Day", "DayOfWeek"]
NUMERIC_FEATURES = POLLUTANTS + DATE_FEATURES
FEATURES = ["City"] + NUMERIC_FEATURES
CATEGORIES = [(50, "Good", "🟢"), (100, "Satisfactory", "🟡"), (200, "Moderate", "🟠"), (300, "Poor", "🔴"), (400, "Very Poor", "🟣"), (float("inf"), "Severe", "🟤")]


class IQRCapping(BaseEstimator, TransformerMixin):
    """Learn the notebook's outlier bounds on training observations only."""

    def fit(self, X, y=None):
        q1, q3 = X.quantile(0.25), X.quantile(0.75)
        self.lower_ = q1 - 1.5 * (q3 - q1)
        self.upper_ = q3 + 1.5 * (q3 - q1)
        return self

    def transform(self, X):
        return X.clip(self.lower_, self.upper_, axis=1)


def prepare_features(frame):
    """Accept raw City/Date/pollutant columns; never accept the target as input."""
    missing = {"City", "Date", *POLLUTANTS} - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    result = frame.copy()
    result["City"] = result["City"].astype("string").str.strip()
    if result["City"].isna().any() or result["City"].eq("").any():
        raise ValueError("Every row needs a city.")
    result["City"] = result["City"].astype(str)
    dates = pd.to_datetime(result["Date"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("Every row needs a valid date (YYYY-MM-DD).")
    for name in POLLUTANTS:
        original = result[name].replace(r"^\s*$", np.nan, regex=True)
        values = pd.to_numeric(original, errors="coerce")
        if (original.notna() & values.isna()).any() or np.isinf(values).any() or values.lt(0).any():
            raise ValueError(f"{name} must contain non-negative, finite numbers or empty cells.")
        result[name] = values
    if result[POLLUTANTS].isna().all(axis=1).any():
        raise ValueError("Provide at least one pollutant measurement in each row.")
    result["Year"] = dates.dt.year
    result["Month"] = dates.dt.month
    result["Day"] = dates.dt.day
    result["DayOfWeek"] = dates.dt.dayofweek
    return result[FEATURES]


def category(value):
    for upper, label, icon in CATEGORIES:
        if value <= upper:
            return label, icon


def load_bundle():
    # Load only this trusted, locally generated artifact, never an uploaded pickle.
    return joblib.load(MODEL_PATH)


def predict(frame, bundle):
    features = prepare_features(frame)
    unknown = set(features["City"]) - set(bundle["cities"])
    if unknown:
        raise ValueError("Unknown cities: " + ", ".join(sorted(unknown)))
    estimates = np.maximum(0, bundle["pipeline"].predict(features))
    result = frame.copy()
    result["Predicted_AQI"] = np.round(estimates, 1)
    result["Predicted_Category"] = [category(float(v))[0] for v in estimates]
    return result

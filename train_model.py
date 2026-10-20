"""Train and export the deployment model: python train_model.py."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aqi_model import IQRCapping, MODEL_PATH, NUMERIC_FEATURES, POLLUTANTS, ROOT, prepare_features


def main():
    data_path = ROOT / "data" / "city_day.csv"
    data = pd.read_csv(data_path).dropna(subset=["AQI"])
    data = data.loc[data[POLLUTANTS].notna().any(axis=1)].copy()
    X, y = prepare_features(data), data["AQI"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numeric = Pipeline([
        ("outlier_capping", IQRCapping()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City"]),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessing),
        ("model", RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=2)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metadata = {
        "model": "Random Forest", "n_estimators": 100, "max_depth": 20,
        "min_samples_leaf": 2, "random_state": 42,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "r2": r2_score(y_test, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": mean_absolute_error(y_test, predictions),
        "date_min": str(data["Date"].min()), "date_max": str(data["Date"].max()),
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "evaluation": "Random 80:20 held-out split; preprocessing fitted only on training rows. This is not a future-date or unseen-city evaluation.",
    }
    bundle = {
        "pipeline": pipeline, "metadata": metadata,
        "cities": sorted(data["City"].unique().tolist()),
        "medians": X_train[POLLUTANTS].median().to_dict(),
        "ranges": {p: [float(X_train[p].min()), float(X_train[p].max())] for p in POLLUTANTS},
        "example": data.loc[data[POLLUTANTS].notna().all(axis=1)].iloc[0][["City", "Date"] + POLLUTANTS].to_dict(),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    (MODEL_PATH.parent / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()

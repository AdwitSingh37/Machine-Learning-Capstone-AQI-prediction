"""Streamlit interface for the capstone's trained AQI regression model."""
from datetime import date
import io

import pandas as pd
import streamlit as st

from aqi_model import CATEGORIES, MODEL_PATH, POLLUTANTS, category, load_bundle, predict

st.set_page_config(page_title="AirLens · AQI Predictor", page_icon="🌿", layout="wide")


@st.cache_resource
def get_model():
    return load_bundle()


if not MODEL_PATH.exists():
    st.error("The trained model is missing. Run `python train_model.py` before starting the app.")
    st.stop()

bundle = get_model()
meta = bundle["metadata"]

with st.sidebar:
    st.title("🌿 AirLens")
    st.caption("AIR QUALITY INTELLIGENCE")
    st.divider()
    st.markdown("**From measurements to insight.**")
    st.write("Estimate air quality from daily pollutant readings using a trained Random Forest model.")
    st.divider()
    st.markdown("**AQI category scale**")
    previous = 0
    for upper, label, icon in CATEGORIES:
        interval = f"{previous}–{int(upper)}" if upper != float("inf") else "401+"
        st.write(f"{icon} **{label}** · {interval}")
        if upper != float("inf"):
            previous = int(upper) + 1
    st.divider()
    st.caption("ML Capstone · Tharun G Nambiar, Adhitya Ranjit & Adwit Singh")

st.caption("ML CAPSTONE / INTERACTIVE DEMO")
st.title("A clearer picture of the air.")
st.write("Enter daily pollutant concentrations to estimate the Air Quality Index and its category.")
single, batch, about = st.tabs(["Predict AQI", "Batch predictions", "Model & evaluation"])

with single:
    inputs, output = st.columns([1.6, 1], gap="large")
    with inputs:
        st.subheader("Your measurements")
        use_example = st.toggle("Use a sample from the dataset", value=True)
        source = bundle["example"] if use_example else bundle["medians"]
        with st.form("prediction"):
            c1, c2 = st.columns(2)
            cities = bundle["cities"]
            city = c1.selectbox("City", cities, index=cities.index(source.get("City", "Delhi")))
            sample_date = date.fromisoformat(source.get("Date", meta["date_max"]))
            reading_date = c2.date_input("Measurement date", value=sample_date, min_value=date(2000, 1, 1), max_value=date(2100, 12, 31))
            st.caption("Concentrations in µg/m³, except CO in mg/m³. Prefilled values are a dataset sample or training medians; replace them with your readings.")
            columns = st.columns(3)
            values = {}
            for i, pollutant in enumerate(POLLUTANTS):
                unit = "mg/m³" if pollutant == "CO" else "µg/m³"
                values[pollutant] = columns[i % 3].number_input(f"{pollutant} ({unit})", min_value=0.0, value=float(source[pollutant]), step=0.1, format="%.2f")
            submitted = st.form_submit_button("Estimate air quality", type="primary", use_container_width=True)
        if submitted:
            row = {"City": city, "Date": reading_date.isoformat(), **values}
            result = predict(pd.DataFrame([row]), bundle)
            st.session_state["result"] = {"aqi": float(result.iloc[0]["Predicted_AQI"]), "category": result.iloc[0]["Predicted_Category"], "row": row}
    with output:
        st.subheader("Air quality estimate")
        if "result" in st.session_state:
            result = st.session_state["result"]
            row = result["row"]
            with st.container(border=True):
                st.caption(f"{row['City'].upper()} · {row['Date']}")
                st.metric("Predicted AQI", f"{result['aqi']:.1f}")
                icon = next(icon for _, label, icon in CATEGORIES if label == result["category"])
                st.subheader(f"{icon} {result['category']}")
                st.caption("Category is derived from the unrounded regression estimate using the scale shown at left.")
            st.caption("Result reflects the last submitted inputs. Submit again after editing readings.")
            if not meta["date_min"] <= row["Date"] <= meta["date_max"]:
                st.warning("This date is outside the training period. Reliability on future dates has not been measured.")
            outside = [p for p in POLLUTANTS if not bundle["ranges"][p][0] <= row[p] <= bundle["ranges"][p][1]]
            if outside:
                st.warning("Outside observed training ranges: " + ", ".join(outside) + ". Estimates may be less reliable.")
            st.download_button("Download this prediction", predict(pd.DataFrame([row]), bundle).to_csv(index=False), "aqi_prediction.csv", "text/csv")
        else:
            st.info("Submit your measurements to see the predicted AQI and category here.")
        st.caption(f"Held-out average absolute error: {meta['mae']:.1f} AQI points. This is a dataset average, not a confidence interval for this reading.")
        st.caption("Educational model estimate. This app does not provide live monitoring or an official AQI calculation.")

with batch:
    st.subheader("Predict for a collection of readings")
    st.write("Upload a CSV with City, Date and all 12 pollutant columns. Dates use YYYY-MM-DD. Empty pollutant cells use training medians; at least one measurement per row is required.")
    template = pd.DataFrame([bundle["example"]])
    st.download_button("Download example CSV", template.to_csv(index=False), "aqi_input_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload pollutant readings", type=["csv"])
    if uploaded is not None:
        try:
            if uploaded.size > 5 * 1024**2:
                raise ValueError("Please upload a CSV smaller than 5 MB.")
            frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), nrows=5001)
            if frame.empty or len(frame) > 5000:
                raise ValueError("Upload between 1 and 5,000 rows.")
            result = predict(frame, bundle)
            st.success(f"Predicted AQI for {len(result):,} readings.")
            dates = pd.to_datetime(frame["Date"], format="mixed")
            if ((dates < pd.Timestamp(meta["date_min"])) | (dates > pd.Timestamp(meta["date_max"]))).any():
                st.warning("Some dates fall outside the training period; reliability on these dates has not been measured.")
            st.dataframe(result, hide_index=True, use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False), "aqi_predictions.csv", "text/csv")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(str(error))

with about:
    st.subheader("A trained model, with measurable limits")
    a, b, c = st.columns(3)
    a.metric("Held-out R²", f"{meta['r2']:.3f}")
    b.metric("RMSE", f"{meta['rmse']:.1f}")
    c.metric("MAE", f"{meta['mae']:.1f}")
    st.write(f"Random Forest · 100 trees · {meta['train_rows']:,} training rows · {meta['test_rows']:,} test rows")
    st.write(f"Dataset: city_day.csv · {len(bundle['cities'])} cities · {meta['date_min']} to {meta['date_max']}")
    st.write("The deployment model follows the regression notebook's approach: date features, training-only IQR capping, median imputation, standard scaling and one-hot city encoding. It is trained separately with bounded tree depth and leaf size to keep hosting lightweight; these metrics belong to this exported model.")
    st.info(meta["evaluation"])
    st.write("Inputs exclude AQI and AQI_Bucket to prevent target leakage. The displayed category comes from the regression estimate; it is not the separate classification notebook's prediction.")
    st.write("The model estimates AQI for the supplied daily measurements. It does not forecast future pollutant concentrations. Historical data, missing inputs and extreme readings limit its reliability.")
    st.markdown("[Project source](https://github.com/AdwitSingh37/Machine-Learning-Capstone-AQI-prediction)")

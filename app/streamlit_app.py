import json
import sys
from pathlib import Path

# `streamlit run app/streamlit_app.py` only puts app/ on the path; the saved
# pipelines need `app.transformer` importable to load.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from app.config import MODELS_DIR
from app.extractor import DataExtractor

st.set_page_config(page_title="Tele-Cluster", layout="wide")


@st.cache_resource
def load_models():
    return (
        joblib.load(MODELS_DIR / "segmentation.joblib"),
        json.loads((MODELS_DIR / "segmentation.json").read_text()),
        joblib.load(MODELS_DIR / "churn.joblib"),
        json.loads((MODELS_DIR / "churn.json").read_text()),
    )


@st.cache_data
def load_customers():
    extractor = DataExtractor()
    extractor.load()
    return extractor.features_and_target()[0]


segmentation, segments, churn_model, churn_info = load_models()
customers = load_customers()

st.title("Tele-Cluster")
st.caption(f"Customer segment and churn risk - {churn_info['model']} "
           f"(test F1 {churn_info['test_metrics']['test_f1']:.2f}, "
           f"ROC-AUC {churn_info['test_metrics']['test_roc_auc']:.2f})")

with st.form("customer"):
    columns = st.columns(3)
    values = {}
    for i, col in enumerate(customers.columns):
        series = customers[col]
        with columns[i % 3]:
            if col == "SeniorCitizen":
                values[col] = st.selectbox(col, [0, 1])
            elif pd.api.types.is_numeric_dtype(series):
                values[col] = st.number_input(col, min_value=0.0, value=float(series.median()))
            else:
                values[col] = st.selectbox(col, sorted(series.dropna().unique()))
    submitted = st.form_submit_button("Predict")

if submitted:
    customer = pd.DataFrame([values])
    cluster = str(segmentation.predict(customer)[0])
    profile = segments["profiles"][cluster]
    proba = float(churn_model.predict_proba(customer)[0, 1])

    left, middle, right = st.columns(3)
    left.metric("Segment", f"Cluster {cluster}", profile["name"], delta_color="off")
    middle.metric("Prediction", "Will churn" if proba >= 0.5 else "Will stay")
    right.metric("Churn probability", f"{proba:.0%}")
    st.progress(proba)
    st.caption(f"Customers in this segment: {profile['size']}, historical churn rate {profile['churn_rate']:.0%}")

with st.expander("Segment profiles"):
    st.dataframe(pd.DataFrame(segments["profiles"]).T)

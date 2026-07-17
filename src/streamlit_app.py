"""
Streamlit UI for the loan-approval service.

Two pages (sidebar):
  * Make a prediction – single form OR batch CSV upload -> POST /predict.
  * Past predictions  – date range + source filter -> GET /past-predictions.

The webapp NEVER imports the model or touches the database directly: everything
goes through the API (API_URL env var). Results are shown as dataframes with the
features used for the prediction.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

FEATURE_COLS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

st.set_page_config(page_title="Loan Approval", page_icon="🏦", layout="wide")
page = st.sidebar.radio("Navigation", ["Make a prediction", "Past predictions"])


def call_predict(rows: list[dict]):
    """Send a batch of applications to the API and return the Response."""
    payload = {"source": "webapp", "applications": rows}
    return requests.post(f"{API_URL}/predict", json=payload, timeout=30)


def single_prediction():
    st.subheader("Single prediction")
    with st.form("loan_form"):
        c1, c2 = st.columns(2)
        with c1:
            no_of_dependents = st.number_input("Dependents", 0, 15, 2)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self employed", ["No", "Yes"])
            income_annum = st.number_input(
                "Annual income", 0, value=9_600_000, step=100_000
            )
            loan_amount = st.number_input(
                "Loan amount", 0, value=29_900_000, step=100_000
            )
            loan_term = st.number_input("Loan term (years)", 1, 40, 12)
        with c2:
            cibil_score = st.number_input("CIBIL score", 300, 900, 778)
            residential = st.number_input(
                "Residential assets", 0, value=2_400_000, step=100_000
            )
            commercial = st.number_input(
                "Commercial assets", 0, value=17_600_000, step=100_000
            )
            luxury = st.number_input("Luxury assets", 0, value=22_700_000, step=100_000)
            bank = st.number_input("Bank assets", 0, value=8_000_000, step=100_000)
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return
    row = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential,
        "commercial_assets_value": commercial,
        "luxury_assets_value": luxury,
        "bank_asset_value": bank,
    }
    try:
        resp = call_predict([row])
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return
    res = resp.json()["results"][0]
    out = pd.DataFrame([{**row, **res}])
    if res["prediction"] == "Approved":
        st.success(f"✅ Approved (probability {res['probability']:.2%})")
    else:
        st.error(f"❌ Rejected (approval probability {res['probability']:.2%})")
    st.dataframe(out, use_container_width=True)


def batch_prediction():
    st.subheader("Batch prediction (CSV upload)")
    st.caption("Upload a CSV with the feature columns (no label column).")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is None:
        return
    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV: {exc}")
        return

    keep = [c for c in FEATURE_COLS if c in df.columns]
    df = df[keep]
    st.write("Preview of uploaded data:")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Predict batch"):
        try:
            resp = call_predict(df.to_dict(orient="records"))
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            return
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return
        results = resp.json()["results"]
        out = df.copy()
        out["prediction"] = [r["prediction"] for r in results]
        out["probability"] = [r["probability"] for r in results]
        st.success(f"Predicted {len(results)} applications.")
        st.dataframe(out, use_container_width=True)


def prediction_page():
    st.title("🏦 Loan Approval — Prediction")
    tab_single, tab_batch = st.tabs(["Single", "Batch (CSV upload)"])
    with tab_single:
        single_prediction()
    with tab_batch:
        batch_prediction()


def past_predictions_page():
    st.title("📜 Past predictions")
    c1, c2, c3 = st.columns(3)
    start = c1.date_input("Start date", value=date.today() - timedelta(days=7))
    end = c2.date_input("End date", value=date.today())
    source = c3.selectbox("Source", ["all", "webapp", "scheduled"])

    params = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "limit": 1000,
    }
    if source != "all":
        params["source"] = source

    try:
        resp = requests.get(f"{API_URL}/past-predictions", params=params, timeout=30)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return

    rows = resp.json()
    if not rows:
        st.info("No predictions found for the selected filters.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.caption(f"{len(df)} rows")


if page == "Make a prediction":
    prediction_page()
else:
    past_predictions_page()

"""
Streamlit UI for the loan-approval service.

Two pages (sidebar):
  * Make a prediction – form -> POST /predict on the API.
  * Past predictions  – table from GET /predictions, with a source filter.

The API base URL comes from the API_URL environment variable.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Loan Approval", page_icon="🏦", layout="centered")
page = st.sidebar.radio("Navigation", ["Make a prediction", "Past predictions"])


def prediction_page() -> None:
    st.title("🏦 Loan Approval — Prediction")
    st.caption("Fill in the applicant details and get an instant decision.")

    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
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
        with col2:
            cibil_score = st.number_input("CIBIL score", 300, 900, 778)
            residential_assets_value = st.number_input(
                "Residential assets", 0, value=2_400_000, step=100_000
            )
            commercial_assets_value = st.number_input(
                "Commercial assets", 0, value=17_600_000, step=100_000
            )
            luxury_assets_value = st.number_input(
                "Luxury assets", 0, value=22_700_000, step=100_000
            )
            bank_asset_value = st.number_input(
                "Bank assets", 0, value=8_000_000, step=100_000
            )
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "source": "webapp",
            "applications": [
                {
                    "no_of_dependents": no_of_dependents,
                    "education": education,
                    "self_employed": self_employed,
                    "income_annum": income_annum,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "cibil_score": cibil_score,
                    "residential_assets_value": residential_assets_value,
                    "commercial_assets_value": commercial_assets_value,
                    "luxury_assets_value": luxury_assets_value,
                    "bank_asset_value": bank_asset_value,
                }
            ],
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            return
        if resp.status_code != 200:
            st.error(f"API error {resp.status_code}: {resp.text}")
            return
        result = resp.json()["results"][0]
        if result["prediction"] == "Approved":
            st.success(f"✅ Approved  (probability {result['probability']:.2%})")
        else:
            st.error(f"❌ Rejected  (approval probability {result['probability']:.2%})")


def past_predictions_page() -> None:
    st.title("📜 Past predictions")
    source = st.selectbox("Source", ["all", "webapp", "scheduled"])
    limit = st.slider("Rows", 10, 500, 50, step=10)

    params = {"limit": limit}
    if source != "all":
        params["source"] = source
    try:
        resp = requests.get(f"{API_URL}/predictions", params=params, timeout=15)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}: {resp.text}")
        return

    rows = resp.json()
    if not rows:
        st.info("No predictions yet.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.caption(f"{len(df)} rows")


if page == "Make a prediction":
    prediction_page()
else:
    past_predictions_page()

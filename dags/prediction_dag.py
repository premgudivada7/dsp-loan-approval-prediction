from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pendulum
import requests
from airflow.decorators import dag, task
from airflow.models import Variable

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
sys.path.append(str(PROJECT_ROOT))

GOOD_DATA = PROJECT_ROOT / "data" / "good_data"
API_URL = os.getenv("API_URL", "http://api:8000")
PROCESSED_VAR = "processed_prediction_files"

NUMERIC_COLS = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
INT_COLS = ["no_of_dependents", "loan_term", "cibil_score"]
FEATURE_COLS = ["education", "self_employed"] + NUMERIC_COLS


def clean_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows the API schema will accept (coerce numerics, drop bad rows)."""
    df = df.copy()
    df = df[[c for c in FEATURE_COLS if c in df.columns]]

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])

    df = df[df["education"].isin(["Graduate", "Not Graduate"])]
    df = df[df["self_employed"].isin(["Yes", "No"])]
    df = df[df["cibil_score"].between(300, 900)]
    df = df[df["no_of_dependents"].between(0, 15)]
    df = df[df["loan_term"].between(1, 40)]
    for c in ["income_annum", "loan_amount", "residential_assets_value",
              "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]:
        df = df[df[c] >= 0]

    for c in INT_COLS:
        df[c] = df[c].astype(int)
    return df


@dag(
    dag_id="loan_scheduled_prediction",
    schedule="*/5 * * * *",           # every 5 minutes
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["dsp", "prediction"],
)
def prediction_dag():

    @task.short_circuit
    def check_for_new_data() -> list[str]:
        processed = set(json.loads(Variable.get(PROCESSED_VAR, default_var="[]")))
        all_files = {str(p) for p in GOOD_DATA.glob("*.csv")}
        new_files = sorted(all_files - processed)
        if not new_files:
            print("No new data in good_data — skipping run.")
            return []          # falsy -> downstream skipped
        print(f"Found {len(new_files)} new file(s).")
        return new_files

    @task
    def make_predictions(new_files: list[str]) -> dict:
        """Read all new files, clean, and make ONE batch prediction API call."""
        frames = [pd.read_csv(fp) for fp in new_files]
        batch = clean_batch(pd.concat(frames, ignore_index=True))

        result = {"files": len(new_files), "rows_scored": 0}
        if not batch.empty:
            payload = {"source": "scheduled",
                       "applications": batch.to_dict(orient="records")}
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            result["rows_scored"] = len(resp.json()["results"])
        else:
            print("No scoreable rows after cleaning — nothing sent to the API.")

        # Mark files processed (do NOT move/delete them).
        processed = set(json.loads(Variable.get(PROCESSED_VAR, default_var="[]")))
        processed.update(new_files)
        Variable.set(PROCESSED_VAR, json.dumps(sorted(processed)))
        return result

    make_predictions(check_for_new_data())


prediction_dag()
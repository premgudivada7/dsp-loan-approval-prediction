from __future__ import annotations

import json
import os
import random
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pendulum
from airflow.decorators import dag, task

# Add the project source folder to the path.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
sys.path.append(str(PROJECT_ROOT))

from src.alerts import send_teams_alert            # noqa: E402
from src.gx_validation import validate_batch        # noqa: E402

RAW_DATA = PROJECT_ROOT / "data" / "raw_data"
GOOD_DATA = PROJECT_ROOT / "data" / "good_data"
BAD_DATA = PROJECT_ROOT / "data" / "bad_data"
POSTGRES_CONN_ID = "loan_postgres"


@dag(
    dag_id="loan_data_ingestion",
    schedule="*/5 * * * *",           # every 5 minutes
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["dsp", "ingestion"],
)
def ingestion_dag():

    @task
    def read_data() -> dict | None:
        """Pick one random raw file, read it, delete it to avoid reprocessing."""
        files = list(RAW_DATA.glob("*.csv"))
        if not files:
            return None
        chosen = random.choice(files)
        df = pd.read_csv(chosen)
        payload = {"file_path": str(chosen), "records": df.to_json(orient="split")}
        chosen.unlink()  # delete after reading
        return payload

    @task
    def validate_data(payload: dict | None) -> dict | None:
        """Validate with Great Expectations; attach the summary + data."""
        if not payload:
            return None
        df = pd.read_json(StringIO(payload["records"]), orient="split")
        summary = validate_batch(df, Path(payload["file_path"]).name)
        summary["records"] = payload["records"]
        summary["file_path"] = payload["file_path"]
        return summary

    @task
    def save_statistics(summary: dict | None) -> None:
        """Persist a summary row to ingestion_stats via PostgresHook."""
        if not summary:
            return
        from airflow.providers.postgres.hooks.postgres import PostgresHook
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        hook.run(
            """
            INSERT INTO ingestion_stats
                (ingested_at, file_name, total_rows, valid_rows, invalid_rows,
                 invalid_pct, criticality, error_summary, report_name)
            VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            parameters=(
                summary["file_name"], summary["total_rows"], summary["valid_rows"],
                summary["invalid_rows"], summary["invalid_pct"],
                summary["criticality"], json.dumps(summary["error_summary"]),
                summary["report_name"],
            ),
        )

    @task
    def send_alerts(summary: dict | None) -> None:
        """Send a Teams alert only for Medium/High criticality."""
        if not summary:
            return
        send_teams_alert(summary)

    @task
    def split_and_save_data(summary: dict | None) -> None:
        """Route rows to good_data / bad_data based on validation."""
        if not summary:
            return
        GOOD_DATA.mkdir(parents=True, exist_ok=True)
        BAD_DATA.mkdir(parents=True, exist_ok=True)
        df = pd.read_json(StringIO(summary["records"]), orient="split")
        name = summary["file_name"]
        invalid = set(summary["invalid_indices"])

        if summary["missing_column"] or summary["invalid_rows"] == summary["total_rows"]:
            df.to_csv(BAD_DATA / name, index=False)            # all bad
        elif summary["invalid_rows"] == 0:
            df.to_csv(GOOD_DATA / name, index=False)           # all good
        else:                                                   # split
            good = df.loc[~df.index.isin(invalid)]
            bad = df.loc[df.index.isin(invalid)]
            good.to_csv(GOOD_DATA / name, index=False)
            bad.to_csv(BAD_DATA / name, index=False)

    raw = read_data()
    summary = validate_data(raw)
# Run the remaining three tasks in parallel after data validation.
    save_statistics(summary)
    send_alerts(summary)
    split_and_save_data(summary)


ingestion_dag()
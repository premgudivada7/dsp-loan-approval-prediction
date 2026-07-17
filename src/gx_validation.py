"""
Validates the ingested loan data using Great Expectations.

It generates a validation summary that is used to save statistics,
send alerts, and separate valid and invalid records.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GX_ROOT = os.getenv("GX_ROOT_DIR", str(ROOT / "great_expectations"))

REQUIRED_COLUMNS = [
    "loan_id",
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
NON_NEGATIVE = [
    "income_annum",
    "loan_amount",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
SUITE_NAME = "loan_ingestion_suite"


def compute_criticality(missing_column: bool, invalid_pct: float) -> str:
    """High: missing column or >50% invalid. Medium: 10-50%. Low: >0 & <10%."""
    if missing_column or invalid_pct > 50:
        return "High"
    if invalid_pct >= 10:
        return "Medium"
    if invalid_pct > 0:
        return "Low"
    return "None"


def _build_suite(gx, gxe):
    """Create (or return) the reusable Expectation Suite."""
    suite = gx.ExpectationSuite(name=SUITE_NAME)
    for col in REQUIRED_COLUMNS:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))
    # Completeness
    for col in ["loan_id", "cibil_score", "education", "self_employed", "income_annum"]:
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))
    # Validity (ranges)
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="cibil_score", min_value=300, max_value=900
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(
            column="no_of_dependents", min_value=0, max_value=15
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="loan_term", min_value=1, max_value=40)
    )
    for col in NON_NEGATIVE:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column=col, min_value=0, max_value=1_000_000_000
            )
        )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="education", value_set=["Graduate", "Not Graduate"]
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="self_employed", value_set=["Yes", "No"])
    )
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="loan_id"))
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInTypeList(
            column="cibil_score", type_list=["INTEGER", "int64", "int", "float64"]
        )
    )
    return suite


def _gx_validate(df: pd.DataFrame, file_name: str) -> dict:
    import great_expectations as gx
    import great_expectations.expectations as gxe
    from great_expectations.checkpoint import UpdateDataDocsAction
    from great_expectations.core.result_format import ResultFormat

    context = gx.get_context(mode="file", project_root_dir=GX_ROOT)

    try:
        ds = context.data_sources.add_pandas("loan_pandas")
    except Exception:
        ds = context.data_sources.get("loan_pandas")
    try:
        asset = ds.add_dataframe_asset("ingested_batch")
    except Exception:
        asset = ds.get_asset("ingested_batch")
    try:
        batch_def = asset.add_batch_definition_whole_dataframe("whole_batch")
    except Exception:
        batch_def = asset.get_batch_definition("whole_batch")

    try:
        suite = context.suites.add(_build_suite(gx, gxe))
    except Exception:
        context.suites.delete(name=SUITE_NAME)
        suite = context.suites.add(_build_suite(gx, gxe))

    try:
        vdef = context.validation_definitions.add(
            gx.ValidationDefinition(name="loan_vdef", data=batch_def, suite=suite)
        )
    except Exception:
        vdef = context.validation_definitions.get("loan_vdef")

    try:
        checkpoint = context.checkpoints.add(
            gx.Checkpoint(
                name="loan_checkpoint",
                validation_definitions=[vdef],
                actions=[UpdateDataDocsAction(name="update_data_docs")],
                result_format=ResultFormat.COMPLETE,
            )
        )
    except Exception:
        checkpoint = context.checkpoints.get("loan_checkpoint")

    result = checkpoint.run(batch_parameters={"dataframe": df})
    return _summarise_gx(result, df, file_name)


def _summarise_gx(result, df: pd.DataFrame, file_name: str) -> dict:
    """Turn a GX CheckpointResult into the DAG summary dict."""
    invalid_idx: set[int] = set()
    error_summary: dict[str, int] = {}
    missing_column = False

    run_results = list(result.run_results.values())
    for vres in run_results:
        for r in vres["results"]:
            if r["success"]:
                continue
            etype = r["expectation_config"]["type"]
            res = r.get("result", {})
            if etype == "expect_column_to_exist":
                missing_column = True
                error_summary[etype] = error_summary.get(etype, 0) + 1
                continue
            idx_list = res.get("unexpected_index_list") or []
            count = res.get("unexpected_count", len(idx_list))
            error_summary[etype] = error_summary.get(etype, 0) + int(count)
            invalid_idx.update(int(i) for i in idx_list)

    return _finalise(
        df,
        file_name,
        invalid_idx,
        error_summary,
        missing_column,
        report_name="index.html",
    )


def _pandas_validate(df: pd.DataFrame, file_name: str) -> dict:
    invalid_idx: set[int] = set()
    error_summary: dict[str, int] = {}
    missing_column = any(c not in df.columns for c in REQUIRED_COLUMNS)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            error_summary[f"missing_column:{c}"] = 1

    def flag(mask: pd.Series, name: str):
        idx = df.index[mask.fillna(True)].tolist()
        if idx:
            error_summary[name] = error_summary.get(name, 0) + len(idx)
            invalid_idx.update(int(i) for i in idx)

    if "cibil_score" in df:
        num = pd.to_numeric(df["cibil_score"], errors="coerce")
        flag(num.isna() | ~num.between(300, 900), "cibil_score_invalid")
    if "education" in df:
        flag(~df["education"].isin(["Graduate", "Not Graduate"]), "education_invalid")
    if "self_employed" in df:
        flag(~df["self_employed"].isin(["Yes", "No"]), "self_employed_invalid")
    for c in NON_NEGATIVE:
        if c in df:
            num = pd.to_numeric(df[c], errors="coerce")
            flag(num.isna() | (num < 0), f"{c}_invalid")
    if "loan_id" in df:
        flag(df["loan_id"].duplicated(keep=False), "duplicate_loan_id")

    return _finalise(
        df,
        file_name,
        invalid_idx,
        error_summary,
        missing_column,
        report_name="pandas_fallback.html",
    )


def _finalise(df, file_name, invalid_idx, error_summary, missing_column, report_name):
    total = len(df)
    invalid = total if missing_column else len(invalid_idx)
    valid = max(total - invalid, 0)
    invalid_pct = round(100 * invalid / total, 2) if total else 0.0
    return {
        "file_name": file_name,
        "total_rows": total,
        "valid_rows": valid,
        "invalid_rows": invalid,
        "invalid_pct": invalid_pct,
        "criticality": compute_criticality(missing_column, invalid_pct),
        "error_summary": error_summary,
        "invalid_indices": sorted(invalid_idx),
        "missing_column": missing_column,
        "report_name": report_name,
    }


def validate_batch(df: pd.DataFrame, file_name: str) -> dict:
    """Validate a batch, preferring GX and falling back to pandas."""
    try:
        return _gx_validate(df.reset_index(drop=True), file_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[gx_validation] GX unavailable ({exc}); using pandas fallback.")
        return _pandas_validate(df.reset_index(drop=True), file_name)

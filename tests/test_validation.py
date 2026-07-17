"""Tests for the validation summary contract (uses the pandas fallback path)."""

from __future__ import annotations

import pandas as pd

from src.gx_validation import _pandas_validate

GOOD_ROW = {
    "loan_id": 1,
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000,
}


def _df(rows):
    return pd.DataFrame(rows)


def test_all_good_rows():
    out = _pandas_validate(_df([GOOD_ROW, GOOD_ROW | {"loan_id": 2}]), "f.csv")
    assert out["invalid_rows"] == 0
    assert out["criticality"] == "None"


def test_detects_bad_cibil():
    # 1 bad of 2 rows -> 50% invalid -> Medium (High requires >50%).
    bad = GOOD_ROW | {"loan_id": 2, "cibil_score": 5000}
    out = _pandas_validate(_df([GOOD_ROW, bad]), "f.csv")
    assert out["invalid_rows"] == 1
    assert out["invalid_pct"] == 50.0
    assert out["criticality"] == "Medium"


def test_high_when_majority_invalid():
    # 2 bad of 3 rows -> 66% invalid -> High.
    bad1 = GOOD_ROW | {"loan_id": 2, "cibil_score": 5000}
    bad2 = GOOD_ROW | {"loan_id": 3, "education": "XYZ"}
    out = _pandas_validate(_df([GOOD_ROW, bad1, bad2]), "f.csv")
    assert out["criticality"] == "High"


def test_missing_column_is_high():
    df = _df([GOOD_ROW]).drop(columns=["bank_asset_value"])
    out = _pandas_validate(df, "f.csv")
    assert out["missing_column"] is True
    assert out["criticality"] == "High"


def test_detects_invalid_category():
    bad = GOOD_ROW | {"loan_id": 2, "education": "XYZ"}
    out = _pandas_validate(_df([GOOD_ROW, bad]), "f.csv")
    assert "education_invalid" in out["error_summary"]

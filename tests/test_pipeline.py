"""Unit tests for feature engineering, error injection, splitting and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_ORDER, NumpyLogReg, apply_scaler, engineer, fit_scaler
from src.generate_errors import DEFAULTS, inject_errors
from src.gx_validation import compute_criticality
from src.split_data import ROWS_PER_FILE, split


SAMPLE = pd.DataFrame([{
    "loan_id": 1, "no_of_dependents": 2, "education": "Graduate",
    "self_employed": "No", "income_annum": 9600000, "loan_amount": 29900000,
    "loan_term": 12, "cibil_score": 778, "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000, "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000, "loan_status": "Approved",
}])


def test_engineer_shapes_and_order():
    X = engineer(SAMPLE)
    assert X.shape == (1, len(FEATURE_ORDER))
    assert X[0, FEATURE_ORDER.index("is_graduate")] == 1.0
    assert X[0, FEATURE_ORDER.index("is_self_employed")] == 0.0


def test_scaler_roundtrip():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = fit_scaler(X)
    Z = apply_scaler(X, scaler)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)


def test_numpy_logreg_learns_separable():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.5, (50, 2)), rng.normal(2, 0.5, (50, 2))])
    y = np.array([0] * 50 + [1] * 50)
    model = NumpyLogReg(epochs=500).fit(X, y)
    acc = ((model.predict_proba(X)[:, 1] >= 0.5).astype(int) == y).mean()
    assert acc > 0.95


def test_inject_errors_produces_problems():
    df = pd.concat([SAMPLE] * 100, ignore_index=True)
    corrupted, log = inject_errors(df, DEFAULTS, seed=1)
    assert sum(v for k, v in log.items()) > 0
    assert len(corrupted) == len(df)


@pytest.mark.parametrize("missing,pct,expected", [
    (True, 0, "High"),
    (False, 80, "High"),
    (False, 30, "Medium"),
    (False, 5, "Low"),
    (False, 0, "None"),
])
def test_compute_criticality(missing, pct, expected):
    assert compute_criticality(missing, pct) == expected


def test_split_exactly_10_rows(tmp_path):
    src = tmp_path / "src.csv"
    pd.concat([SAMPLE] * 50, ignore_index=True).to_csv(src, index=False)
    out = tmp_path / "raw"
    files = split(str(src), str(out), n_files=5)
    assert len(files) == 5
    for f in files:
        assert len(pd.read_csv(f)) == ROWS_PER_FILE

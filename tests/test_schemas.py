"""Pydantic request/response validation tests (per CI example requirements)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import LoanApplication, PredictionRequest, PredictionResponse

VALID = {
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


def test_valid_application_parses():
    app = LoanApplication(**VALID)
    assert app.cibil_score == 778
    assert app.education == "Graduate"


def test_cibil_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        LoanApplication(**{**VALID, "cibil_score": 2000})


def test_negative_income_is_rejected():
    with pytest.raises(ValidationError):
        LoanApplication(**{**VALID, "income_annum": -1})


def test_invalid_category_is_rejected():
    with pytest.raises(ValidationError):
        LoanApplication(**{**VALID, "education": "XYZ"})


def test_prediction_request_needs_at_least_one_application():
    with pytest.raises(ValidationError):
        PredictionRequest(applications=[])


def test_response_model_structure():
    resp = PredictionResponse(
        model_name="random_forest",
        results=[{"prediction": "Approved", "probability": 0.91}],
    )
    assert resp.results[0].prediction == "Approved"
    assert resp.model_name == "random_forest"

"""Pydantic request/response models for the prediction API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    """One loan application. Field constraints mirror the data-quality rules."""

    no_of_dependents: int = Field(..., ge=0, le=15, examples=[2])
    education: Literal["Graduate", "Not Graduate"] = Field(..., examples=["Graduate"])
    self_employed: Literal["Yes", "No"] = Field(..., examples=["No"])
    income_annum: float = Field(..., ge=0, examples=[9600000])
    loan_amount: float = Field(..., ge=0, examples=[29900000])
    loan_term: int = Field(..., ge=1, le=40, examples=[12])
    cibil_score: int = Field(..., ge=300, le=900, examples=[778])
    residential_assets_value: float = Field(..., ge=0, examples=[2400000])
    commercial_assets_value: float = Field(..., ge=0, examples=[17600000])
    luxury_assets_value: float = Field(..., ge=0, examples=[22700000])
    bank_asset_value: float = Field(..., ge=0, examples=[8000000])


class PredictionRequest(BaseModel):
    """Batch prediction request. ``source`` records where it came from."""

    applications: list[LoanApplication] = Field(..., min_length=1)
    source: Literal["webapp", "scheduled"] = "webapp"


class PredictionItem(BaseModel):
    prediction: Literal["Approved", "Rejected"]
    probability: float


class PredictionResponse(BaseModel):
    model_name: str
    results: list[PredictionItem]


class PastPrediction(BaseModel):
    id: int
    created_at: datetime
    source: str
    prediction: str
    probability: float
    cibil_score: Optional[int] = None
    loan_amount: Optional[float] = None
    income_annum: Optional[float] = None

    model_config = {"from_attributes": True}

"""
Shared feature engineering for the loan-approval model.

Kept deliberately dependency-light (numpy + pandas only) so that training,
the API and the batch-prediction DAG all encode inputs identically. sklearn is
optional; when it is unavailable a numpy logistic-regression backend is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLS = [
    "no_of_dependents", "income_annum", "loan_amount", "loan_term",
    "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value",
]
# One-hot (drop-first) encodings for the two binary categoricals.
DERIVED_COLS = ["is_graduate", "is_self_employed"]
FEATURE_ORDER = NUMERIC_COLS + DERIVED_COLS

TARGET = "loan_status"
LABELS = {0: "Rejected", 1: "Approved"}


def engineer(df: pd.DataFrame) -> np.ndarray:
    """Turn a raw dataframe into an ordered numeric feature matrix."""
    out = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    out["is_graduate"] = (df["education"].astype(str).str.strip() == "Graduate").astype(float)
    out["is_self_employed"] = (df["self_employed"].astype(str).str.strip() == "Yes").astype(float)
    return out[FEATURE_ORDER].to_numpy(dtype=float)


def fit_scaler(X: np.ndarray) -> dict:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std}


def apply_scaler(X: np.ndarray, scaler: dict) -> np.ndarray:
    return (X - scaler["mean"]) / scaler["std"]


class NumpyLogReg:
    """Minimal L2-regularised logistic regression trained with gradient descent.

    Used as a zero-dependency fallback when scikit-learn is not installed so the
    project always produces a usable model artifact.
    """

    def __init__(self, lr: float = 0.1, epochs: int = 2000, l2: float = 1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogReg":
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self._sigmoid(X @ self.w + self.b)
        return np.column_stack([1 - p1, p1])

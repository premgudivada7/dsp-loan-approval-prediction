from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd

from src.features import apply_scaler, engineer

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[1] / "model" / "model.pkl")


class ModelService:
    def __init__(self, artifact: dict):
        self.estimator = artifact["estimator"]
        self.scaler = artifact["scaler"]
        self.labels = artifact["labels"]
        self.model_name = artifact.get("model_name", "unknown")
        self.backend = artifact.get("backend", "unknown")

    @classmethod
    def load(cls, path: str | None = None) -> "ModelService":
        path = path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model file not found at {path}. Run `python -m src.train` first."
            )
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    def predict(self, df: pd.DataFrame) -> list[dict]:
        """Score a dataframe of raw applications; returns label + probability."""
        X = apply_scaler(engineer(df), self.scaler)
        proba = self.estimator.predict_proba(X)[:, 1]
        out = []
        for p in proba:
            label = self.labels[1] if p >= 0.5 else self.labels[0]
            out.append({"prediction": label, "probability": round(float(p), 4)})
        return out

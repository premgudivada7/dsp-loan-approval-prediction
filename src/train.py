"""
Train candidate classifiers on the clean (GOOD) data, compare them, and persist
the best model to ``model/model.pkl`` for the API to load at startup.

When scikit-learn is available it trains LogisticRegression, RandomForest and
GradientBoosting and keeps the best by F1. When it is not, it falls back to a
numpy logistic regression so a usable artifact is always produced.

Run:  python -m src.train
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    LABELS,
    TARGET,
    FEATURE_ORDER,
    NumpyLogReg,
    apply_scaler,
    engineer,
    fit_scaler,
)

ROOT = Path(__file__).resolve().parents[1]
CLEAN_CSV = ROOT / "data" / "clean_loan_data.csv"
MODEL_PATH = ROOT / "model" / "model.pkl"
METRICS_PATH = ROOT / "model" / "metrics.json"


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    acc = float((pred == y_true).mean())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # ROC-AUC via rank statistic (Mann-Whitney U), no sklearn needed.
    order = np.argsort(proba)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(proba) + 1)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    auc = (
        (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        if n_pos and n_neg
        else 0.5
    )
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "roc_auc": round(auc, 4)}


def _split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * (1 - test_size))
    tr, te = idx[:cut], idx[cut:]
    return X[tr], X[te], y[tr], y[te]


def run() -> dict:
    df = pd.read_csv(CLEAN_CSV)
    X_raw = engineer(df)
    y = (df[TARGET].astype(str).str.strip() == "Approved").astype(int).to_numpy()

    scaler = fit_scaler(X_raw)
    X = apply_scaler(X_raw, scaler)
    X_tr, X_te, y_tr, y_te = _split(X, y)

    candidates: dict = {}
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        candidates = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }
        backend = "sklearn"
    except ImportError:
        candidates = {"logistic_regression_numpy": NumpyLogReg()}
        backend = "numpy"

    results, best = {}, None
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        m = _metrics(y_te, proba)
        results[name] = m
        if best is None or m["f1"] > results[best[0]]["f1"]:
            best = (name, model)

    artifact = {
        "backend": backend,
        "estimator": best[1],
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "labels": LABELS,
        "model_name": best[0],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(artifact, fh)

    summary = {"backend": backend, "best_model": best[0], "comparison": results}
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()

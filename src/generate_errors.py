"""
Data-error generation script.

Takes a clean dataset and randomly injects errors so the ingestion/validation
pipeline has something to catch. Each error type has its own probability, so
every generated file has a different mix (realistic Grafana variation).

Implements 8 error types (min required is 7):
  Required
    1. completeness      – null in a required column               (row)
    2. validity          – value outside valid range               (row)
    3. consistency       – invalid categorical value               (row)
    4. schema            – a required column is dropped             (file)
    5. type              – non-numeric string in a numeric column   (row)
  Additional
    6. duplicate_pk      – duplicated loan_id                       (row)
    7. outlier           – absurd statistical outlier               (row)
    8. impossible        – impossible combination (loan_amount = 0) (row)

Usage:
  python -m src.generate_errors --input data/clean_loan_data.csv \
      --output data/corrupted.csv --null-prob 0.05 --validity-prob 0.05 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_NUMERIC = "cibil_score"
NUMERIC_FOR_TYPE = "income_annum"


def _rows_to_hit(n: int, prob: float, rng) -> np.ndarray:
    return np.where(rng.random(n) < prob)[0]


def inject_errors(df: pd.DataFrame, probs: dict, seed: int = None) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    df = df.copy().astype(object)
    n = len(df)
    log: dict[str, int] = {}

    # 1. completeness: null in a required column
    idx = _rows_to_hit(n, probs["null"], rng)
    df.iloc[idx, df.columns.get_loc(REQUIRED_NUMERIC)] = np.nan
    log["completeness_null"] = len(idx)

    # 2. validity: value outside valid range (cibil far out of [300, 900])
    idx = _rows_to_hit(n, probs["validity"], rng)
    df.iloc[idx, df.columns.get_loc("cibil_score")] = rng.choice([-10, 2000], size=len(idx))
    log["validity_range"] = len(idx)

    # 3. consistency: invalid categorical value
    idx = _rows_to_hit(n, probs["consistency"], rng)
    df.iloc[idx, df.columns.get_loc("education")] = "XYZ"
    log["consistency_category"] = len(idx)

    # 4. schema: drop a required column entirely (file-level)
    schema_dropped = False
    if rng.random() < probs["schema"]:
        df = df.drop(columns=["bank_asset_value"])
        schema_dropped = True
    log["schema_missing_column"] = int(schema_dropped)

    # 5. type: non-numeric string in a numeric column
    idx = _rows_to_hit(n, probs["type"], rng)
    df.iloc[idx, df.columns.get_loc(NUMERIC_FOR_TYPE)] = "not_a_number"
    log["type_wrong_dtype"] = len(idx)

    # 6. duplicate primary keys
    idx = _rows_to_hit(n, probs["duplicate"], rng)
    if len(idx):
        df.iloc[idx, df.columns.get_loc("loan_id")] = df.iloc[0, df.columns.get_loc("loan_id")]
    log["duplicate_pk"] = len(idx)

    # 7. statistical outlier
    idx = _rows_to_hit(n, probs["outlier"], rng)
    df.iloc[idx, df.columns.get_loc("loan_amount")] = 999_999_999
    log["statistical_outlier"] = len(idx)

    # 8. impossible value (loan_amount = 0)
    idx = _rows_to_hit(n, probs["impossible"], rng)
    df.iloc[idx, df.columns.get_loc("loan_amount")] = 0
    log["impossible_value"] = len(idx)

    return df, log


DEFAULTS = {
    "null": 0.05, "validity": 0.05, "consistency": 0.05, "schema": 0.10,
    "type": 0.05, "duplicate": 0.03, "outlier": 0.03, "impossible": 0.03,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inject data-quality errors.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=None)
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key}-prob", type=float, default=val, dest=key)
    return p


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)
    probs = {k: getattr(args, k) for k in DEFAULTS}
    corrupted, log = inject_errors(df, probs, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    corrupted.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(corrupted)} rows). Injected: {log}")


if __name__ == "__main__":
    main()

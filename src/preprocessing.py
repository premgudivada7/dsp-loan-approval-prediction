"""
Preprocessing & data-quality segregation for the loan-approval dataset.

Pipeline:
  1. Load the raw CSV and normalise column/value whitespace.
  2. Run EDA and save a correlation heatmap (used to drop redundant/weak features).
  3. Apply data-quality rules to split rows into GOOD (valid) and BAD (violating) sets.
  4. Persist the clean GOOD dataset that every downstream component (training,
     API, DAGs) works on, plus a rejected-rows file and a quality report.

Run:  python -m src.preprocessing
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---- Column groups / valid domains -----------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "loan_approval_dataset.csv"
CLEAN_CSV = ROOT / "data" / "clean_loan_data.csv"
REJECTED_CSV = ROOT / "reports" / "rejected_rows.csv"
HEATMAP_PNG = ROOT / "reports" / "correlation_heatmap.png"
REPORT_JSON = ROOT / "reports" / "preprocessing_report.json"

NUMERIC_COLS = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
CATEGORICAL_COLS = ["education", "self_employed"]
TARGET = "loan_status"

VALID_EDUCATION = {"Graduate", "Not Graduate"}
VALID_SELF_EMPLOYED = {"Yes", "No"}
VALID_STATUS = {"Approved", "Rejected"}


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw data and strip whitespace from headers and string cells."""
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def build_heatmap(df: pd.DataFrame, out: Path = HEATMAP_PNG) -> None:
    """Save a correlation heatmap of numeric features + encoded target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr_df = df[NUMERIC_COLS].copy()
    corr_df["approved"] = (df[TARGET] == "Approved").astype(int)
    corr = corr_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation heatmap (loan approval)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def quality_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series: True = GOOD row, False = BAD row.

    Rules derived from EDA of the dataset:
      - no nulls in any required column
      - cibil_score in [300, 900]
      - all monetary/count features non-negative (residential assets had -100k)
      - loan_term > 0, no_of_dependents in [0, 5]
      - categoricals and target within their allowed domains
    """
    required = NUMERIC_COLS + CATEGORICAL_COLS + [TARGET]
    good = df[required].notna().all(axis=1)

    good &= df["cibil_score"].between(300, 900)
    good &= df["no_of_dependents"].between(0, 5)
    good &= df["loan_term"] > 0
    for col in [
        "income_annum",
        "loan_amount",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]:
        good &= df[col] >= 0

    good &= df["education"].isin(VALID_EDUCATION)
    good &= df["self_employed"].isin(VALID_SELF_EMPLOYED)
    good &= df[TARGET].isin(VALID_STATUS)
    return good


def run() -> dict:
    df = load_raw()
    build_heatmap(df)

    mask = quality_mask(df)
    good_df, bad_df = df[mask].copy(), df[~mask].copy()

    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    REJECTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    good_df.to_csv(CLEAN_CSV, index=False)
    bad_df.to_csv(REJECTED_CSV, index=False)

    # Correlation of each numeric feature with the target (for feature notes).
    corr_target = (
        df[NUMERIC_COLS]
        .assign(approved=(df[TARGET] == "Approved").astype(int))
        .corr(numeric_only=True)["approved"]
        .drop("approved")
        .round(3)
        .to_dict()
    )

    report = {
        "total_rows": int(len(df)),
        "good_rows": int(len(good_df)),
        "bad_rows": int(len(bad_df)),
        "target_distribution": good_df[TARGET].value_counts().to_dict(),
        "target_correlation": corr_target,
        "clean_csv": str(CLEAN_CSV.relative_to(ROOT)),
        "rejected_csv": str(REJECTED_CSV.relative_to(ROOT)),
        "heatmap": str(HEATMAP_PNG.relative_to(ROOT)),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()

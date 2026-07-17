"""
Dataset splitting script.

Splits a dataset into N separate files of EXACTLY 10 rows each and stores them
in the raw_data folder that feeds the ingestion job. Optionally injects errors
into each file first (so the ingestion demo sees a realistic mix of problems).

Args: dataset path, raw_data folder path, and the NUMBER OF FILES to generate
(not rows per file — each file always has 10 rows).

Usage:
  python -m src.split_data --input data/clean_loan_data.csv \
      --raw-data data/raw_data --n-files 30 --corrupt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.generate_errors import DEFAULTS, inject_errors

ROWS_PER_FILE = 10  # requirement: exactly 10 rows per file


def split(input_path: str, raw_data: str, n_files: int,
          corrupt: bool = False, seed: int = 42) -> list[str]:
    df = pd.read_csv(input_path).sample(frac=1, random_state=seed).reset_index(drop=True)
    needed = n_files * ROWS_PER_FILE
    if len(df) < needed:  # cycle rows if the source is too small
        reps = -(-needed // len(df))
        df = pd.concat([df] * reps, ignore_index=True)

    out_dir = Path(raw_data)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i in range(n_files):
        chunk = df.iloc[i * ROWS_PER_FILE:(i + 1) * ROWS_PER_FILE].copy()
        if corrupt:
            chunk, _ = inject_errors(chunk, DEFAULTS, seed=seed + i)
        path = out_dir / f"batch_{i:03d}.csv"
        chunk.to_csv(path, index=False)
        written.append(str(path))
    print(f"Wrote {len(written)} files ({ROWS_PER_FILE} rows each) to {out_dir}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description="Split dataset into 10-row files.")
    p.add_argument("--input", required=True)
    p.add_argument("--raw-data", required=True)
    p.add_argument("--n-files", type=int, required=True)
    p.add_argument("--corrupt", action="store_true",
                   help="Inject data-quality errors into each file.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    split(args.input, args.raw_data, args.n_files, args.corrupt, args.seed)


if __name__ == "__main__":
    main()

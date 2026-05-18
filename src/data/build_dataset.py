"""
Build the processed modelling dataset from raw Elexon system prices.

Steps:
  1. Load   data/raw/system_prices.csv
  2. Validate — flag duplicates and missing settlement periods
  3. Add    settlement_datetime (UTC start of each 30-min period)
  4. Denoise — Tukey outer-fence winsorisation on SSP (IQR-based)
               preserves negative prices that are market-valid while
               capping extreme outliers; stores ssp_raw for reference
  5. Derive base columns (abs_imbalance_volume, price_derivation_code_P)
  6. Save   data/processed/dataset.csv

Denoising rationale (Weron 2014 §3.1):
  Electricity prices exhibit rare but large spikes that distort model
  training. The Tukey outer fence (Q1 − 3·IQR, Q3 + 3·IQR) is a
  parameter-free, rank-based method robust to heavy tails and small
  samples. Values outside the fence are clipped (winsorised); an
  `is_spike` flag marks affected rows for downstream analysis.

Usage:
    python src/data/build_dataset.py
    python src/data/build_dataset.py --raw data/raw/system_prices.csv --out data/processed/dataset.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "system_prices.csv"
OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "dataset.csv"

PERIODS_PER_DAY = 48

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["settlement_date"])
    df["settlement_date"] = df["settlement_date"].dt.date
    log.info("Loaded %d rows from %s", len(df), path)
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=["settlement_date", "settlement_period"])
    dropped = n_before - len(df)
    if dropped:
        log.warning("Dropped %d duplicate (date, period) rows", dropped)

    all_dates = pd.date_range(
        start=str(min(df["settlement_date"])),
        end=str(max(df["settlement_date"])),
        freq="D",
    ).date
    expected = len(all_dates) * PERIODS_PER_DAY
    actual = len(df)
    if actual < expected:
        log.warning(
            "Expected %d rows for %d days, found %d — %d periods missing",
            expected, len(all_dates), actual, expected - actual,
        )
    else:
        log.info("Coverage check passed: %d rows for %d days", actual, len(all_dates))

    return df


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    offset_minutes = (df["settlement_period"] - 1) * 30
    df["settlement_datetime"] = pd.to_datetime(df["settlement_date"].astype(str)) + pd.to_timedelta(
        offset_minutes, unit="min"
    )
    return df


def denoise(df: pd.DataFrame, col: str = "ssp") -> pd.DataFrame:
    """
    Winsorise `col` using the Tukey outer fence (Q1 − 3·IQR, Q3 + 3·IQR).

    Stores the original values in `{col}_raw` and an `is_spike` boolean
    flag. Clipping replaces spike values with the fence boundary rather
    than dropping them, preserving the time-series structure.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr

    df[f"{col}_raw"] = df[col].copy()
    df["is_spike"] = (df[col] < lower) | (df[col] > upper)
    df[col] = df[col].clip(lower=lower, upper=upper)

    n_spikes = df["is_spike"].sum()
    log.info(
        "Denoising: lower fence=%.2f, upper fence=%.2f, %d spike(s) winsorised",
        lower, upper, n_spikes,
    )
    return df


def derive_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["abs_imbalance_volume"] = df["net_imbalance_volume"].abs()
    # Binary encoding of price derivation methodology
    df["price_derivation_code_P"] = (df["price_derivation_code"] == "P").astype(np.int8)
    # replacement_price is null for "N" periods (no replacement applied).
    # Impute with ssp so the column has no NaN gaps in the processed file.
    # WARNING: do NOT use replacement_price as a model feature — for P-code
    # periods it equals SSP by definition (API returns SSP=replacement_price),
    # and after this fillna it equals SSP for all N-code nulls too. Using it
    # as a feature is near-perfect target leakage (corr=0.9999 with SSP).
    df["replacement_price"] = df["replacement_price"].fillna(df["ssp"])
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)


def build(raw_path: Path, out_path: Path) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = validate(df)
    df = add_datetime(df)
    df = denoise(df, col="ssp")
    df = derive_base_columns(df)
    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    save(df, out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed modelling dataset")
    parser.add_argument("--raw", default=str(RAW_FILE))
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()
    build(Path(args.raw), Path(args.out))


if __name__ == "__main__":
    main()

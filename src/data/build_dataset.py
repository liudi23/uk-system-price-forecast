"""
Build the processed modelling dataset from raw Elexon system prices.

Steps:
  1. Load  data/raw/system_prices.csv
  2. Validate — flag duplicates and missing settlement periods
  3. Add settlement_datetime (UTC start of each 30-min period)
  4. Derive mid_price and abs_imbalance_volume
  5. Save data/processed/dataset.csv

Usage:
    python src/data/build_dataset.py
    python src/data/build_dataset.py --raw data/raw/system_prices.csv --out data/processed/dataset.csv
"""

import argparse
import logging
from pathlib import Path

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
    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=["settlement_date", "settlement_period"])
    dropped = n_before - len(df)
    if dropped:
        log.warning("Dropped %d duplicate (date, period) rows", dropped)

    # Report missing settlement periods
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
    """Add settlement_datetime: UTC start-of-period timestamp.

    Settlement period 1 starts at 00:00, period 2 at 00:30, etc.
    This is a naive UTC approximation; clock-change days may have 46/50 periods.
    """
    offset_minutes = (df["settlement_period"] - 1) * 30
    df["settlement_datetime"] = pd.to_datetime(df["settlement_date"].astype(str)) + pd.to_timedelta(
        offset_minutes, unit="min"
    )
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_price"] = (df["ssp"] + df["sbp"]) / 2
    df["abs_imbalance_volume"] = df["net_imbalance_volume"].abs()
    return df


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)


def build(raw_path: Path, out_path: Path) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = validate(df)
    df = add_datetime(df)
    df = derive_features(df)
    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    save(df, out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed modelling dataset")
    parser.add_argument("--raw", default=str(RAW_FILE), help="Path to raw system_prices.csv")
    parser.add_argument("--out", default=str(OUT_FILE), help="Output path for processed dataset.csv")
    args = parser.parse_args()
    build(Path(args.raw), Path(args.out))


if __name__ == "__main__":
    main()

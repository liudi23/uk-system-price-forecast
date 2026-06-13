"""
Extend dataset_5yr.csv with new rows from system_prices.csv.

Reads the processed 5-year dataset and appends any settlement periods that
are present in system_prices.csv but not yet in dataset_5yr.csv.  Applies
the same Tukey outer-fence winsorisation and derived columns as build_dataset.py
so the extended file is schema-consistent.

Usage:
    python src/data/extend_dataset.py
    python src/data/extend_dataset.py \
        --raw  data/raw/system_prices.csv \
        --base data/processed/dataset_5yr.csv \
        --out  data/processed/dataset_5yr.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

BASE_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "dataset_5yr.csv"
RAW_FILE  = Path(__file__).resolve().parents[2] / "data" / "raw"       / "system_prices.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extend(base_path: Path, raw_path: Path, out_path: Path) -> None:
    base = pd.read_csv(base_path, parse_dates=["settlement_date", "settlement_datetime"])
    raw  = pd.read_csv(raw_path,  parse_dates=["settlement_date"])

    last_date = base["settlement_date"].max()
    new       = raw[raw["settlement_date"] > last_date].copy()

    if new.empty:
        log.info("dataset_5yr.csv already up to date (last=%s). Nothing to append.", last_date.date())
        return

    log.info("Appending %d rows (%s → %s) to %s",
             len(new), new["settlement_date"].min().date(), new["settlement_date"].max().date(), out_path)

    new["settlement_datetime"] = (
        new["settlement_date"] + pd.to_timedelta((new["settlement_period"] - 1) * 30, unit="min")
    )

    # Tukey outer-fence winsorisation — reuse fence from existing data
    q1, q3 = base["ssp_raw"].quantile(0.25), base["ssp_raw"].quantile(0.75)
    upper_fence = q3 + 3 * (q3 - q1)
    lower_fence = q1 - 3 * (q3 - q1)

    new["ssp_raw"]  = new["ssp"]
    new["ssp"]      = new["ssp_raw"].clip(lower_fence, upper_fence)
    new["is_spike"] = (new["ssp_raw"] > upper_fence).astype(int)
    new["abs_imbalance_volume"]    = new["net_imbalance_volume"].abs()
    new["price_derivation_code_P"] = new["price_derivation_code"].str.startswith("P")

    extended = (
        pd.concat([base, new], ignore_index=True)
        .sort_values("settlement_datetime")
        .reset_index(drop=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extended.to_csv(out_path, index=False)
    log.info("Saved %d rows → %s  (last date: %s)",
             len(extended), out_path, extended["settlement_date"].max().date())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend dataset_5yr.csv with new price rows")
    parser.add_argument("--raw",  default=str(RAW_FILE))
    parser.add_argument("--base", default=str(BASE_FILE))
    parser.add_argument("--out",  default=str(BASE_FILE))
    args = parser.parse_args()
    extend(Path(args.base), Path(args.raw), Path(args.out))


if __name__ == "__main__":
    main()

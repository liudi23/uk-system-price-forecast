"""
Feature engineering pipeline for UK electricity price forecasting.

Reads the cleaned dataset produced by build_dataset.py and appends:
  1. Calendar / temporal features  (calendar_features.py)
  2. Lag and rolling-window features  (lag_features.py)
  3. Weather features (weather_features.py) — optional, requires weather_uk.csv

Output: data/processed/features.csv

Usage:
    python src/features/build_features.py
    python src/features/build_features.py --input data/processed/dataset_5yr.csv \
                                           --output data/processed/features_5yr.csv
    python src/features/build_features.py --no-weather   # skip weather merge
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from calendar_features import add_calendar_features
from lag_features import add_lag_features
from weather_features import WEATHER_FILE, add_weather_features, load_weather, merge_weather

IN_FILE  = Path(__file__).resolve().parents[2] / "data" / "processed" / "dataset.csv"
OUT_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_features(in_path: Path, out_path: Path, use_weather: bool = True) -> pd.DataFrame:
    log.info("Loading dataset from %s", in_path)
    df = pd.read_csv(in_path, parse_dates=["settlement_datetime", "settlement_date"])
    df = df.sort_values("settlement_datetime").reset_index(drop=True)
    n_raw = len(df)

    log.info("Adding calendar features…")
    df = add_calendar_features(df)

    log.info("Adding lag and rolling features…")
    df = add_lag_features(df)

    if use_weather and WEATHER_FILE.exists():
        log.info("Adding weather features from %s…", WEATHER_FILE)
        weather = load_weather(WEATHER_FILE)
        df = merge_weather(df, weather)
        df = add_weather_features(df)
    elif use_weather:
        log.warning("Weather file not found (%s) — skipping weather features. "
                    "Run src/data/fetch_weather.py to generate it.", WEATHER_FILE)

    # Completeness report — binding constraint is ssp_lag_336 (7-day warm-up)
    lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c or "_diff_" in c]
    n_complete = df[lag_cols].notna().all(axis=1).sum()
    log.info(
        "Feature matrix: %d rows total, %d complete (%.0f%%) — "
        "%d rows have NaN from insufficient lag history",
        n_raw, n_complete, 100 * n_complete / n_raw, n_raw - n_complete,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Saved %d rows × %d columns → %s", len(df), len(df.columns), out_path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature-engineered dataset")
    parser.add_argument("--input",      default=str(IN_FILE))
    parser.add_argument("--output",     default=str(OUT_FILE))
    parser.add_argument("--no-weather", action="store_true", help="Skip weather feature merge")
    args = parser.parse_args()
    build_features(Path(args.input), Path(args.output), use_weather=not args.no_weather)


if __name__ == "__main__":
    main()

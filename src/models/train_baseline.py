"""
Naive baseline forecasting models for UK system price (SSP).

Three baselines (all parameter-free, no training required):

  1. Naive (persistence)  — ŷ_t = SSP_{t−48}
     Predicts today's settlement period with the same period yesterday.
     Strong baseline in electricity markets due to daily seasonality.

  2. Seasonal naive       — ŷ_t = SSP_{t−336}
     Same period one week ago. Captures weekly seasonality
     (weekday vs weekend profiles).

  3. Rolling mean (24 h)  — ŷ_t = mean(SSP_{t−48} … SSP_{t−1})
     Smoothed prediction using the previous 24-hour window.

Evaluation uses the last TEST_DAYS of the usable dataset as the
hold-out test set (identical split used by train_lgbm.py).

Usage:
    python src/models/train_baseline.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from src/models/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import metrics, print_report, save_metrics, compare_table

FEATURES_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"
METRICS_FILE = Path(__file__).resolve().parents[2] / "model_assets" / "baseline_metrics.json"

TEST_DAYS = 7

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_usable(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["settlement_datetime"])
    df = df.sort_values("settlement_datetime").reset_index(drop=True)
    lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c or "_diff_" in c]
    df = df[df[lag_cols].notna().all(axis=1)].reset_index(drop=True)
    log.info("Usable rows after lag warm-up: %d", len(df))
    return df


def split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["settlement_datetime"].max() - pd.Timedelta(days=test_days)
    train = df[df["settlement_datetime"] <= cutoff]
    test = df[df["settlement_datetime"] > cutoff]
    log.info(
        "Train: %d rows (%s → %s) | Test: %d rows (%s → %s)",
        len(train), train["settlement_datetime"].min().date(), train["settlement_datetime"].max().date(),
        len(test), test["settlement_datetime"].min().date(), test["settlement_datetime"].max().date(),
    )
    return train, test


def main() -> None:
    df = load_usable(FEATURES_FILE)
    _, test = split(df, TEST_DAYS)

    y_true = test["ssp"].values

    baselines = {
        "Naive (t−48)":       test["ssp_lag_48"].values,
        "Seasonal naive (t−336)": test["ssp_lag_336"].values,
        "Rolling mean 24 h":  test["ssp_roll_mean_48"].values,
    }

    all_metrics = {}
    for name, y_pred in baselines.items():
        m = metrics(y_true, y_pred)
        print_report(name, m)
        all_metrics[name] = m

    print("\nComparison (sorted by MAE):")
    print(compare_table(all_metrics).to_string())

    save_metrics(all_metrics, METRICS_FILE)


if __name__ == "__main__":
    main()

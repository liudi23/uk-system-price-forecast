"""
Phase 3 training: level-shape decomposition.

Two-stage model eliminating recursive error propagation:

  Stage 1 — Daily level model (quantile HGBR P10/P50/P90)
      Target  : daily mean of ssp_raw across all 48 settlement periods
      Features: daily-aggregated lags (shift ≥ 1 day) + calendar +
                day-ahead weather for day D
      Guarantee: all features reference data before day D starts → no leakage

  Stage 2 — Intra-day shape model (HGBR P50)
      Target  : ssp_raw_h − actual_daily_mean_D  (deviation from daily mean)
      Features: fixed lag-48+ features only (ssp_lag_48/96/336,
                weather_lag_48, niv_lag_48, calendar, daily-level lags)
      Guarantee: rolling windows (roll_6/48/336) EXCLUDED because
                 shift(1).rolling(w) includes within-day prices for SPs 2–48.
                 Only fixed-point lags (≥ 48) are used — safe for all 48 SPs.

Evaluation: non-recursive two-stage simulation.
  For each test day D:
    1. level_pred   ← level model(daily features before D)     [no recursion]
    2. deviation_h  ← shape model(SP-level lag-48+ features)   [no recursion]
    3. forecast_h   = level_pred + deviation_h

Outputs (model_assets/):
    level_q10.pkl, level_q50.pkl, level_q90.pkl
    shape_q50.pkl
    level_feature_cols.json, shape_feature_cols.json
    phase3_metrics.json
    test_predictions_phase3.csv

Usage:
    python src/models/train_phase3.py
    python src/models/train_phase3.py --features data/processed/features_5yr.csv
    python src/models/train_phase3.py --test-days 7 --val-days 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))

from evaluate import metrics, print_report, save_metrics
from level_features import LEVEL_TARGET, build_level_dataset, get_level_feature_cols

FEATURES_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_5yr.csv"
ASSETS_DIR    = Path(__file__).resolve().parents[2] / "model_assets"

TEST_DAYS = 7
VAL_DAYS  = 3
QUANTILES = [0.10, 0.50, 0.90]

# HGBR hyperparameters — same base as Phase 2 for fair comparison
BASE_PARAMS = {
    "max_iter":          1000,
    "learning_rate":     0.05,
    "max_leaf_nodes":    31,
    "min_samples_leaf":  10,
    "l2_regularization": 0.1,
    "max_bins":          255,
    "early_stopping":    True,
    "n_iter_no_change":  50,
    "random_state":      42,
}

# ── Columns excluded from the shape feature set ───────────────────────────────
# Strict criterion: only features whose lag is ≥ 48 SPs (1 full day) for EVERY
# settlement period in the forecast window may enter the shape model.
# Rolling windows shift(1).rolling(w) include within-day actual prices for
# SPs 2–48, so ALL rolling features are excluded regardless of window size.

SHAPE_ALWAYS_EXCLUDE = {
    # Identifiers and targets
    "settlement_date", "settlement_datetime", "settlement_period",
    "ssp", "ssp_raw", "is_spike",
    "ssp_shape_target",
    # Contemporaneous values (unknown at forecast time)
    "net_imbalance_volume", "abs_imbalance_volume",
    "price_derivation_code", "price_derivation_code_P",
    "replacement_price", "sell_price_adjustment", "buy_price_adjustment",
    "_raw_temp_c", "_raw_wind_ms", "_raw_solar_wm2", "_raw_precip_mm",
    # Short-lag price/NIV (lag < 48)
    "ssp_lag_1", "ssp_lag_2",
    "ssp_diff_1_48",                    # depends on ssp_lag_1
    "net_imbalance_volume_lag_1",
    # Short-lag weather (lag < 48)
    "temp_c_lag_1", "wind_ms_lag_1", "solar_wm2_lag_1", "precip_mm_lag_1",
    "wind_ramp_6",                      # lag_1 − lag_7
    # Daily level contemporaneous aggregates merged for target computation only
    LEVEL_TARGET,
    "ssp_daily_mean",
}

# All columns whose name contains these substrings are rolling features —
# contaminated by within-day prices for SPs 2–48.
SHAPE_EXCLUDE_SUBSTRINGS = (
    "_roll_6", "_roll_48", "_roll_336",
    "spike_count_roll_",
    "neg_count_roll_",
    "abs_niv_roll_",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and splitting
# ─────────────────────────────────────────────────────────────────────────────

def load_sp_data(path: Path) -> pd.DataFrame:
    """Load and clean the SP-level feature matrix."""
    df = pd.read_csv(path, parse_dates=["settlement_datetime"])
    df = df.sort_values("settlement_datetime").reset_index(drop=True)
    # Drop rows with NaN in any lag/rolling feature (warm-up period)
    lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c or "_diff_" in c]
    return df[df[lag_cols].notna().all(axis=1)].reset_index(drop=True)


def split_dates(df: pd.DataFrame, test_days: int, val_days: int):
    t_max    = df["settlement_datetime"].max()
    test_cut = t_max - pd.Timedelta(days=test_days)
    val_cut  = test_cut - pd.Timedelta(days=val_days)
    train = df[df["settlement_datetime"] <= val_cut]
    val   = df[(df["settlement_datetime"] > val_cut) & (df["settlement_datetime"] <= test_cut)]
    test  = df[df["settlement_datetime"] > test_cut]
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        log.info("%s: %d rows  (%s → %s)", name, len(part),
                 part["settlement_datetime"].min().date(),
                 part["settlement_datetime"].max().date())
    return train, val, test


def split_daily(daily: pd.DataFrame, test_days: int, val_days: int):
    d_max    = daily["date"].max()
    test_cut = d_max - pd.Timedelta(days=test_days)
    val_cut  = test_cut - pd.Timedelta(days=val_days)
    # Drop rows with NaN lag features (daily warm-up)
    feat_cols = get_level_feature_cols(daily)
    daily_clean = daily[daily[feat_cols].notna().all(axis=1)].reset_index(drop=True)
    train = daily_clean[daily_clean["date"] <= val_cut]
    val   = daily_clean[(daily_clean["date"] > val_cut) & (daily_clean["date"] <= test_cut)]
    test  = daily_clean[daily_clean["date"] > test_cut]
    log.info("Daily — Train: %d days  Val: %d days  Test: %d days",
             len(train), len(val), len(test))
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Shape feature construction
# ─────────────────────────────────────────────────────────────────────────────

def build_shape_data(sp_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge daily-level lag features into the SP-level frame and compute the
    shape target.  Only daily-lagged features (from days before D) are merged.

    Shape target  : ssp_raw_h − actual_daily_mean_D
    New SP features:
        ssp_daily_mean_lag1d   — yesterday's daily mean SSP (winsorised)
        ssp_daily_mean_lag7d   — 7 days ago daily mean
        ssp_raw_daily_max_lag1d — yesterday's daily max raw SSP (spike memory)
        niv_daily_mean_lag1d   — yesterday's daily mean NIV
        spike_count_lag1d      — yesterday's spike count
        spike_count_roll_7d_daily — 7-day rolling spike count (daily)
        ssp_lag48_deviation    — ssp_lag_48 minus yesterday's daily mean
        ssp_lag336_deviation   — ssp_lag_336 minus 7-day-ago daily mean
    """
    sp = sp_df.copy()
    sp["_date"] = sp["settlement_datetime"].dt.normalize()

    # Columns from daily_df to merge (lagged — safe for all SPs of day D)
    merge_cols = [
        "date", LEVEL_TARGET, "ssp_daily_mean",
        "ssp_daily_mean_lag1d", "ssp_daily_mean_lag7d",
        "ssp_raw_daily_max_lag1d", "niv_daily_mean_lag1d",
        "spike_count_lag1d", "spike_count_roll_7d",
        "ssp_daily_roll_mean_7d", "ssp_daily_roll_std_7d",
    ]
    # Add weather daily lags if present
    for v in ["temp_c", "wind_ms", "solar_wm2"]:
        col = f"{v}_daily_mean_lag1d"
        if col in daily_df.columns:
            merge_cols.append(col)

    merge_cols = [c for c in merge_cols if c in daily_df.columns]
    daily_sub  = daily_df[merge_cols].rename(columns={"date": "_date"})

    sp = sp.merge(daily_sub, on="_date", how="left")

    # Shape target: deviation of this SP's raw price from the day's actual mean
    sp["ssp_shape_target"] = sp["ssp_raw"] - sp[LEVEL_TARGET]

    # Relative position features (lag-48+ daily means are leakage-free)
    if "ssp_daily_mean_lag1d" in sp.columns and "ssp_lag_48" in sp.columns:
        sp["ssp_lag48_deviation"]  = sp["ssp_lag_48"]  - sp["ssp_daily_mean_lag1d"]
    if "ssp_daily_mean_lag7d" in sp.columns and "ssp_lag_336" in sp.columns:
        sp["ssp_lag336_deviation"] = sp["ssp_lag_336"] - sp["ssp_daily_mean_lag7d"]

    sp = sp.drop(columns=["_date"])
    return sp


def get_shape_feature_cols(sp: pd.DataFrame) -> list:
    """Return shape model feature columns: lag-48+ fixed features + calendar."""
    def _excluded(col: str) -> bool:
        if col in SHAPE_ALWAYS_EXCLUDE:
            return True
        return any(sub in col for sub in SHAPE_EXCLUDE_SUBSTRINGS)
    return [c for c in sp.columns if not _excluded(c)]


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_level_models(train_daily, val_daily, feat_cols):
    """Train P10/P50/P90 quantile HGBR on daily SSP mean."""
    combined = pd.concat([train_daily, val_daily], ignore_index=True)
    X = combined[feat_cols].values
    y = combined[LEVEL_TARGET].values
    val_frac = len(val_daily) / len(combined)

    models = {}
    for q in QUANTILES:
        log.info("Level model q=%.2f …", q)
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                          validation_fraction=val_frac,
                                          **BASE_PARAMS)
        m.fit(X, y)
        log.info("  best_iter=%d", m.n_iter_)
        models[q] = m
    return models


def train_shape_model(train_sp, val_sp, feat_cols):
    """Train P50 HGBR on intra-day shape deviations."""
    combined = pd.concat([train_sp, val_sp], ignore_index=True)
    mask     = combined["ssp_shape_target"].notna() & combined[feat_cols].notna().all(axis=1)
    combined = combined[mask]
    X        = combined[feat_cols].values
    y        = combined["ssp_shape_target"].values
    val_frac = len(val_sp) / len(combined)

    log.info("Shape model (P50) — %d training rows …", len(combined))
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.50,
                                      validation_fraction=val_frac,
                                      **BASE_PARAMS)
    m.fit(X, y)
    log.info("  best_iter=%d", m.n_iter_)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage evaluation (no recursion)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_phase3(level_models, shape_model,
                    test_daily, test_sp, shape_sp,
                    level_feat_cols, shape_feat_cols):
    """
    Non-recursive two-stage evaluation on the test set.

    Both models use only data available before the forecast day starts:
      - Level model: daily-aggregated lags from before day D
      - Shape model: fixed SP-level lags ≥ 48 (no within-day data)

    No recursion is required — errors do not compound across SPs.
    """
    test_dates = sorted(test_daily["date"].dt.date.unique())
    records = []

    for target_date in test_dates:
        # ── Stage 1: daily level ──────────────────────────────────────────
        day_row = test_daily[test_daily["date"].dt.date == target_date]
        if day_row.empty or day_row[level_feat_cols].isna().any(axis=1).any():
            log.warning("Skipping %s — missing level features", target_date)
            continue

        x_lvl = day_row[level_feat_cols].values
        l10 = float(level_models[0.10].predict(x_lvl)[0])
        l50 = float(level_models[0.50].predict(x_lvl)[0])
        l90 = float(level_models[0.90].predict(x_lvl)[0])
        # Enforce quantile ordering at level
        l10 = min(l10, l50)
        l90 = max(l90, l50)
        actual_level = float(day_row[LEVEL_TARGET].values[0])

        # ── Stage 2: intra-day shape ──────────────────────────────────────
        day_sps = (
            shape_sp[shape_sp["settlement_datetime"].dt.date == target_date]
            .sort_values("settlement_datetime")
        )
        if day_sps.empty:
            continue

        mask = day_sps[shape_feat_cols].notna().all(axis=1)
        X_sh = day_sps.loc[mask, shape_feat_cols].values
        dev  = shape_model.predict(X_sh)

        for i, (_, row) in enumerate(day_sps[mask].iterrows()):
            d = float(dev[i])
            records.append({
                "settlement_datetime": row["settlement_datetime"],
                "settlement_date":     str(target_date),
                "settlement_period":   int(row["settlement_period"]),
                "ssp_actual":          float(row["ssp_raw"]),
                "ssp_winsorised":      float(row["ssp"]),
                "actual_daily_level":  actual_level,
                "pred_level_q50":      l50,
                "pred_level_q10":      l10,
                "pred_level_q90":      l90,
                "pred_deviation":      d,
                "ssp_q50":             l50 + d,
                "ssp_q10":             l10 + d,
                "ssp_q90":             l90 + d,
                "ssp_predicted":       l50 + d,
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition metrics
# ─────────────────────────────────────────────────────────────────────────────

def decomposition_metrics(pred_df: pd.DataFrame) -> dict:
    """
    Report error decomposed into level error and shape error.

    level_mae     : |pred_daily_level − actual_daily_level| averaged over days
    shape_corr    : mean Pearson r between predicted and actual 48-SP profile
                    per day (independent of the level)
    peak_timing   : mean |argmax(pred) − argmax(actual)| in SPs per day
    """
    dates = pred_df["settlement_date"].unique()
    level_errs, shape_corrs, peak_gaps = [], [], []

    for d in dates:
        day = pred_df[pred_df["settlement_date"] == d].sort_values("settlement_period")
        if len(day) < 2:
            continue

        actual_level = float(day["actual_daily_level"].iloc[0])
        pred_level   = float(day["pred_level_q50"].iloc[0])
        level_errs.append(abs(pred_level - actual_level))

        # Shape correlation: compare intra-day profiles centred on zero
        act_shape  = day["ssp_actual"].values  - day["ssp_actual"].values.mean()
        pred_shape = day["ssp_predicted"].values - day["ssp_predicted"].values.mean()
        if act_shape.std() > 0 and pred_shape.std() > 0:
            r, _ = pearsonr(act_shape, pred_shape)
            shape_corrs.append(r)

        # Peak timing
        actual_peak = int(np.argmax(day["ssp_actual"].values))
        pred_peak   = int(np.argmax(day["ssp_predicted"].values))
        peak_gaps.append(abs(pred_peak - actual_peak))

    return {
        "level_mae":       round(float(np.mean(level_errs)),   2),
        "shape_corr_mean": round(float(np.mean(shape_corrs)),  4),
        "peak_timing_mae": round(float(np.mean(peak_gaps)),    2),
        "n_days":          len(dates),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",  default=str(FEATURES_FILE))
    parser.add_argument("--test-days", type=int, default=TEST_DAYS)
    parser.add_argument("--val-days",  type=int, default=VAL_DAYS)
    args = parser.parse_args()

    # ── Load SP-level data ────────────────────────────────────────────────
    log.info("Loading SP-level features from %s …", args.features)
    sp_df = load_sp_data(Path(args.features))
    sp_train, sp_val, sp_test = split_dates(sp_df, args.test_days, args.val_days)

    # ── Build daily level dataset ─────────────────────────────────────────
    log.info("Building daily level dataset …")
    daily_df = build_level_dataset(sp_df)
    level_feat_cols = get_level_feature_cols(daily_df)
    log.info("Level features: %d", len(level_feat_cols))

    daily_train, daily_val, daily_test = split_daily(daily_df, args.test_days, args.val_days)

    # ── Build shape dataset ───────────────────────────────────────────────
    log.info("Building shape dataset (merging daily lags + computing deviations) …")
    shape_sp_all   = build_shape_data(sp_df,    daily_df)
    shape_sp_train = build_shape_data(sp_train, daily_df)
    shape_sp_val   = build_shape_data(sp_val,   daily_df)
    shape_sp_test  = build_shape_data(sp_test,  daily_df)

    shape_feat_cols = get_shape_feature_cols(shape_sp_all)
    log.info("Shape features: %d", len(shape_feat_cols))
    log.info("Shape target range: [%.1f, %.1f]  mean=%.2f",
             shape_sp_train["ssp_shape_target"].min(),
             shape_sp_train["ssp_shape_target"].max(),
             shape_sp_train["ssp_shape_target"].mean())

    # ── Train models ──────────────────────────────────────────────────────
    log.info("Training level models (P10/P50/P90) on daily SSP mean …")
    level_models = train_level_models(daily_train, daily_val, level_feat_cols)

    log.info("Training shape model (P50) on intra-day deviations …")
    shape_model = train_shape_model(shape_sp_train, shape_sp_val, shape_feat_cols)

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info("Running two-stage non-recursive evaluation on test set …")
    pred_df = evaluate_phase3(
        level_models, shape_model,
        daily_test, sp_test, shape_sp_test,
        level_feat_cols, shape_feat_cols,
    )

    y_true = pred_df["ssp_actual"].values
    y_pred = pred_df["ssp_predicted"].values
    m_q50  = metrics(y_true, y_pred)
    print_report("Phase 3 P50 — two-stage non-recursive (honest)", m_q50)

    # Level error metrics
    dc = decomposition_metrics(pred_df)
    log.info("Level MAE: %.2f £/MWh  Shape corr: %.3f  Peak timing: %.1f SPs",
             dc["level_mae"], dc["shape_corr_mean"], dc["peak_timing_mae"])

    # Reference: naive daily mean (predict yesterday's daily mean for every SP)
    naive_level = np.repeat(
        daily_test["ssp_daily_mean_lag1d"].values,
        [len(pred_df[pred_df["settlement_date"] == str(d.date())])
         for d in daily_test["date"]],
    )
    if len(naive_level) == len(y_true):
        m_naive = metrics(y_true, naive_level[:len(y_true)])
        log.info("Reference — naive (lag-1d daily mean) MAE: %.2f", m_naive["MAE"])

    # ── Save everything ───────────────────────────────────────────────────
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    for q, model in level_models.items():
        name = {0.10: "q10", 0.50: "q50", 0.90: "q90"}[q]
        joblib.dump(model, ASSETS_DIR / f"level_{name}.pkl")
    joblib.dump(shape_model, ASSETS_DIR / "shape_q50.pkl")
    log.info("Models saved → %s", ASSETS_DIR)

    with open(ASSETS_DIR / "level_feature_cols.json", "w") as f:
        json.dump(level_feat_cols, f)
    with open(ASSETS_DIR / "shape_feature_cols.json", "w") as f:
        json.dump(shape_feat_cols, f)

    pred_df.to_csv(ASSETS_DIR / "test_predictions_phase3.csv", index=False)
    log.info("Test predictions → %s", ASSETS_DIR / "test_predictions_phase3.csv")

    all_metrics = {
        "Phase3_P50_two_stage": m_q50,
        "decomposition": dc,
    }
    save_metrics(all_metrics, ASSETS_DIR / "phase3_metrics.json")

    print("\nPhase 3 vs Phase 2 comparison:")
    print(f"  Phase 2 recursive P50 MAE : £25.40/MWh")
    print(f"  Phase 3 two-stage P50 MAE : £{m_q50['MAE']:.2f}/MWh")
    print(f"  Level MAE                 : £{dc['level_mae']:.2f}/MWh/day")
    print(f"  Shape correlation         : {dc['shape_corr_mean']:.3f}")
    print(f"  Peak timing error         : {dc['peak_timing_mae']:.1f} SPs (30-min slots)")


if __name__ == "__main__":
    main()

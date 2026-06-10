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
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))

from evaluate import metrics, print_report, save_metrics
from level_features import LEVEL_TARGET, build_level_dataset, get_level_feature_cols

FEATURES_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_5yr.csv"
ASSETS_DIR    = Path(__file__).resolve().parents[2] / "model_assets"

TEST_DAYS   = 7   # last 7 days — full week, mix of normal + extreme days
VAL_DAYS    = 5   # 5 days before test — stable early-stopping signal
TRAIN_YEARS = 3   # drop data older than 3 years to reduce inflation drift
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
    # CPI deflator — used to scale training targets only, not a predictor
    "cpi_deflator",
    # Contemporaneous generation mix — actual day-D values, not available day-ahead;
    # the lag-48/lag-336 versions (wind_pct_lag_48 etc.) remain as features
    "wind_pct", "gas_pct",
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

# SP-level rolling features use shift(1).rolling(w), which contaminates SPs 2-48
# with within-day actual prices. The pattern to detect them: contains "_roll_" but
# NOT "_daily_roll_" (daily aggregates are safe) and NOT a "_roll_Nd" day-suffix
# (e.g. spike_count_roll_7d is safe, ssp_roll_mean_6 is not).
_SP_ROLL_RE = re.compile(r"_roll_\d+d$")

# ── H+2 additional exclusions ─────────────────────────────────────────────────
# Forecasting D+2 from data ending at D: lag-48 references D+1 (not settled yet).
# Any feature whose value depends on the H+1 day must be excluded from the H+2 model.
# The H+2 model uses lag-96 (D same SP, 2 days before D+2) as its primary signal.
_H2_EXTRA_EXCLUDE_PATTERNS = [
    "_lag_48", "_lag_49",            # reference D+1 (not settled)
    "same_sp_",                       # computed from lag-48 through lag-336 → includes D+1
    "ssp_sp_deviation", "ssp_sp_trend_7d",
    "_ramp_48",                       # lag-48 − lag-49
    "ssp_lag48_deviation",            # ssp_lag_48 − ssp_daily_mean_lag1d
    # Daily stats referencing yesterday of H+2 = D+1 (not settled)
    "ssp_daily_mean_lag1d",
    "ssp_raw_daily_max_lag1d",
    "niv_daily_mean_lag1d",
    "spike_count_lag1d",
    "temp_c_daily_mean_lag1d",
    "wind_ms_daily_mean_lag1d",
    "solar_wm2_daily_mean_lag1d",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading and splitting
# ─────────────────────────────────────────────────────────────────────────────

def load_sp_data(path: Path) -> pd.DataFrame:
    """Load and clean the SP-level feature matrix."""
    df = pd.read_csv(path, parse_dates=["settlement_datetime"])
    df = df.sort_values("settlement_datetime").reset_index(drop=True)
    # Drop rows where CORE lag features (ssp/niv) are NaN (warm-up period).
    # Weather, wind/gas, and CPI lag features are optional — HGBR handles missing
    # values natively, so rows are only dropped for price/NIV warm-up (lag ≤ 336).
    _optional = {"wind_pct_", "gas_pct_", "cpi_",
                 "temp_c_", "wind_ms_", "solar_wm2_", "precip_mm_",
                 "heating_", "cooling_"}
    core_lag_cols = [c for c in df.columns
                     if ("_lag_" in c or "_roll_" in c or "_diff_" in c)
                     and not any(c.startswith(p) for p in _optional)]
    return df[df[core_lag_cols].notna().all(axis=1)].reset_index(drop=True)


def split_dates(df: pd.DataFrame, test_days: int, val_days: int,
                train_years: int = TRAIN_YEARS):
    t_max       = df["settlement_datetime"].max()
    test_cut    = t_max - pd.Timedelta(days=test_days)
    val_cut     = test_cut - pd.Timedelta(days=val_days)
    train_start = t_max - pd.Timedelta(days=train_years * 365)
    train = df[(df["settlement_datetime"] > train_start) &
               (df["settlement_datetime"] <= val_cut)]
    val   = df[(df["settlement_datetime"] > val_cut) &
               (df["settlement_datetime"] <= test_cut)]
    test  = df[df["settlement_datetime"] > test_cut]
    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        log.info("%s: %d rows  (%s → %s)", name, len(part),
                 part["settlement_datetime"].min().date(),
                 part["settlement_datetime"].max().date())
    return train, val, test


def split_daily(daily: pd.DataFrame, test_days: int, val_days: int,
                train_years: int = TRAIN_YEARS):
    d_max       = daily["date"].max()
    test_cut    = d_max - pd.Timedelta(days=test_days)
    val_cut     = test_cut - pd.Timedelta(days=val_days)
    train_start = d_max - pd.Timedelta(days=train_years * 365)
    # Drop rows with NaN in core daily features (lag warm-up); weather/wind/gas optional
    feat_cols = get_level_feature_cols(daily)
    _opt = {"wind_pct_", "gas_pct_", "cpi_",
            "temp_c_", "wind_ms_", "solar_wm2_", "precip_mm_",
            "heating_", "cooling_"}
    core_feat = [c for c in feat_cols
                 if not any(c.startswith(p) for p in _opt)]
    daily_clean = daily[daily[core_feat].notna().all(axis=1)].reset_index(drop=True)
    train = daily_clean[(daily_clean["date"] > train_start) &
                        (daily_clean["date"] <= val_cut)]
    val   = daily_clean[(daily_clean["date"] > val_cut) &
                        (daily_clean["date"] <= test_cut)]
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
    """Return H+1 shape model feature columns: lag-48+ fixed features + calendar."""
    def _excluded(col: str) -> bool:
        if col in SHAPE_ALWAYS_EXCLUDE:
            return True
        if "_roll_" in col and "_daily_roll_" not in col and not _SP_ROLL_RE.search(col):
            return True
        return False
    return [c for c in sp.columns if not _excluded(c)]


def get_shape_feature_cols_h2(sp: pd.DataFrame) -> list:
    """Return H+2 shape feature columns: lag-96+ only (D+1 not settled at inference)."""
    def _excluded_h2(col: str) -> bool:
        if col in SHAPE_ALWAYS_EXCLUDE:
            return True
        if "_roll_" in col and "_daily_roll_" not in col and not _SP_ROLL_RE.search(col):
            return True
        if any(pat in col for pat in _H2_EXTRA_EXCLUDE_PATTERNS):
            return True
        return False
    return [c for c in sp.columns if not _excluded_h2(c)]


def build_shape_data_h2(sp_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Like build_shape_data() but merges lag-2d daily features instead of lag-1d,
    so all merged columns are available when forecasting D+2 from data ending at D.

    Shape target is the same: ssp_raw_h − actual_daily_mean_D+2.
    New SP features using only lag ≥ 96 relative to D+2:
        ssp_daily_mean_lag2d   — D daily mean (2 days before D+2)
        ssp_daily_mean_lag7d   — D-5 daily mean
        ssp_raw_daily_max_lag2d
        ssp_lag96_deviation    — ssp_lag_96 minus D daily mean
        ssp_lag336_deviation   — ssp_lag_336 minus D-5 daily mean
    """
    sp = sp_df.copy()
    sp["_date"] = sp["settlement_datetime"].dt.normalize()

    # H+2-safe daily columns: lag-2d and lag-7d are available 2 days before D+2
    merge_cols = ["date", LEVEL_TARGET, "ssp_daily_mean",
                  "ssp_daily_mean_lag2d", "ssp_daily_mean_lag7d",
                  "ssp_raw_daily_max_lag2d",
                  "spike_count_lag7d", "spike_count_roll_7d",
                  "ssp_daily_roll_mean_7d"]
    merge_cols = [c for c in merge_cols if c in daily_df.columns]
    daily_sub  = daily_df[merge_cols].rename(columns={"date": "_date"})

    sp = sp.merge(daily_sub, on="_date", how="left")
    sp["ssp_shape_target"] = sp["ssp_raw"] - sp[LEVEL_TARGET]

    # Relative position features using lag-96 (safe for H+2)
    if "ssp_daily_mean_lag2d" in sp.columns and "ssp_lag_96" in sp.columns:
        sp["ssp_lag96_deviation"] = sp["ssp_lag_96"] - sp["ssp_daily_mean_lag2d"]
    if "ssp_daily_mean_lag7d" in sp.columns and "ssp_lag_336" in sp.columns:
        sp["ssp_lag336_deviation"] = sp["ssp_lag_336"] - sp["ssp_daily_mean_lag7d"]

    sp = sp.drop(columns=["_date"])
    return sp


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_level_models(train_daily, val_daily, feat_cols):
    """Train P10/P50/P90 quantile HGBR on daily SSP mean (CPI-deflated to real terms)."""
    combined = pd.concat([train_daily, val_daily], ignore_index=True)
    X = combined[feat_cols].values
    # Apply CPI deflator so model trains on real (current-money) prices
    deflator = combined["cpi_deflator"].values if "cpi_deflator" in combined.columns else 1.0
    y = combined[LEVEL_TARGET].values * deflator
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


def train_neg_day_classifier(train_daily: pd.DataFrame, val_daily: pd.DataFrame) -> object:
    """
    Train a binary HGBR classifier: will tomorrow have ≥ 3 negative-price SPs?

    This targets the renewable oversupply regime (high wind + solar, low demand)
    that causes deep midday price crashes.  The classifier output (probability)
    is added as a level model feature at inference, improving the daily mean
    prediction on high-renewable days.

    Features used (all lag ≥ 1 day, no leakage):
        neg_sp_count_lag1d/7d, neg_sp_count_roll_7d  — recent negative-price history
        wind_pct_daily_mean_lag1d/7d                 — recent wind generation
        solar_wm2_daily_mean_lag1d                   — recent solar
        is_weekend, month, day_of_year               — calendar (negative prices more
                                                        common on weekends + summer)
    """
    NEG_THRESHOLD = 3   # ≥ 3 negative SPs in a day → "negative-price day"
    _clf_feats = [
        "neg_sp_count_lag1d", "neg_sp_count_lag7d", "neg_sp_count_roll_7d",
        "wind_pct_daily_mean_lag1d", "wind_pct_daily_mean_lag7d",
        "solar_wm2_daily_mean_lag1d",
        "is_weekend", "month", "day_of_year",
    ]
    combined = pd.concat([train_daily, val_daily], ignore_index=True)
    avail    = [c for c in _clf_feats if c in combined.columns]
    if not avail or "neg_sp_daily_count" not in combined.columns:
        log.warning("Neg-day classifier: required features missing — skipping")
        return None

    mask = combined[avail].notna().all(axis=1)
    X    = combined.loc[mask, avail].values
    y    = (combined.loc[mask, "neg_sp_daily_count"] >= NEG_THRESHOLD).astype(int).values

    if len(y) == 0:
        for col in avail:
            log.warning("  %s: %d NaN / %d rows", col,
                        combined[col].isna().sum(), len(combined))
        log.warning("Neg-day classifier: 0 samples after NaN filter — skipping")
        return None

    pos_frac = y.mean()
    log.info("Neg-day classifier: %d samples, %.1f%% positive (≥%d neg SPs)",
             len(y), 100 * pos_frac, NEG_THRESHOLD)
    if pos_frac < 0.01 or pos_frac > 0.99:
        log.warning("Neg-day classifier: class imbalance extreme — skipping")
        return None

    val_frac = len(val_daily) / len(combined)
    clf = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.50,   # predict median probability
        max_iter=500, learning_rate=0.05,
        max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=0.1, max_bins=64,
        early_stopping=True, n_iter_no_change=30,
        validation_fraction=val_frac, random_state=42,
    )
    clf.fit(X, y.astype(float))
    train_pos = (clf.predict(X) > 0.5).mean()
    log.info("  best_iter=%d  predicted positive rate=%.1f%%", clf.n_iter_, 100 * train_pos)
    return clf, avail


def train_shape_model(train_sp, val_sp, feat_cols):
    """Train P50 HGBR on intra-day shape deviations (CPI-deflated to real terms)."""
    combined = pd.concat([train_sp, val_sp], ignore_index=True)
    mask     = combined["ssp_shape_target"].notna() & combined[feat_cols].notna().all(axis=1)
    combined = combined[mask]
    X        = combined[feat_cols].values
    # Deflate shape target to real terms (deviation * deflator stays interpretable)
    deflator = combined["cpi_deflator"].values if "cpi_deflator" in combined.columns else 1.0
    y        = combined["ssp_shape_target"].values * deflator
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
# Walk-forward cross-validation
# ─────────────────────────────────────────────────────────────────────────────

# Four seasonal 30-day test windows covering summer / autumn / winter / spring
WALK_FOLDS = [
    ("2025-07-01", "2025-07-30", "summer-2025"),
    ("2025-10-01", "2025-10-30", "autumn-2025"),
    ("2025-12-01", "2025-12-30", "winter-2025"),   # Jan 2026 has CI API gap
    ("2026-04-01", "2026-04-30", "spring-2026"),
]


def _split_at(sp_df, daily_df, fold_start, val_days):
    """Return (sp_train, sp_val, daily_train, daily_val) for data before fold_start."""
    cutoff = fold_start - pd.Timedelta(days=1)
    val_cut = cutoff - pd.Timedelta(days=val_days)
    sp_train = sp_df[sp_df["settlement_datetime"] <= val_cut]
    sp_val   = sp_df[(sp_df["settlement_datetime"] > val_cut) &
                     (sp_df["settlement_datetime"] <= cutoff)]

    feat_cols = get_level_feature_cols(daily_df)
    _opt = {"wind_pct_", "gas_pct_", "cpi_",
            "temp_c_", "wind_ms_", "solar_wm2_", "precip_mm_",
            "heating_", "cooling_"}
    core_feat = [c for c in feat_cols
                 if not any(c.startswith(p) for p in _opt)]
    daily_clean = daily_df[daily_df[core_feat].notna().all(axis=1)].reset_index(drop=True)
    d_train = daily_clean[daily_clean["date"] <= val_cut]
    d_val   = daily_clean[(daily_clean["date"] > val_cut) & (daily_clean["date"] <= cutoff)]
    return sp_train, sp_val, d_train, d_val


def walk_forward_eval(sp_df: pd.DataFrame, daily_df: pd.DataFrame,
                      val_days: int = VAL_DAYS) -> pd.DataFrame:
    """
    Walk-forward cross-validation across seasonal folds.

    For each fold the model is re-trained on all data strictly before the
    fold's start date (minus val_days for early-stopping), then evaluated on
    the 30-day fold window.  Production models are NOT overwritten.
    """
    shape_sp_full = build_shape_data(sp_df, daily_df)
    shape_feat_cols = get_shape_feature_cols(build_shape_data(sp_df, daily_df))
    level_feat_cols = get_level_feature_cols(daily_df)

    all_fold_preds = []

    for fold_start_str, fold_end_str, label in WALK_FOLDS:
        fold_start = pd.Timestamp(fold_start_str)
        fold_end   = pd.Timestamp(fold_end_str)

        # Skip folds whose end is beyond the available data
        if fold_end > sp_df["settlement_datetime"].max():
            log.info("Skipping fold %s — data ends before %s", label, fold_end_str)
            continue

        log.info("── Fold %s  (%s → %s) ──", label, fold_start_str, fold_end_str)

        sp_train, sp_val, d_train, d_val = _split_at(sp_df, daily_df, fold_start, val_days)
        log.info("  Train: %d rows  Val: %d rows", len(sp_train), len(sp_val))

        shape_train = build_shape_data(sp_train, daily_df)
        shape_val   = build_shape_data(sp_val,   daily_df)

        level_models = train_level_models(d_train, d_val, level_feat_cols)
        shape_model  = train_shape_model(shape_train, shape_val, shape_feat_cols)

        # Test slice
        d_test  = daily_df[(daily_df["date"] >= fold_start) & (daily_df["date"] <= fold_end)]
        sp_test = sp_df[(sp_df["settlement_datetime"] >= fold_start) &
                        (sp_df["settlement_datetime"] <= fold_end + pd.Timedelta(hours=23, minutes=30))]
        sh_test = shape_sp_full[(shape_sp_full["settlement_datetime"] >= fold_start) &
                                (shape_sp_full["settlement_datetime"] <= fold_end + pd.Timedelta(hours=23, minutes=30))]

        pred = evaluate_phase3(level_models, shape_model,
                               d_test, sp_test, sh_test,
                               level_feat_cols, shape_feat_cols)
        pred["fold"] = label
        all_fold_preds.append(pred)

        m = metrics(pred["ssp_actual"].values, pred["ssp_predicted"].values)
        dc = decomposition_metrics(pred)
        log.info("  MAE=£%.2f  sMAPE=%.1f%%  LevelMAE=£%.2f  ShapeCorr=%.3f  PeakTiming=%.1f SPs",
                 m["MAE"], m["sMAPE"], dc["level_mae"], dc["shape_corr_mean"], dc["peak_timing_mae"])

    if not all_fold_preds:
        log.warning("No folds completed — check data date range")
        return pd.DataFrame()

    combined = pd.concat(all_fold_preds, ignore_index=True)
    return combined


def _print_walk_forward_report(combined: pd.DataFrame) -> None:
    """Print per-fold and aggregate walk-forward metrics."""
    print("\n" + "─" * 68)
    print(f"  Walk-forward cross-validation — {combined['fold'].nunique()} seasonal folds")
    print("─" * 68)
    print(f"  {'Fold':<16} {'Days':>4}  {'MAE':>8}  {'sMAPE':>7}  {'LvlMAE':>8}  {'ShapeR':>7}")
    print("─" * 68)
    for fold_label in combined["fold"].unique():
        f = combined[combined["fold"] == fold_label]
        m  = metrics(f["ssp_actual"].values, f["ssp_predicted"].values)
        dc = decomposition_metrics(f)
        n_days = f["settlement_date"].nunique()
        print(f"  {fold_label:<16} {n_days:>4}  £{m['MAE']:>7.2f}  {m['sMAPE']:>6.1f}%"
              f"  £{dc['level_mae']:>7.2f}  {dc['shape_corr_mean']:>7.3f}")

    print("─" * 68)
    m_all  = metrics(combined["ssp_actual"].values, combined["ssp_predicted"].values)
    dc_all = decomposition_metrics(combined)
    n_days_all = combined["settlement_date"].nunique()
    print(f"  {'AGGREGATE':<16} {n_days_all:>4}  £{m_all['MAE']:>7.2f}  {m_all['sMAPE']:>6.1f}%"
          f"  £{dc_all['level_mae']:>7.2f}  {dc_all['shape_corr_mean']:>7.3f}")
    print("─" * 68)
    print(f"  RMSE (aggregate): £{m_all['RMSE']:.2f}/MWh   n={len(combined)} periods")
    print("─" * 68)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",     default=str(FEATURES_FILE))
    parser.add_argument("--test-days",    type=int, default=TEST_DAYS)
    parser.add_argument("--val-days",     type=int, default=VAL_DAYS)
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run seasonal walk-forward CV instead of standard holdout")
    parser.add_argument("--train-years", type=int, default=TRAIN_YEARS,
                        help="Max training history in years (default 3)")
    args = parser.parse_args()

    # ── Load SP-level data ────────────────────────────────────────────────
    log.info("Loading SP-level features from %s …", args.features)
    sp_df = load_sp_data(Path(args.features))

    # ── Build daily level dataset ─────────────────────────────────────────
    log.info("Building daily level dataset …")
    daily_df = build_level_dataset(sp_df)
    level_feat_cols = get_level_feature_cols(daily_df)
    log.info("Level features: %d", len(level_feat_cols))

    # ── Walk-forward mode: retrain per fold, do not overwrite production models
    if args.walk_forward:
        log.info("Running seasonal walk-forward cross-validation …")
        combined = walk_forward_eval(sp_df, daily_df, val_days=args.val_days)
        if not combined.empty:
            _print_walk_forward_report(combined)
            wf_path = ASSETS_DIR / "walk_forward_predictions.csv"
            combined.to_csv(wf_path, index=False)
            log.info("Walk-forward predictions → %s", wf_path)
        return

    # ── Standard holdout training ─────────────────────────────────────────
    sp_train, sp_val, sp_test = split_dates(sp_df, args.test_days, args.val_days, args.train_years)
    daily_train, daily_val, daily_test = split_daily(daily_df, args.test_days, args.val_days, args.train_years)

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
    log.info("Training negative-price regime classifier …")
    _clf_result = train_neg_day_classifier(daily_train, daily_val)
    neg_clf, neg_clf_feats = (_clf_result if _clf_result else (None, []))
    if neg_clf:
        joblib.dump(neg_clf, ASSETS_DIR / "neg_day_classifier.pkl")
        with open(ASSETS_DIR / "neg_day_classifier_feats.json", "w") as _f:
            json.dump(neg_clf_feats, _f)
        log.info("Neg-day classifier saved → %s", ASSETS_DIR / "neg_day_classifier.pkl")

    # ── Add classifier predictions as level feature ───────────────────────
    # Use in-sample predictions on train+val so the level model can learn to
    # use neg_price_risk_prob as a signal during training.
    if neg_clf and neg_clf_feats:
        for _dset in [daily_train, daily_val, daily_test]:
            _avail = [c for c in neg_clf_feats if c in _dset.columns]
            if _avail:
                _x = _dset[_avail].fillna(0).values
                _dset["neg_price_risk_prob"] = neg_clf.predict(_x)
        if "neg_price_risk_prob" not in level_feat_cols:
            level_feat_cols = level_feat_cols + ["neg_price_risk_prob"]
            log.info("  Added neg_price_risk_prob to level features (%d total)", len(level_feat_cols))

    log.info("Training level models (P10/P50/P90) on daily SSP mean …")
    level_models = train_level_models(daily_train, daily_val, level_feat_cols)

    log.info("Training shape model (P50) on intra-day deviations …")
    shape_model = train_shape_model(shape_sp_train, shape_sp_val, shape_feat_cols)

    # ── H+2 shape model ───────────────────────────────────────────────────
    log.info("Building H+2 shape dataset (lag ≥ 96 features only) …")
    shape_h2_all   = build_shape_data_h2(sp_df,    daily_df)
    shape_h2_train = build_shape_data_h2(sp_train, daily_df)
    shape_h2_val   = build_shape_data_h2(sp_val,   daily_df)
    shape_h2_test  = build_shape_data_h2(sp_test,  daily_df)
    shape_h2_feat_cols = get_shape_feature_cols_h2(shape_h2_all)
    log.info("H+2 shape features: %d", len(shape_h2_feat_cols))
    log.info("Training H+2 shape model (P50) …")
    shape_h2_model = train_shape_model(shape_h2_train, shape_h2_val, shape_h2_feat_cols)

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
    joblib.dump(shape_model,    ASSETS_DIR / "shape_q50.pkl")
    joblib.dump(shape_h2_model, ASSETS_DIR / "shape_h2_q50.pkl")
    log.info("Models saved → %s", ASSETS_DIR)

    with open(ASSETS_DIR / "level_feature_cols.json", "w") as f:
        json.dump(level_feat_cols, f)
    with open(ASSETS_DIR / "shape_feature_cols.json", "w") as f:
        json.dump(shape_feat_cols, f)
    with open(ASSETS_DIR / "shape_h2_feature_cols.json", "w") as f:
        json.dump(shape_h2_feat_cols, f)

    pred_df.to_csv(ASSETS_DIR / "test_predictions_phase3.csv", index=False)
    log.info("Test predictions → %s", ASSETS_DIR / "test_predictions_phase3.csv")

    # ── Feature importance ────────────────────────────────────────────────────
    # Level model: Spearman |correlation| with target over full training set.
    #   Permutation importance is unreliable with O(1000) daily rows; Spearman
    #   rank correlation is a stable proxy for feature-target association.
    # Shape model: permutation importance on val set (144 SP rows, well-behaved).
    log.info("Computing feature importances …")
    try:
        from scipy.stats import spearmanr
        _lvl_all = pd.concat([daily_train, daily_val], ignore_index=True)
        _defl    = _lvl_all["cpi_deflator"].values if "cpi_deflator" in _lvl_all.columns else 1.0
        _y_lvl   = _lvl_all[LEVEL_TARGET].values * _defl
        _corrs, _stds = [], []
        for _fc in level_feat_cols:
            _x = _lvl_all[_fc].values
            _ok = ~(np.isnan(_x) | np.isnan(_y_lvl))
            if _ok.sum() > 5:
                _r, _ = spearmanr(_x[_ok], _y_lvl[_ok])
                _corrs.append(abs(float(_r)))
            else:
                _corrs.append(0.0)
            _stds.append(0.0)
        pd.DataFrame({
            "feature":         level_feat_cols,
            "importance_mean": _corrs,
            "importance_std":  _stds,
        }).to_csv(ASSETS_DIR / "phase3_level_importance.csv", index=False)

        # Shape model: permutation importance on val set
        _shp_mask = shape_sp_val["ssp_shape_target"].notna() & \
                    shape_sp_val[shape_feat_cols].notna().all(axis=1)
        _shp_val  = shape_sp_val[_shp_mask]
        if len(_shp_val) > 0:
            _deflshp = _shp_val["cpi_deflator"].values if "cpi_deflator" in _shp_val.columns else 1.0
            _pi_shp = permutation_importance(
                shape_model,
                _shp_val[shape_feat_cols].values,
                _shp_val["ssp_shape_target"].values * _deflshp,
                n_repeats=5, random_state=42, n_jobs=-1,
            )
            pd.DataFrame({
                "feature":         shape_feat_cols,
                "importance_mean": _pi_shp.importances_mean,
                "importance_std":  _pi_shp.importances_std,
            }).to_csv(ASSETS_DIR / "phase3_shape_importance.csv", index=False)
        log.info("Feature importances saved → %s", ASSETS_DIR)
    except Exception as _e:
        log.warning("Feature importance computation failed: %s", _e)

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

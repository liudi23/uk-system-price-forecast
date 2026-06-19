"""
Weekly production-monitoring report for the Phase 3 UK SSP forecasting system.

Read-only over production outputs. No model changes.

Sections
--------
§1  PI Coverage — overall (WF backtest baseline vs live)
§2  PI Coverage by price bucket
§3  Ex-ante spike-risk-flag conditional coverage
§4  Kalman state health (post-correction residuals)
§5  Spike widening monitor    [INACTIVE while spike_widening: false]
§6  Classifier calibration    [INACTIVE while spike_widening: false]
§7  Step-3 readiness gate     [counts usable spike-bearing training autumns]

Data sources
------------
WF backtest  : model_assets/walk_forward_predictions.csv  (119 days, 2025-07 to 2026-04)
Live forecasts: model_assets/forecasts/forecast_phase3_YYYY-MM-DD.csv
Settled actuals: data/raw/system_prices.csv  (rolling ~60-day window)
PI calibration : model_assets/pi_calibration_v1.json
Spike classifier eval: model_assets/spike_classifier_v1_eval.json
Features (training): data/processed/features_5yr.csv   (Step-3 gate)
Features (recent) : data/processed/features_recent.csv (live spike_risk_flag)
Corrector config  : model_assets/corrector_config.json

Usage
-----
    python src/monitoring/build_monitoring_report.py [--week 2026-W25] [--out reports/monitoring/]
"""

import argparse
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "model_assets"
DATA_DIR   = ROOT / "data"
NOWCAST_BANDS_JSON_PATH = ASSETS_DIR / "nowcast_bands.json"

# ── Thresholds ─────────────────────────────────────────────────────────────────
COV_TARGET     = 0.798   # WF+PI calibration target coverage
COV_WARN_LO    = 0.720   # flag if below
COV_WARN_HI    = 0.920   # flag if above (over-covering → width waste)
COV_MIN_N      = 200     # suppress 🔴/🟢 flag below this SP-row count
COV_MIN_BUCKET = 30      # per-bucket minimum N for flags

RESID_WARN     = 10.0    # £/MWh winsorised-mean residual WARN threshold
RESID_ALERT    = 20.0    # £/MWh ALERT threshold
RESID_WIN_PCTS = (0.05, 0.05)  # winsorise at 5%/95%

PRICE_BUCKET_BINS   = [-np.inf, 85.0, 120.0, 150.0, 250.0, np.inf]
PRICE_BUCKET_LABELS = ["<£85", "£85-120", "£120-150", "£150-250", "£250+"]

# spike_risk_flag formula constants (mirrors join_derivation_code.py)
ELEVATED_THR       = 120.0
ELEVATED_COUNT_MIN = 5
WIND_LOW_THR       = 10.0
HIGH_RISK_MONTHS   = {9, 10, 11, 3, 4, 5}

# Step-3 gate constants
CRISIS_YEARS   = {2021, 2022}  # energy crisis — spike rates 95-100%, misleading tail
EXCLUDED_YEARS = {2023}        # post-crisis transition, not regime-weighted in training
QUIET_ANNUAL   = 0.10          # annual spike rate < this → "too quiet" (2024 = 7.7%)
STEP3_MIN_AUTUMNS = 2          # GREEN when ≥ this many usable spike-bearing autumns

SPIKE_THR_TRAIN = 150.0        # £/MWh threshold for "spike day" in training data

# Nowcast band monitoring
NOWCAST_COV_TARGET       = 0.80   # nominal 80% target
NOWCAST_COV_WARN_LO      = 0.78   # flag 🔴 if overall coverage drifts below this
NOWCAST_BANDS_STALE_DAYS = 31     # flag 🟡 if nowcast_bands.json older than this
NOWCAST_H3_MIN_MONTHS    = 6      # H+3 DA-feature readiness: months of forecast archive


# ── Wilson confidence interval ─────────────────────────────────────────────────

def wilson_ci(k: int, n: int, alpha: float = 0.10):
    """90% Wilson CI for a binomial proportion p = k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.6449  # z_{0.95} for 90% CI (two-sided)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _status(value, warn_lo, warn_hi, n, min_n, invert=False):
    """Return a status emoji. Marks low-confidence when n < min_n."""
    if n < min_n:
        return "⚠️"
    if invert:
        ok = warn_lo <= abs(value) < warn_hi
        return "🟡" if warn_lo <= abs(value) < warn_hi else ("🟢" if abs(value) < warn_lo else "🔴")
    if np.isnan(value):
        return "⚠️"
    if value < warn_lo:
        return "🔴"
    if value > warn_hi:
        return "🟡"
    return "🟢"


def _resid_status(v, n, min_n=COV_MIN_N):
    if n < min_n:
        return "⚠️"
    if np.isnan(v):
        return "⚠️"
    av = abs(v)
    if av >= RESID_ALERT:
        return "🔴"
    if av >= RESID_WARN:
        return "🟡"
    return "🟢"


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pi_calibration():
    """Return dict {sp_int: delta_float} from pi_calibration_v1.json."""
    pi_path = ASSETS_DIR / "pi_calibration_v1.json"
    if not pi_path.exists():
        return {}
    with open(pi_path) as f:
        pi = json.load(f)
    return {int(k): float(v) for k, v in pi.get("delta_by_sp", {}).items()}


def load_wf_with_pi(delta_map: dict) -> pd.DataFrame:
    """
    Load walk_forward_predictions.csv and apply PI calibration to raw q10/q90.
    Returns columns: date, sp, q10, q90, q50, actual, spike_risk_flag, period="backtest".
    """
    wf_path = ASSETS_DIR / "walk_forward_predictions.csv"
    if not wf_path.exists():
        return pd.DataFrame()
    wf = pd.read_csv(wf_path, usecols=[
        "settlement_date", "settlement_period",
        "ssp_q10", "ssp_q50", "ssp_q90", "ssp_actual", "spike_risk_flag",
    ])
    wf["settlement_period"] = wf["settlement_period"].astype(int)
    wf["settlement_date"]   = pd.to_datetime(wf["settlement_date"]).dt.date

    if delta_map:
        delta_arr = wf["settlement_period"].map(delta_map).fillna(0.0)
        wf["q10"] = wf["ssp_q10"] - delta_arr
        wf["q90"] = wf["ssp_q90"] + delta_arr
    else:
        wf["q10"] = wf["ssp_q10"]
        wf["q90"] = wf["ssp_q90"]

    wf["q50"]    = wf["ssp_q50"]
    wf["actual"] = pd.to_numeric(wf["ssp_actual"], errors="coerce")
    wf["period"] = "backtest"
    wf["is_forecast"] = True  # all WF rows are forecasts (no splicing)
    return wf[["settlement_date", "settlement_period", "q10", "q90", "q50",
                "actual", "spike_risk_flag", "period", "is_forecast"]].rename(
        columns={"settlement_date": "date", "settlement_period": "sp"}
    ).dropna(subset=["actual"])


def load_system_prices() -> pd.DataFrame:
    """Load data/raw/system_prices.csv; returns (date, sp, actual) indexed."""
    sp_path = DATA_DIR / "raw" / "system_prices.csv"
    if not sp_path.exists():
        return pd.DataFrame(columns=["date", "sp", "actual", "price_derivation_code"])
    sp = pd.read_csv(sp_path, usecols=["settlement_date", "settlement_period",
                                        "ssp", "price_derivation_code"])
    sp["date"]   = pd.to_datetime(sp["settlement_date"]).dt.date
    sp["sp"]     = sp["settlement_period"].astype(int)
    sp["actual"] = pd.to_numeric(sp["ssp"], errors="coerce")
    return sp[["date", "sp", "actual", "price_derivation_code"]].dropna(subset=["actual"])


def load_live_data(system_prices: pd.DataFrame, delta_map=None) -> pd.DataFrame:
    """
    Load all forecast_phase3_*.csv from model_assets/forecasts/.
    Join with system_prices for actual values.

    The forecast CSVs store PI-calibrated q10/q90 (backfilled 2026-06-17).
    delta_map is accepted for API compatibility but ignored — PI calibration
    has already been applied to the CSV files.

    Handles two file schemas:
      - WITH is_actual column: filter to is_actual=False (Kalman-corrected unsettled SPs)
      - WITHOUT is_actual column: use all 48 rows (PI-calibrated daily forecast)
    Returns columns: date, sp, q10, q90, q50, actual, is_forecast, file_type, period="live".
    """
    fc_dir = ASSETS_DIR / "forecasts"
    if not fc_dir.exists():
        return pd.DataFrame()

    actuals_idx = system_prices.set_index(["date", "sp"])[["actual"]]
    rows = []
    n_daily   = 0
    n_intraday = 0

    for fc_path in sorted(fc_dir.glob("forecast_phase3_*.csv")):
        date_str = fc_path.stem.replace("forecast_phase3_", "")
        try:
            fc_date = date.fromisoformat(date_str)
        except ValueError:
            continue

        try:
            fc = pd.read_csv(fc_path)
        except Exception:
            continue

        has_is_actual = "is_actual" in fc.columns
        if has_is_actual:
            # Keep only genuinely forecast rows (not spliced with actuals)
            fc = fc[fc["is_actual"] == False].copy()
            file_type = "intraday"
            n_intraday += 1
        else:
            file_type = "daily"
            n_daily += 1

        fc["sp"]  = fc["settlement_period"].astype(int)
        fc["date"] = fc_date
        fc["q50"] = pd.to_numeric(fc["ssp_q50"], errors="coerce")
        fc["q10"] = pd.to_numeric(fc["ssp_q10"], errors="coerce")
        fc["q90"] = pd.to_numeric(fc["ssp_q90"], errors="coerce")
        fc["file_type"] = file_type
        fc["is_forecast"] = True

        # Join actuals
        fc = fc.join(actuals_idx, on=["date", "sp"], how="left")
        fc = fc.dropna(subset=["actual", "q10", "q90", "q50"])

        rows.append(fc[["date", "sp", "q10", "q90", "q50",
                          "actual", "is_forecast", "file_type"]])

    if not rows:
        return pd.DataFrame()

    live = pd.concat(rows, ignore_index=True)
    live["period"]  = "live"
    live["spike_risk_flag"] = False  # will be filled in by compute_spike_risk_live
    print(f"  Live data: {len(live)} SP-rows from {live['date'].nunique()} dates "
          f"({n_daily} daily-snapshot files, {n_intraday} intraday-updated files)")
    return live


def compute_spike_risk_live(
    live: pd.DataFrame,
    system_prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recompute ex-ante spike_risk_flag for live dates using D-1 actuals from
    system_prices.csv and wind_pct_lag_48 from features_recent.csv.
    Updates live["spike_risk_flag"] in-place (by date).
    """
    if live.empty:
        return live

    live = live.copy()

    # ── elevated_count_lag1d from system_prices ───────────────────────────────
    # For forecast date D: count SPs on D-1 with actual > ELEVATED_THR
    daily_elev = (
        system_prices.groupby("date")["actual"]
        .apply(lambda x: (x > ELEVATED_THR).sum())
        .reset_index(name="n_elevated")
    )
    daily_elev["date"] = pd.to_datetime(daily_elev["date"]).dt.date
    # Shift forward: lag1d_date = D → use n_elevated from D-1
    lag1d = {
        (pd.to_datetime(d).date() + timedelta(days=1)): int(n)
        for d, n in zip(daily_elev["date"], daily_elev["n_elevated"])
    }

    # ── wind_pct_daily_mean_lag1d from features_recent ────────────────────────
    feat_path = DATA_DIR / "processed" / "features_recent.csv"
    wind_lag1d = {}
    if feat_path.exists():
        feat = pd.read_csv(feat_path, usecols=["settlement_date", "wind_pct_lag_48"])
        feat["settlement_date"] = pd.to_datetime(feat["settlement_date"]).dt.date
        # wind_pct_lag_48 at row (date D, sp h) = wind_pct on (D-1, h)
        # Mean per day D = D-1 actual mean wind
        wind_lag1d = (
            feat.groupby("settlement_date")["wind_pct_lag_48"]
            .mean()
            .to_dict()
        )

    # ── Apply flag formula ────────────────────────────────────────────────────
    live_dates = live["date"].unique()
    date_flags = {}
    for d in live_dates:
        month = d.month
        elev_cnt  = lag1d.get(d, 0)
        wind_mean = wind_lag1d.get(d, 100.0)  # default 100% (not low-wind) if missing
        flag = bool(
            (elev_cnt >= ELEVATED_COUNT_MIN)
            or (wind_mean < WIND_LOW_THR and month in HIGH_RISK_MONTHS)
        )
        date_flags[d] = flag

    live["spike_risk_flag"] = live["date"].map(date_flags).fillna(False)
    return live


# ── Coverage metrics ───────────────────────────────────────────────────────────

def _covered(df: pd.DataFrame, q10_col="q10", q90_col="q90", actual_col="actual"):
    return (df[actual_col] >= df[q10_col]) & (df[actual_col] <= df[q90_col])


def compute_coverage(df: pd.DataFrame) -> dict:
    """Overall PI coverage + Wilson 90% CI."""
    sub = df.dropna(subset=["q10", "q90", "actual"])
    n   = len(sub)
    if n == 0:
        return {"n": 0, "coverage": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    k = _covered(sub).sum()
    cov = float(k) / n
    lo, hi = wilson_ci(k, n)
    return {"n": n, "coverage": round(cov * 100, 2),
            "ci_lo": round(lo * 100, 2), "ci_hi": round(hi * 100, 2)}


def compute_bucket_coverage(df: pd.DataFrame) -> dict:
    """Coverage per price bucket."""
    sub = df.dropna(subset=["q10", "q90", "actual"])
    sub = sub.copy()
    sub["bucket"] = pd.cut(sub["actual"], bins=PRICE_BUCKET_BINS, labels=PRICE_BUCKET_LABELS)
    result = {}
    for b in PRICE_BUCKET_LABELS:
        bsub = sub[sub["bucket"] == b]
        result[b] = compute_coverage(bsub)
    return result


def compute_residuals(df: pd.DataFrame) -> dict:
    """
    Post-correction residuals (actual − q50) for forecast rows.
    Returns winsorised mean and median + counts.
    """
    sub = df.dropna(subset=["q50", "actual"])
    n = len(sub)
    if n == 0:
        return {"n": 0, "win_mean": float("nan"), "median": float("nan"),
                "n_intraday": 0, "n_daily": 0}

    resid = (sub["actual"] - sub["q50"]).values
    # Winsorise at 5%/95%
    lo_q, hi_q = np.quantile(resid, [RESID_WIN_PCTS[0], 1 - RESID_WIN_PCTS[1]])
    clipped = np.clip(resid, lo_q, hi_q)
    win_mean = float(np.mean(clipped))
    median   = float(np.median(resid))

    n_intraday = int((sub.get("file_type", pd.Series(["daily"] * n)) == "intraday").sum())
    n_daily    = n - n_intraday
    return {
        "n":          n,
        "win_mean":   round(win_mean, 2),
        "median":     round(median, 2),
        "n_intraday": n_intraday,
        "n_daily":    n_daily,
    }


def compute_daily_xhat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily mean(actual − q50) per forecast date — the 'filter is working hard' signal.
    Returns DataFrame with columns: date, xhat_proxy, n.
    """
    sub = df.dropna(subset=["q50", "actual"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["date", "xhat_proxy", "n"])
    sub["resid"] = sub["actual"] - sub["q50"]
    daily = sub.groupby("date").agg(
        xhat_proxy=("resid", "mean"),
        n=("resid", "count"),
    ).reset_index()
    return daily


# ── Step-3 readiness gate ──────────────────────────────────────────────────────

def compute_step3_gate() -> dict:
    """
    Counts usable spike-bearing training autumns (Sep–Nov) from features_5yr.csv.

    Exclusions (hard-coded per user instruction):
      CRISIS_YEARS   = {2021, 2022}  — annual spike rate > 95%, crisis period
      EXCLUDED_YEARS = {2023}        — post-crisis transition, not regime-weighted
      QUIET_ANNUAL   < 0.10          — annual spike rate too quiet (2024 = 7.7%)
    GREEN: ≥ 2 usable spike-bearing autumns.
    """
    feat_path = DATA_DIR / "processed" / "features_5yr.csv"
    if not feat_path.exists():
        return {"error": "features_5yr.csv not found", "n_usable_autumns": 0, "green": False}

    feat = pd.read_csv(feat_path, usecols=["settlement_date", "ssp_raw"])
    feat["settlement_date"] = pd.to_datetime(feat["settlement_date"])
    feat["year"]     = feat["settlement_date"].dt.year
    feat["month"]    = feat["settlement_date"].dt.month
    feat["is_autumn"]= feat["month"].isin([9, 10, 11])
    feat["is_spike"] = feat["ssp_raw"] > SPIKE_THR_TRAIN

    daily = feat.groupby("settlement_date").agg(
        year=("year", "first"), month=("month", "first"),
        is_autumn=("is_autumn", "first"), spike_day=("is_spike", "any"),
    ).reset_index()

    # Annual spike rate per year
    annual = daily.groupby("year").agg(
        n_days=("spike_day", "count"), spike_days=("spike_day", "sum"),
    )
    annual["annual_rate"] = annual["spike_days"] / annual["n_days"]

    # Autumn spike rate per year
    autumn = daily[daily["is_autumn"]].groupby("year").agg(
        n_autumn=("spike_day", "count"), spike_autumn=("spike_day", "sum"),
    )
    autumn["autumn_rate"] = autumn["spike_autumn"] / autumn["n_autumn"]

    summary = annual.join(autumn, how="outer")
    summary["annual_rate"]  = summary["annual_rate"].fillna(0)
    summary["autumn_rate"]  = summary["autumn_rate"].fillna(0)

    usable_autumns = []
    all_autumns    = []
    year_statuses  = {}
    for yr, row in summary.iterrows():
        ar = float(row["annual_rate"])
        n_aut      = row.get("n_autumn", 0)
        n_aut      = 0 if (n_aut is None or (isinstance(n_aut, float) and np.isnan(n_aut))) else int(n_aut)
        aut_rate   = row.get("autumn_rate", 0.0)
        aut_rate   = 0.0 if (aut_rate is None or (isinstance(aut_rate, float) and np.isnan(aut_rate))) else float(aut_rate)

        # Classify: years without any autumn data are "pending" (autumn hasn't happened yet)
        if n_aut == 0:
            status = "pending"
        elif yr in CRISIS_YEARS:
            status = "crisis"
        elif yr in EXCLUDED_YEARS:
            status = "transition"
        elif ar < QUIET_ANNUAL:
            status = "quiet"
        else:
            status = "usable"

        year_statuses[int(yr)] = {
            "status":        status,
            "annual_rate":   round(ar, 4),
            "autumn_rate":   round(aut_rate, 4),
            "n_autumn_days": n_aut,
        }
        if n_aut > 0:
            all_autumns.append(int(yr))
            if status == "usable":
                usable_autumns.append(int(yr))

    train_start = feat["settlement_date"].min().date()
    train_end   = feat["settlement_date"].max().date()
    # earliest year that has autumn data (any status)
    years_with_autumn = [y for y, v in year_statuses.items() if v["n_autumn_days"] > 0]
    usable_start = date(min(years_with_autumn, default=date.today().year), 1, 1)

    n_usable = len(usable_autumns)
    green    = n_usable >= STEP3_MIN_AUTUMNS

    return {
        "train_start":      str(train_start),
        "train_end":        str(train_end),
        "usable_start":     str(usable_start),
        "all_autumns":      all_autumns,
        "usable_autumns":   usable_autumns,
        "n_usable_autumns": n_usable,
        "min_autumns":      STEP3_MIN_AUTUMNS,
        "green":            green,
        "year_statuses":    year_statuses,
        "trigger":          "autumn 2026 settled (est. Dec 2026)",
    }


# ── Nowcast band monitoring ────────────────────────────────────────────────────

def load_nowcast_bands() -> dict:
    """Load model_assets/nowcast_bands.json; return {} if absent."""
    if not NOWCAST_BANDS_JSON_PATH.exists():
        return {}
    with open(NOWCAST_BANDS_JSON_PATH) as f:
        return json.load(f)


def compute_nowcast_coverage(system_prices: pd.DataFrame, bands: dict) -> dict:
    """
    Compute live coverage of the shipped persistence-nowcast bands from settled actuals.

    For each SP at time t (y0 = actual[t], regime = price_derivation_code[t]) and each
    horizon h ∈ {1,2,3}, check whether actual[t+h] − y0 ∈ [P10_h, P90_h] from the
    regime-matched band.  Mirrors the residual definition in build_nowcast_bands.py.

    Uses system_prices.csv (~60-day rolling window).  Returns dict shaped as
    {h1: {overall, NP, EN}, h2: …, h3: …} or {} if data or bands are unavailable.
    """
    if system_prices.empty or not bands or "horizons" not in bands:
        return {}

    sp      = system_prices.sort_values(["date", "sp"]).reset_index(drop=True)
    actuals = sp["actual"].values
    codes   = (sp["price_derivation_code"].values
               if "price_derivation_code" in sp.columns
               else np.full(len(sp), "?"))

    result = {}
    for h in [1, 2, 3]:
        h_key = f"h{h}"
        if h_key not in bands["horizons"]:
            continue
        h_bands = bands["horizons"][h_key]
        n = len(sp) - h
        if n <= 0:
            continue

        y0   = actuals[:n]
        yh   = actuals[h : h + n]
        c0   = codes[:n]
        valid = np.isfinite(y0) & np.isfinite(yh)
        r_h  = yh - y0

        sub = {}
        for regime_key, mask in [
            ("overall", valid),
            ("NP",      valid & (c0 == "N")),
            ("EN",      valid & (c0 != "N")),   # P/K-code → EN regime
        ]:
            bnd = h_bands.get(regime_key, h_bands["overall"])
            p10, p90 = bnd["p10"], bnd["p90"]
            r_sub = r_h[mask]
            n_sub = int(mask.sum())
            hits  = int(((r_sub >= p10) & (r_sub <= p90)).sum())
            cov   = round(hits / n_sub * 100, 1) if n_sub > 0 else float("nan")
            lo, hi = wilson_ci(hits, n_sub) if n_sub > 0 else (float("nan"), float("nan"))
            sub[regime_key] = {
                "coverage": cov, "n": n_sub, "hits": hits,
                "ci_lo": round(lo * 100, 1), "ci_hi": round(hi * 100, 1),
            }
        result[h_key] = sub

    return result


def check_nowcast_bands_staleness(bands: dict) -> dict:
    """Return staleness metadata for nowcast_bands.json."""
    if not bands:
        return {"generated": None, "age_days": None, "stale": True, "status": "🔴"}
    gen_str = bands.get("generated", "")
    if not gen_str:
        return {"generated": None, "age_days": None, "stale": True, "status": "🔴"}
    try:
        gen_date = date.fromisoformat(gen_str)
    except ValueError:
        return {"generated": gen_str, "age_days": None, "stale": True, "status": "🔴"}
    age   = (date.today() - gen_date).days
    stale = age > NOWCAST_BANDS_STALE_DAYS
    return {
        "generated": gen_str,
        "age_days":  age,
        "stale":     stale,
        "status":    "🟡" if stale else "🟢",
    }


def compute_nowcast_h3_readiness() -> dict:
    """
    Count calendar months of archived forecast_phase3_*.csv files.
    GREEN when ≥ NOWCAST_H3_MIN_MONTHS months are present — the threshold before the
    DA-Q50 feature improvement (partial R² ~10% at h+3, Experiment 2) can be validated
    out-of-sample and merged into production.
    """
    fc_dir = ASSETS_DIR / "forecasts"
    if not fc_dir.exists():
        return {"n_months": 0, "min_months": NOWCAST_H3_MIN_MONTHS,
                "green": False, "first_date": None, "last_date": None, "trigger": "—"}

    dates = []
    for p in fc_dir.glob("forecast_phase3_*.csv"):
        try:
            dates.append(date.fromisoformat(p.stem.replace("forecast_phase3_", "")))
        except ValueError:
            pass

    if not dates:
        return {"n_months": 0, "min_months": NOWCAST_H3_MIN_MONTHS,
                "green": False, "first_date": None, "last_date": None, "trigger": "—"}

    first_date = min(dates)
    last_date  = max(dates)
    n_months   = len(set((d.year, d.month) for d in dates))
    green      = n_months >= NOWCAST_H3_MIN_MONTHS

    # Estimate: the calendar month that is NOWCAST_H3_MIN_MONTHS after first_date
    y, m = first_date.year, first_date.month
    for _ in range(NOWCAST_H3_MIN_MONTHS - 1):
        m += 1
        if m > 12:
            m, y = 1, y + 1
    trigger_str = date(y, m, 1).strftime("%b %Y")

    return {
        "n_months":   n_months,
        "min_months": NOWCAST_H3_MIN_MONTHS,
        "green":      green,
        "first_date": str(first_date),
        "last_date":  str(last_date),
        "trigger":    f"≈ {trigger_str} ({NOWCAST_H3_MIN_MONTHS} calendar months archived)",
    }


def compute_pipeline_health() -> dict:
    """
    Assess intraday pipeline health as of report-generation time.

    Returns
    -------
    dict with keys:
      last_update        - ISO string from kalman_state.json, or None
      forecast_date      - ISO string, or None
      n_settled          - int, SPs settled on the last Kalman update
      x_hat              - float, last Kalman state estimate
      age_hours          - float, hours since last_update (None if unknown)
      days_with_actuals  - int, archive days that have at least one is_actual=True row
      days_total         - int, total archive days
      status             - 'ok' | 'stale' | 'unknown'
      status_emoji       - '🟢' | '🔴' | '⚠️'
    """
    from datetime import datetime, timezone

    result: dict = {
        "last_update": None,
        "forecast_date": None,
        "n_settled": 0,
        "x_hat": float("nan"),
        "age_hours": None,
        "days_with_actuals": 0,
        "days_total": 0,
        "status": "unknown",
        "status_emoji": "⚠️",
    }

    ks_path = ASSETS_DIR / "kalman_state.json"
    if ks_path.exists():
        try:
            with open(ks_path) as _f:
                ks = json.load(_f)
            result["last_update"]   = ks.get("last_update")
            result["forecast_date"] = ks.get("forecast_date")
            result["n_settled"]     = int(ks.get("last_n_settled", 0))
            result["x_hat"]         = float(ks.get("x_hat", float("nan")))
            if result["last_update"]:
                last_dt = datetime.fromisoformat(result["last_update"]).replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                result["age_hours"] = (now_utc - last_dt).total_seconds() / 3600
        except Exception:
            pass

    fc_dir = ASSETS_DIR / "forecasts"
    if fc_dir.exists():
        for p in fc_dir.glob("forecast_phase3_*.csv"):
            result["days_total"] += 1
            try:
                df_tmp = pd.read_csv(p, usecols=lambda c: c == "is_actual")
                if "is_actual" in df_tmp.columns and df_tmp["is_actual"].any():
                    result["days_with_actuals"] += 1
            except Exception:
                pass

    age_h    = result["age_hours"]
    fc_date  = result.get("forecast_date") or ""
    now_utc  = datetime.now(timezone.utc)
    today    = str(now_utc.date())

    if age_h is None:
        result["status"] = "unknown"
        result["status_emoji"] = "⚠️"
    elif age_h <= 4:
        result["status"] = "ok"
        result["status_emoji"] = "🟢"
    elif fc_date == today:
        # Today's forecast exists but Kalman hasn't updated in >4h → true staleness
        result["status"] = "stale"
        result["status_emoji"] = "🔴"
    elif fc_date < today and now_utc.hour >= 14:
        # Daily pipeline did not advance forecast_date to today despite being
        # well past its expected 12:30 UTC + ~1h completion time
        result["status"] = "daily_missed"
        result["status_emoji"] = "🔴"
    else:
        # fc_date < today and hour < 14 → expected overnight/morning gap
        result["status"] = "morning_gap"
        result["status_emoji"] = "🟡"

    return result


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_coverage_timeline(wf: pd.DataFrame, live: pd.DataFrame, out_path: Path) -> None:
    """
    Top panel: 7-day rolling coverage over time with CI ribbon.
    Bottom panel: PI coverage by price bucket (WF+PI vs live bar chart).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # ── Top: rolling coverage timeline ───────────────────────────────────────
    # Combine WF + live into one date-sorted frame
    combined = pd.concat([
        wf[["date", "q10", "q90", "actual", "period"]],
        live[["date", "q10", "q90", "actual", "period"]],
    ], ignore_index=True).dropna(subset=["q10", "q90", "actual"])

    combined["covered"] = _covered(combined).astype(float)
    combined["date"]    = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date")

    # Daily coverage (fraction of SPs covered per date)
    daily_cov = combined.groupby(["date", "period"]).agg(
        k=("covered", "sum"), n=("covered", "count")
    ).reset_index()
    daily_cov["cov"] = daily_cov["k"] / daily_cov["n"]

    # 7-day rolling
    daily_cov = daily_cov.sort_values("date").reset_index(drop=True)
    daily_cov["roll7"] = daily_cov["cov"].rolling(7, min_periods=1).mean()
    daily_cov["roll7_lo"] = daily_cov["k"].rolling(7, min_periods=1).sum() / \
                            daily_cov["n"].rolling(7, min_periods=1).sum()

    for period, color in [("backtest", "#4C72B0"), ("live", "#DD8452")]:
        mask = daily_cov["period"] == period
        sub  = daily_cov[mask]
        if sub.empty:
            continue
        ax1.plot(sub["date"], sub["roll7"] * 100, color=color,
                 lw=1.8, label=f"{period} (7d roll)")
        ax1.fill_between(sub["date"],
                         sub["roll7"].clip(0, 1) * 100 - 3,
                         sub["roll7"].clip(0, 1) * 100 + 3,
                         alpha=0.15, color=color)
        ax1.scatter(sub["date"], sub["cov"] * 100, s=8, alpha=0.4, color=color)

    ax1.axhline(COV_TARGET * 100, ls="--", lw=1.5, color="green", label=f"Target {COV_TARGET*100:.0f}%")
    ax1.axhline(COV_WARN_LO * 100, ls=":", lw=1.2, color="red",   label=f"WARN {COV_WARN_LO*100:.0f}%")
    ax1.axhline(COV_WARN_HI * 100, ls=":", lw=1.2, color="orange",label=f"WARN {COV_WARN_HI*100:.0f}%")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("PI Coverage %")
    ax1.set_title("§1 PI Coverage — Q10–Q90 interval (7-day rolling)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.tick_params(axis="x", rotation=30)

    # ── Bottom: bucket bar chart ──────────────────────────────────────────────
    wf_bucket   = compute_bucket_coverage(wf)
    live_bucket = compute_bucket_coverage(live)

    x   = np.arange(len(PRICE_BUCKET_LABELS))
    w   = 0.35
    for i, b in enumerate(PRICE_BUCKET_LABELS):
        wf_v  = wf_bucket[b]["coverage"]
        liv_v = live_bucket[b]["coverage"]
        wf_n  = wf_bucket[b]["n"]
        liv_n = live_bucket[b]["n"]
        ax2.bar(i - w/2, wf_v, width=w, color="#4C72B0", alpha=0.85,
                label="WF+PI (ref)" if i == 0 else "")
        ax2.bar(i + w/2, liv_v if not np.isnan(liv_v) else 0, width=w,
                color="#DD8452", alpha=0.85 if liv_n >= COV_MIN_BUCKET else 0.3,
                label="Live" if i == 0 else "")
        if not np.isnan(wf_v):
            lo, hi = wilson_ci(int(wf_v/100 * wf_n), wf_n, alpha=0.20)
            ax2.errorbar(i - w/2, wf_v, yerr=[[wf_v - lo*100], [hi*100 - wf_v]],
                         fmt="none", color="black", capsize=4, lw=1.2)
        if not np.isnan(liv_v) and liv_n >= 1:
            lo, hi = wilson_ci(int(liv_v/100 * liv_n), liv_n, alpha=0.20)
            ax2.errorbar(i + w/2, liv_v, yerr=[[liv_v - lo*100], [hi*100 - liv_v]],
                         fmt="none", color="black", capsize=4, lw=1.2)

    ax2.axhline(COV_TARGET * 100, ls="--", lw=1.2, color="green")
    ax2.set_xticks(x)
    ax2.set_xticklabels(PRICE_BUCKET_LABELS)
    ax2.set_ylabel("Coverage %")
    ax2.set_title("§2 PI Coverage by Price Bucket — WF+PI ref vs live (80% CI error bars)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(live: pd.DataFrame, out_path: Path) -> None:
    """
    Top: daily x̂_proxy (mean actual−q50) over time.
    Bottom: residual scatter coloured by price bucket.
    """
    daily = compute_daily_xhat(live)
    sub   = live.dropna(subset=["q50", "actual"]).copy()
    sub["resid"] = sub["actual"] - sub["q50"]
    sub["date"]  = pd.to_datetime(sub["date"])
    sub["bucket"] = pd.cut(sub["actual"], bins=PRICE_BUCKET_BINS, labels=PRICE_BUCKET_LABELS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    # ── Top: daily x̂_proxy ────────────────────────────────────────────────────
    if not daily.empty:
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        ax1.scatter(daily["date"], daily["xhat_proxy"], s=20, alpha=0.7,
                    color="#4C72B0", zorder=3, label="Daily mean(actual−q50)")
        roll7 = daily["xhat_proxy"].rolling(7, min_periods=1).mean()
        ax1.plot(daily["date"], roll7, lw=2.0, color="#C44E52", label="7-day rolling mean")
        ax1.axhline(0,    ls="-",  lw=1.0, color="black")
        ax1.axhline( RESID_WARN,  ls="--", lw=1.2, color="orange", label=f"WARN ±£{RESID_WARN:.0f}")
        ax1.axhline(-RESID_WARN,  ls="--", lw=1.2, color="orange")
        ax1.axhline( RESID_ALERT, ls=":",  lw=1.2, color="red",    label=f"ALERT ±£{RESID_ALERT:.0f}")
        ax1.axhline(-RESID_ALERT, ls=":",  lw=1.2, color="red")

    ax1.set_ylabel("£/MWh")
    ax1.set_title("§4 Kalman/Forecast Residuals — mean(actual − q50) per day\n"
                  "⚠ Note: most files are daily-pipeline snapshots (pre-Kalman)")
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=30)

    # ── Bottom: scatter by bucket ─────────────────────────────────────────────
    bucket_colors = {"<£85": "#4C72B0", "£85-120": "#55A868", "£120-150": "#C44E52",
                     "£150-250": "#8172B2", "£250+": "#937860"}
    for b in PRICE_BUCKET_LABELS:
        bsub = sub[sub["bucket"] == b]
        if bsub.empty:
            continue
        ax2.scatter(bsub["date"], bsub["resid"], s=6, alpha=0.5,
                    color=bucket_colors.get(b, "grey"), label=b)
    ax2.axhline(0, ls="-", lw=1.0, color="black")
    ax2.axhline( RESID_ALERT, ls=":", lw=1.0, color="red")
    ax2.axhline(-RESID_ALERT, ls=":", lw=1.0, color="red")
    ax2.set_ylabel("actual − q50  (£/MWh)")
    ax2.set_title("Residual scatter by price bucket")
    ax2.legend(fontsize=7, ncol=3)
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_step3(gate: dict, out_path: Path) -> None:
    """
    Horizontal timeline bar showing which autumns are usable vs excluded.
    """
    year_statuses = gate.get("year_statuses", {})
    all_years = sorted(year_statuses.keys())
    future_yr = max(all_years) + 1 if all_years else 2026

    status_colors = {
        "crisis":     "#C44E52",
        "transition": "#DD8452",
        "quiet":      "#8172B2",
        "usable":     "#55A868",
        "pending":    "#AAAAAA",
    }
    status_labels = {
        "crisis":     "Crisis (excluded)",
        "transition": "Transition (excluded)",
        "quiet":      "Too quiet (excl.)",
        "usable":     "Usable (PASS)",
        "pending":    "Pending",
    }

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.set_xlim(min(all_years, default=2021) - 0.3,
                max(all_years, default=2025) + 2.0)
    ax.set_ylim(-0.5, 1.0)
    ax.set_yticks([])

    seen = set()
    for yr, info in sorted(year_statuses.items()):
        status = info["status"]
        color  = status_colors.get(status, "grey")
        ax.add_patch(mpatches.FancyBboxPatch(
            (yr - 0.45, -0.3), 0.9, 0.6,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor="white", facecolor=color, alpha=0.85,
        ))
        label = status_labels.get(status, status)
        ar  = info["annual_rate"]
        aur = info["autumn_rate"]
        ax.text(yr, -0.02, f"{yr}", ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
        ax.text(yr, -0.25, f"ann {ar:.0%}", ha="center", va="center",
                fontsize=7, color="white")
        ax.text(yr, 0.18, f"aut {aur:.0%}" if aur > 0 else "",
                ha="center", va="center", fontsize=7, color="white")
        if label not in seen:
            seen.add(label)

    # Future trigger year
    ax.add_patch(mpatches.FancyBboxPatch(
        (future_yr - 0.45, -0.3), 0.9, 0.6,
        boxstyle="round,pad=0.05", linewidth=1.5,
        linestyle="--", edgecolor="#55A868", facecolor="lightgrey", alpha=0.5,
    ))
    ax.text(future_yr, 0.0, f"{future_yr}\n(trigger)", ha="center", va="center",
            fontsize=8.5, color="#55A868", fontweight="bold")

    n_usable  = gate.get("n_usable_autumns", 0)
    min_aut   = gate.get("min_autumns", 2)
    usable_list = gate.get("usable_autumns", [])
    green     = gate.get("green", False)
    gate_sym  = "GREEN" if green else "RED"

    ax.text(0.5, 0.82,
            f"Step-3 Gate: {gate_sym}  —  {n_usable}/{min_aut} usable spike-bearing autumns "
            f"({', '.join(str(y) for y in usable_list) or 'none yet'})\n"
            f"Trigger: {gate.get('trigger', 'unknown')}",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Legend
    patches = [mpatches.Patch(color=c, label=l)
               for s, (c, l) in zip(status_colors.keys(),
                                    [(v, status_labels[k])
                                     for k, v in status_colors.items()])]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5, ncol=2)
    ax.set_xlabel("Calendar year")
    ax.set_title("§7 Step-3 Readiness Gate — Usable Spike-Bearing Training Autumns (Sep–Nov)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Report assembly ────────────────────────────────────────────────────────────

def _cov_row(label, m, target=COV_TARGET, min_n=COV_MIN_N):
    n   = m["n"]
    cov = m["coverage"]
    lo  = m["ci_lo"]
    hi  = m["ci_hi"]
    if np.isnan(cov):
        return f"| {label} | {n} | — | — | — | ⚠️ |"
    delta = cov - target * 100
    st    = _status(cov / 100, COV_WARN_LO, COV_WARN_HI, n, min_n)
    ci    = f"[{lo:.1f}%, {hi:.1f}%]" if not np.isnan(lo) else "—"
    note  = " ⚠️ low-confidence" if n < min_n else ""
    return (f"| {label} | {n}{note} | {cov:.1f}% | {ci} "
            f"| {delta:+.1f}pp | {st} |")


def write_report(
    week_str: str,
    wf: pd.DataFrame,
    live: pd.DataFrame,
    gate: dict,
    plot_dir: Path,
    out_path: Path,
    nowcast_bands: Optional[dict] = None,
    nowcast_cov: Optional[dict] = None,
    nowcast_staleness: Optional[dict] = None,
    nowcast_h3: Optional[dict] = None,
    pipeline_health: Optional[dict] = None,
) -> None:
    today = date.today()

    # ── Rolling 4-week live ────────────────────────────────────────────────────
    cutoff_4w = today - timedelta(days=28)
    live_4w   = live[live["date"] >= cutoff_4w] if not live.empty else live

    # ── Coverage metrics ───────────────────────────────────────────────────────
    cov_wf      = compute_coverage(wf)
    cov_live    = compute_coverage(live)
    cov_live_4w = compute_coverage(live_4w)

    wf_bucket   = compute_bucket_coverage(wf)
    live_bucket = compute_bucket_coverage(live)

    # ── Risk-flag split ────────────────────────────────────────────────────────
    def _flag_split(df, label):
        rows = []
        for flag_val in [True, False]:
            sub = df[df["spike_risk_flag"] == flag_val]
            m   = compute_coverage(sub)
            n_days = sub["date"].nunique() if not sub.empty else 0
            spike_rate = float((sub["actual"] > 150).mean()) if not sub.empty else float("nan")
            fl_str = "True (risk)" if flag_val else "False (no-risk)"
            rows.append(f"| {fl_str} — {label} | {m['n']} ({n_days}d) | "
                        f"{m['coverage']:.1f}% | {spike_rate:.1%} |")
        return rows

    risk_wf_rows   = _flag_split(wf, "WF")
    risk_live_rows = _flag_split(live, "live")

    # Flag differential check
    def _flag_delta(df):
        risk   = df[df["spike_risk_flag"] == True]
        norisk = df[df["spike_risk_flag"] == False]
        m_risk   = compute_coverage(risk)
        m_norisk = compute_coverage(norisk)
        return m_risk["coverage"] - m_norisk["coverage"], m_risk["n"], m_norisk["n"]

    live_flag_delta, live_flag_n_risk, live_flag_n_norisk = _flag_delta(live)

    # ── Residuals ──────────────────────────────────────────────────────────────
    resid_live    = compute_residuals(live)
    resid_live_4w = compute_residuals(live_4w)

    # ── Corrector config ───────────────────────────────────────────────────────
    cfg = {}
    cfg_path = ASSETS_DIR / "corrector_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    spike_widening = cfg.get("spike_widening", False)

    # ── Classifier eval (backtest reference) ──────────────────────────────────
    clf_eval = {}
    clf_path = ASSETS_DIR / "spike_classifier_v1_eval.json"
    if clf_path.exists():
        with open(clf_path) as f:
            clf_eval = json.load(f)

    # ── Date ranges ───────────────────────────────────────────────────────────
    live_date_min = min(live["date"]) if not live.empty else "—"
    live_date_max = max(live["date"]) if not live.empty else "—"
    live_n_days   = live["date"].nunique() if not live.empty else 0

    # ── Exec summary ──────────────────────────────────────────────────────────
    cov_live_st    = _status(cov_live["coverage"] / 100, COV_WARN_LO, COV_WARN_HI, cov_live["n"], COV_MIN_N)
    cov_live_4w_st = _status(cov_live_4w["coverage"] / 100, COV_WARN_LO, COV_WARN_HI, cov_live_4w["n"], COV_MIN_N)
    flag_delta_st  = "⚠️" if min(live_flag_n_risk, live_flag_n_norisk) < 50 \
                     else ("🟡" if abs(live_flag_delta) > 15 else "🟢")
    resid_st       = _resid_status(resid_live_4w["win_mean"], resid_live_4w["n"])
    step3_st       = "🟢" if gate.get("green") else "🔴"

    # Nowcast band status variables
    _nb  = nowcast_bands or {}
    _nc  = nowcast_cov or {}
    _ns  = nowcast_staleness or {}
    _nh3 = nowcast_h3 or {}

    def _nc_cov_status(cov_pct: float, ci_hi: float) -> str:
        """
        CI-based nowcast coverage status.
        🔴  CI upper bound is entirely below the 78% floor → genuine under-coverage
        🟡  Point estimate below 80% target but CI upper bound ≥ 78% floor → watch
        🟢  Point estimate ≥ 80% target → on-target
        """
        warn_lo_pct = NOWCAST_COV_WARN_LO * 100
        target_pct  = NOWCAST_COV_TARGET  * 100
        if ci_hi < warn_lo_pct:
            return "🔴"
        if cov_pct < target_pct:
            return "🟡"
        return "🟢"

    def _nc_status(h_key: str) -> tuple[str, str, str]:
        """Returns (cov_str, n_str, status_emoji) for h_key overall coverage."""
        if not _nc or h_key not in _nc:
            return "—", "—", "⚠️"
        d = _nc[h_key].get("overall", {})
        cov_pct = d.get("coverage", float("nan"))
        ci_hi   = d.get("ci_hi",   float("nan"))
        n       = d.get("n", 0)
        if np.isnan(cov_pct) or n == 0:
            return "—", str(n), "⚠️"
        return f"{cov_pct:.1f}%", str(n), _nc_cov_status(cov_pct, ci_hi)

    _nc_h1_cov, _nc_h1_n, _nc_h1_st = _nc_status("h1")
    _nc_h3_cov, _nc_h3_n, _nc_h3_st = _nc_status("h3")
    _stale_st  = _ns.get("status", "⚠️")
    _h3_rdy_st = "🟢" if _nh3.get("green") else "🔴"

    rel = lambda p: p.relative_to(out_path.parent)

    def _rel_plot(name):
        return f"plots/{name}"

    # ── Bucket table ───────────────────────────────────────────────────────────
    def _bucket_table():
        header = ("| Bucket | WF N | WF cov | Live N | Live cov | Live 90% CI | Status |")
        sep    = ("|---|---|---|---|---|---|---|")
        rows = []
        for b in PRICE_BUCKET_LABELS:
            wf_m   = wf_bucket[b]
            live_m = live_bucket[b]
            wf_cov = f"{wf_m['coverage']:.1f}%" if not np.isnan(wf_m['coverage']) else "—"
            if not np.isnan(live_m['coverage']) and live_m['n'] > 0:
                li_cov = f"{live_m['coverage']:.1f}%"
                ci = f"[{live_m['ci_lo']:.1f}%, {live_m['ci_hi']:.1f}%]"
            else:
                li_cov = "—"
                ci = "—"
            st = (_status(live_m['coverage'] / 100, COV_WARN_LO, COV_WARN_HI,
                          live_m['n'], COV_MIN_BUCKET)
                  if not np.isnan(live_m.get('coverage', float('nan'))) else "⚠️")
            rows.append(f"| {b} | {wf_m['n']} | {wf_cov} | "
                        f"{live_m['n']} | {li_cov} | {ci} | {st} |")
        return "\n".join([header, sep] + rows)

    lines = [
        f"# Production Monitoring Report — {week_str}",
        f"",
        f"**Generated:** {today.isoformat()} (weekly cadence)  ",
        f"**WF backtest period:** 2025-07-01 → 2026-04-30 (119 days, PI-calibrated reference)  ",
        f"**Live period:** {live_date_min} → {live_date_max} ({live_n_days} days with settled actuals)  ",
        f"**Config:** corrector={cfg.get('corrector','?')} · spike_widening={spike_widening} · "
        f"spike_tau={cfg.get('spike_tau','?')}",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        f"| Section | Metric | N | Value | Target | Status |",
        f"|---|---|---|---|---|---|",
        f"| §1 | PI coverage (live, all) | {cov_live['n']} | {cov_live['coverage']:.1f}% | {COV_TARGET*100:.1f}% | {cov_live_st} |",
        f"| §1 | PI coverage (rolling 4w) | {cov_live_4w['n']} | {cov_live_4w['coverage']:.1f}% | {COV_TARGET*100:.1f}% | {cov_live_4w_st} |",
        f"| §3 | Risk-flag Δcov (live) | min({live_flag_n_risk},{live_flag_n_norisk}) | {live_flag_delta:+.1f}pp | <15pp | {flag_delta_st} |",
        f"| §4 | Residual win.mean (4w) | {resid_live_4w['n']} | £{resid_live_4w['win_mean']:.1f} | <£{RESID_WARN:.0f} | {resid_st} |",
        f"| §5 | Spike widening | — | {'ACTIVE' if spike_widening else 'INACTIVE'} | — | {'🟢' if spike_widening else '⬛'} |",
        f"| §6 | Classifier calibration | — | {'ACTIVE' if spike_widening else 'INACTIVE'} | — | {'🟢' if spike_widening else '⬛'} |",
        f"| §7 | Step-3 gate | {gate.get('n_usable_autumns',0)}/{gate.get('min_autumns',2)} aut | "
        f"{'GREEN' if gate.get('green') else 'RED'} | ≥{gate.get('min_autumns',2)} aut | {step3_st} |",
        f"| §7 | H+3 nowcast readiness | {_nh3.get('n_months',0)}/{_nh3.get('min_months',6)} mo | "
        f"{'GREEN' if _nh3.get('green') else 'RED'} | ≥{_nh3.get('min_months',6)} mo | {_h3_rdy_st} |",
        f"| §8 | Nowcast h+1 cov (overall) | {_nc_h1_n} | {_nc_h1_cov} | ≥{NOWCAST_COV_WARN_LO*100:.0f}% | {_nc_h1_st} |",
        f"| §8 | Nowcast h+3 cov (overall) | {_nc_h3_n} | {_nc_h3_cov} | ≥{NOWCAST_COV_WARN_LO*100:.0f}% | {_nc_h3_st} |",
        f"| §8 | Nowcast bands age | — | {_ns.get('age_days','—')}d | <{NOWCAST_BANDS_STALE_DAYS}d | {_stale_st} |",
    ]

    _ph_exec = pipeline_health if pipeline_health is not None else {}
    _ph_age_exec = _ph_exec.get("age_hours")
    _ph_age_exec_str = f"{_ph_age_exec:.1f} h" if _ph_age_exec is not None else "—"
    lines += [
        f"| §9 | Pipeline health | — | {_ph_age_exec_str} old | <4 h | {_ph_exec.get('status_emoji', '⚠️')} |",
        f"",
        f"**Status key:** 🟢 OK · 🟡 WARN · 🔴 ALERT · ⬛ INACTIVE · ⚠️ low-confidence (N < minimum)",
        f"",
        f"---",
        f"",
        f"## §1. PI Coverage — Overall",
        f"",
        f"| Window | N (SP-rows) | Coverage | 90% CI | vs Target | Status |",
        f"|---|---|---|---|---|---|",
        _cov_row("WF backtest (PI-calibrated)", cov_wf, min_n=0),
        _cov_row("Live — all data", cov_live),
        _cov_row("Live — rolling 4w", cov_live_4w),
        f"",
        f"Target: {COV_TARGET*100:.1f}% · WARN if < {COV_WARN_LO*100:.1f}% or > {COV_WARN_HI*100:.1f}%  ",
        f"Minimum N for flagging: {COV_MIN_N} SP-rows  ",
        f"Note: WF baseline has PI calibration applied post-hoc (`ssp_q10 − δ_sp`, `ssp_q90 + δ_sp`). "
        f"Live forecast CSVs were backfilled with PI calibration on 2026-06-17; pipeline-generated files "
        f"from 2026-06-17 onwards have calibration applied at generation time.  ",
        f"Live sample is still small ({live_n_days} days × 48 SPs = {live_n_days*48} SP-rows); "
        f"interpret ⚠️ rows with caution.",
        f"",
        f"![PI Coverage Timeline]({_rel_plot(out_path.stem + '_coverage.png')})",
        f"",
        f"---",
        f"",
        f"## §2. PI Coverage by Price Bucket",
        f"",
        _bucket_table(),
        f"",
        f"Minimum N per bucket for flags: {COV_MIN_BUCKET} SP-rows.  ",
        f"Note: \\<£85 bucket has structurally low coverage because the base model tends to "
        f"over-predict at low market prices (q10 > actual), which is independent of the PI "
        f"widening. This is expected and documented in the backtest report §7.",
        f"",
        f"---",
        f"",
        f"## §3. Ex-Ante Risk-Flag Coverage Split",
        f"",
        f"| spike_risk_flag | N (SP-rows, days) | Coverage | Spike-day rate (>£150) |",
        f"|---|---|---|---|",
    ] + risk_wf_rows + risk_live_rows + [
        f"",
        f"Ex-ante flag built from D-1 information only (elevated_count_lag1d, wind, month).  ",
        f"Alert if |cov_risk − cov_norisk| > 15pp with N ≥ 50 on both sides.  ",
        f"Current live Δcov: **{live_flag_delta:+.1f}pp** ({flag_delta_st})",
        f"",
        f"---",
        f"",
        f"## §4. Kalman / Forecast Residuals",
        f"",
        f"Post-correction residual = `actual_ssp − q50_forecast`. Should sit near £0 "
        f"when Kalman is correcting effectively. Reported as a winsorised mean (5%/95% clip) "
        f"to suppress spike-day false alarms.",
        f"",
        f"| Window | N (SP-rows) | Winsorised mean | Median | WARN | Status |",
        f"|---|---|---|---|---|---|",
        f"| Live — all data | {resid_live['n']} | £{resid_live['win_mean']:.1f} | "
        f"£{resid_live['median']:.1f} | ±£{RESID_WARN:.0f} | {_resid_status(resid_live['win_mean'], resid_live['n'])} |",
        f"| Live — rolling 4w | {resid_live_4w['n']} | £{resid_live_4w['win_mean']:.1f} | "
        f"£{resid_live_4w['median']:.1f} | ±£{RESID_WARN:.0f} | {resid_st} |",
        f"",
        f"⚠️ **Data caveat:** {resid_live.get('n_daily', '?')} SP-rows are from daily-pipeline "
        f"snapshots (PI-calibrated, pre-Kalman). Only {resid_live.get('n_intraday', 0)} rows "
        f"are from intraday-updated files (post-Kalman). Residuals primarily reflect pre-Kalman "
        f"bias. A persistent non-zero mean indicates how hard the Kalman filter has to work each "
        f"day; it is NOT a filter failure.",
        f"",
        f"![Kalman Residuals]({_rel_plot(out_path.stem + '_kalman.png')})",
        f"",
        f"---",
        f"",
        f"## §5. Spike Widening Monitor",
        f"",
    ]

    if spike_widening:
        lines += [
            f"Spike widening is **ACTIVE** (τ={cfg.get('spike_tau','?')}).",
            f"Monitor: firing rate · flagged-day precision (spike_day > £150) · "
            f"width cost at SP 33–40 on false-alarm afternoons.",
            f"*(Full monitoring not yet implemented — add after first few weeks of live data.)*",
        ]
    else:
        lines += [
            f"**INACTIVE** — `spike_widening: false` in `model_assets/corrector_config.json`.  ",
            f"Gate results are in `reports/corrector_backtest/report.md §8`.  ",
            f"Enable manually after reviewing the §8 gate table.",
        ]

    lines += [
        f"",
        f"---",
        f"",
        f"## §6. Classifier Calibration",
        f"",
    ]

    if spike_widening:
        lines += [
            f"Spike classifier is **ACTIVE** — calibration monitoring not yet wired for live inference.",
            f"*(Add rolling Brier score once ≥4 weeks of active production runs are available.)*",
        ]
    else:
        brier   = clf_eval.get("brier_score", "—")
        ap      = clf_eval.get("average_precision", "—")
        n_eval  = clf_eval.get("n_eval_days", "—")
        lines += [
            f"**INACTIVE** — spike classifier is not running in production.  ",
            f"Backtest reference (walk-forward window, {n_eval} days): "
            f"Brier={brier}, AP={ap}.  ",
            f"Wind skew note: `wind_pct_daily_mean_lag1d` is CI D-1 actual in training but "
            f"WINDFOR (NG ESO day-ahead) at inference → classifier biased toward conservatism "
            f"on low-wind days (see `spike_classifier_v1_eval.json`).",
        ]

    n_usable = gate.get("n_usable_autumns", 0)
    min_aut  = gate.get("min_autumns", 2)
    green    = gate.get("green", False)
    usable   = gate.get("usable_autumns", [])

    lines += [
        f"",
        f"---",
        f"",
        f"## §7. Step-3 Readiness Gate",
        f"",
        f"**Purpose:** Gate on revisiting the P95/P99 upper-tail head. GREEN only when "
        f"≥{min_aut} USABLE spike-bearing training autumns (Sep–Nov) are present in the "
        f"non-crisis, non-regime-weighted training data.",
        f"",
        f"| | Value |",
        f"|---|---|",
        f"| Training data span | {gate.get('train_start','?')} → {gate.get('train_end','?')} |",
        f"| Usable non-crisis span | {gate.get('usable_start','?')} → present |",
        f"| Usable spike-bearing autumns | {n_usable} / {min_aut} required |",
        f"| Autumns counted | {', '.join(str(y) + ' ✅' for y in usable) or '(none yet)'} |",
        f"| Gate | {'🟢 **GREEN — revisit P95/P99**' if green else '🔴 **RED — Step 3 DEFERRED**'} |",
        f"| Trigger | {gate.get('trigger','?')} |",
        f"",
        f"**Autumn classification:**",
        f"",
        f"| Year | Annual spike rate | Autumn spike rate | Classification |",
        f"|---|---|---|---|",
    ]

    status_labels = {
        "crisis":     "EXCLUDED — energy crisis",
        "transition": "EXCLUDED — post-crisis transition",
        "quiet":      "EXCLUDED — annual rate too quiet",
        "usable":     "USABLE",
        "pending":    "PENDING — autumn not yet settled",
    }
    for yr, info in sorted(gate.get("year_statuses", {}).items()):
        sl      = status_labels.get(info["status"], info["status"])
        aut_str = f"{info['autumn_rate']:.1%}" if info["n_autumn_days"] > 0 else "—"
        lines.append(
            f"| {yr} | {info['annual_rate']:.1%} | {aut_str} | {sl} |"
        )

    lines += [
        f"",
        f"![Step-3 Gate]({_rel_plot(out_path.stem + '_step3.png')})",
        f"",
        f"### H+3 Nowcast Readiness Gate",
        f"",
        f"**Purpose:** The DA-Q50 feature improves h+3 persistence nowcast (partial R² ~10%, "
        f"Experiment 2). This gate tracks when enough forecast archive is available to validate "
        f"the improvement out-of-sample. GREEN at ≥{NOWCAST_H3_MIN_MONTHS} calendar months archived.",
        f"",
        f"| | Value |",
        f"|---|---|",
        f"| Forecast archive months available | {_nh3.get('n_months', 0)} |",
        f"| Required months | {_nh3.get('min_months', NOWCAST_H3_MIN_MONTHS)} |",
        f"| First archive date | {_nh3.get('first_date', '—')} |",
        f"| Latest archive date | {_nh3.get('last_date', '—')} |",
        f"| Gate | {'🟢 **GREEN — validate DA-Q50 feature**' if _nh3.get('green') else '🔴 **RED — archive accumulating**'} |",
        f"| Trigger | {_nh3.get('trigger', '—')} |",
        f"",
        f"---",
        f"",
        f"## §8. Nowcast Band Health",
        f"",
        f"Nowcast bands (`model_assets/nowcast_bands.json`) ship P10/P90 residual quantiles "
        f"for the persistence nowcaster (h+1/h+2/h+3).  "
        f"Target: {NOWCAST_COV_TARGET*100:.0f}% empirical coverage. "
        f"Flag 🔴 if overall coverage drifts below {NOWCAST_COV_WARN_LO*100:.0f}%.",
        f"",
        f"**Bands metadata:** generated={_nb.get('generated', '—')} · "
        f"fit_window={_nb.get('fit_window', {}).get('start', '—')} → "
        f"{_nb.get('fit_window', {}).get('end', '—')} "
        f"({_nb.get('fit_window', {}).get('months', '?')} months, "
        f"{_nb.get('fit_window', {}).get('n_rows', '?'):,} SP pairs)  " if _nb else
        "**Bands metadata:** `nowcast_bands.json` not found — run `python src/models/build_nowcast_bands.py`  ",
        f"",
        f"### §8a. Live Coverage by Horizon and Regime",
        f"",
    ]

    if _nc:
        nc_header = "| Horizon | Regime | Coverage | 90% CI | N | vs Target | Status |"
        nc_sep    = "|---|---|---|---|---|---|---|"
        nc_rows   = [nc_header, nc_sep]
        for h_key in ["h1", "h2", "h3"]:
            if h_key not in _nc:
                continue
            h_disp = h_key.replace("h", "h+")
            for regime_key in ["overall", "NP", "EN"]:
                d = _nc[h_key].get(regime_key, {})
                cov_val = d.get("coverage", float("nan"))
                n_val   = d.get("n", 0)
                ci_lo   = d.get("ci_lo", float("nan"))
                ci_hi   = d.get("ci_hi", float("nan"))
                if np.isnan(cov_val) or n_val == 0:
                    nc_rows.append(f"| {h_disp} | {regime_key} | — | — | {n_val} | — | ⚠️ |")
                else:
                    delta = cov_val - NOWCAST_COV_TARGET * 100
                    st    = _nc_cov_status(cov_val, ci_hi)
                    nc_rows.append(
                        f"| {h_disp} | {regime_key} | {cov_val:.1f}% | "
                        f"[{ci_lo:.1f}%, {ci_hi:.1f}%] | {n_val} | "
                        f"{delta:+.1f}pp | {st} |"
                    )
        lines += nc_rows
        lines += [
            f"",
            f"Coverage computed from `data/raw/system_prices.csv` (~60-day window).  ",
            f"NP regime = N-code (normal auction); EN regime = P/K-code (formula).  ",
            f"Status: 🟢 ≥{NOWCAST_COV_TARGET*100:.0f}% · "
            f"🟡 <{NOWCAST_COV_TARGET*100:.0f}% but 90% CI upper bound ≥{NOWCAST_COV_WARN_LO*100:.0f}% (watch) · "
            f"🔴 90% CI upper bound <{NOWCAST_COV_WARN_LO*100:.0f}% (genuine under-coverage — refresh bands).  ",
            f"To refresh: `python src/models/build_nowcast_bands.py` then commit `model_assets/nowcast_bands.json`.",
        ]
    else:
        lines += [
            f"Coverage data unavailable — either `nowcast_bands.json` is missing or "
            f"`data/raw/system_prices.csv` has insufficient rows.",
        ]

    lines += [
        f"",
        f"### §8b. Bands Staleness",
        f"",
        f"| | Value |",
        f"|---|---|",
        f"| Generated | {_ns.get('generated', '—')} |",
        f"| Age | {_ns.get('age_days', '—')} days |",
        f"| Stale threshold | {NOWCAST_BANDS_STALE_DAYS} days |",
        f"| Status | {_stale_st} {'(refresh recommended)' if _ns.get('stale') else '(OK)'} |",
        f"",
        f"Refresh command: `python src/models/build_nowcast_bands.py` then commit `model_assets/nowcast_bands.json`.",
        f"",
        f"---",
        f"",
        f"## §9. Pipeline Health",
        f"",
        f"Intraday and daily pipeline health as of report generation time.  ",
        f"Status legend: 🟢 ok (updated within 4 h) · "
        f"🟡 morning_gap (forecast_date < today, before 14:00 UTC — expected daily quiet period) · "
        f"🔴 stale (forecast_date == today, >4 h without Kalman update) · "
        f"🔴 daily_missed (forecast_date < today, past 14:00 UTC — daily pipeline may have failed).  ",
        f"Archive coverage = fraction of daily forecast files with at least one `is_actual=True` row "
        f"(proxy for days that received a successful intraday commit).",
        f"",
    ]

    _ph = pipeline_health if pipeline_health is not None else {}
    _ph_last   = _ph.get("last_update", "—")
    _ph_fdate  = _ph.get("forecast_date", "—")
    _ph_ns     = _ph.get("n_settled", "—")
    _ph_xhat   = _ph.get("x_hat", float("nan"))
    _ph_age    = _ph.get("age_hours")
    _ph_da     = _ph.get("days_with_actuals", 0)
    _ph_dt     = _ph.get("days_total", 0)
    _ph_st     = _ph.get("status_emoji", "⚠️")
    _ph_age_str = f"{_ph_age:.1f} h" if _ph_age is not None else "—"
    _ph_xhat_str = f"£{_ph_xhat:+.2f}" if not (isinstance(_ph_xhat, float) and math.isnan(_ph_xhat)) else "—"

    lines += [
        f"| | Value |",
        f"|---|---|",
        f"| Status | {_ph_st} {_ph.get('status', 'unknown')} |",
        f"| Last Kalman update | {_ph_last} UTC |",
        f"| Age at report time | {_ph_age_str} |",
        f"| Active forecast_date | {_ph_fdate} |",
        f"| SPs settled (last update) | {_ph_ns} |",
        f"| Kalman x̂ | {_ph_xhat_str}/MWh |",
        f"| Archive days with intraday actuals | {_ph_da} / {_ph_dt} |",
        f"",
        f"*If status is 🔴 stale, check GitHub Actions → Intraday Forecast Update for recent run logs.*",
        f"",
        f"---",
        f"",
        f"*Report generated by `src/monitoring/build_monitoring_report.py`.*",
    ]

    out_path.write_text("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Weekly production monitoring report")
    ap.add_argument("--week", default=None, help="ISO week string, e.g. 2026-W25")
    ap.add_argument("--out", default=str(ROOT / "reports" / "monitoring"),
                    help="Output directory")
    args = ap.parse_args()

    # Determine week string
    if args.week:
        week_str = args.week
    else:
        iso = date.today().isocalendar()
        week_str = f"{iso[0]}-W{iso[1]:02d}"

    out_dir  = Path(args.out)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / f"{week_str}.md"

    print(f"Building monitoring report {week_str} → {report_path}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading PI calibration …")
    delta_map = load_pi_calibration()
    print(f"  {len(delta_map)} SP deltas loaded (range £{min(delta_map.values()):.2f}–£{max(delta_map.values()):.2f})" if delta_map else "  WARNING: pi_calibration_v1.json not found")

    print("Loading WF backtest …")
    wf = load_wf_with_pi(delta_map)
    print(f"  {len(wf)} rows from {wf['date'].nunique()} days")

    print("Loading system prices (actuals) …")
    sys_prices = load_system_prices()
    print(f"  {len(sys_prices)} rows from {sys_prices['date'].nunique()} dates")

    print("Loading live forecast files …")
    live = load_live_data(sys_prices)  # CSVs already PI-calibrated

    if not live.empty:
        print("Computing spike_risk_flag for live dates …")
        live = compute_spike_risk_live(live, sys_prices)
        n_risk = live.groupby("date")["spike_risk_flag"].first().sum()
        print(f"  {n_risk}/{live['date'].nunique()} live days flagged as spike-risk")

    print("Computing Step-3 gate …")
    gate = compute_step3_gate()
    print(f"  Usable autumns: {gate['n_usable_autumns']}/{gate['min_autumns']}  "
          f"gate={'GREEN' if gate['green'] else 'RED'}")

    print("Loading nowcast bands …")
    nowcast_bands = load_nowcast_bands()
    if nowcast_bands:
        print(f"  Loaded bands (generated {nowcast_bands.get('generated','?')})")
    else:
        print("  WARNING: nowcast_bands.json not found")

    print("Computing nowcast coverage …")
    nowcast_cov = compute_nowcast_coverage(sys_prices, nowcast_bands)
    if nowcast_cov:
        for h_key, sub in nowcast_cov.items():
            ov = sub.get("overall", {})
            print(f"  {h_key}: overall={ov.get('coverage','?'):.1f}%  n={ov.get('n','?')}")

    nowcast_staleness = check_nowcast_bands_staleness(nowcast_bands)
    print(f"  Bands age: {nowcast_staleness.get('age_days','?')} days  {nowcast_staleness.get('status','?')}")

    print("Computing H+3 nowcast readiness …")
    nowcast_h3 = compute_nowcast_h3_readiness()
    print(f"  Archive months: {nowcast_h3['n_months']}/{nowcast_h3['min_months']}  "
          f"gate={'GREEN' if nowcast_h3['green'] else 'RED'}")

    print("Checking pipeline health …")
    pipeline_health = compute_pipeline_health()
    _age_h = pipeline_health.get("age_hours")
    print(f"  Status: {pipeline_health['status_emoji']} {pipeline_health['status']}  "
          f"last_update={pipeline_health.get('last_update', '—')}  "
          f"age={f'{_age_h:.1f}h' if _age_h is not None else '—'}  "
          f"n_settled={pipeline_health.get('n_settled', 0)}  "
          f"archive={pipeline_health.get('days_with_actuals', 0)}/{pipeline_health.get('days_total', 0)} days")

    # ── Plots ─────────────────────────────────────────────────────────────────
    cov_plot   = plot_dir / f"{week_str}_coverage.png"
    kalman_plot = plot_dir / f"{week_str}_kalman.png"
    step3_plot  = plot_dir / f"{week_str}_step3.png"

    print("Generating plots …")
    if not wf.empty or not live.empty:
        plot_coverage_timeline(wf, live if not live.empty else pd.DataFrame(), cov_plot)
    if not live.empty:
        plot_residuals(live, kalman_plot)
    if gate.get("year_statuses"):
        plot_step3(gate, step3_plot)

    # ── Report ────────────────────────────────────────────────────────────────
    print("Writing report …")
    write_report(week_str, wf, live if not live.empty else pd.DataFrame(),
                 gate, plot_dir, report_path,
                 nowcast_bands=nowcast_bands,
                 nowcast_cov=nowcast_cov,
                 nowcast_staleness=nowcast_staleness,
                 nowcast_h3=nowcast_h3,
                 pipeline_health=pipeline_health)

    print(f"\nDone. Report: {report_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Summary ────────────────────────────────────────────────────────────────")
    if not live.empty:
        cov_live = compute_coverage(live)
        print(f"  Live PI coverage:   {cov_live['coverage']:.1f}% (n={cov_live['n']}, "
              f"target {COV_TARGET*100:.1f}%)")
        resid = compute_residuals(live)
        print(f"  Residual win.mean:  £{resid['win_mean']:.1f} (WARN ±£{RESID_WARN:.0f})")
    cov_wf = compute_coverage(wf)
    print(f"  WF+PI baseline:     {cov_wf['coverage']:.1f}% (n={cov_wf['n']})")
    print(f"  Step-3 gate:        {'GREEN' if gate['green'] else 'RED'} "
          f"({gate['n_usable_autumns']}/{gate['min_autumns']} autumns, "
          f"trigger: {gate['trigger']})")


if __name__ == "__main__":
    main()

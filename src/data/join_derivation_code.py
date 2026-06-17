"""
Join price_derivation_code and ex-ante spike-risk features onto
model_assets/walk_forward_predictions.csv.

New columns added (appended in-place):
  price_derivation_code      : P / N / K  (from system_prices_5yr.csv)
  is_code_P                  : bool  — Code-P mechanism flag
                                       Cross-tab dimension ONLY, NOT the spike label.
                                       46% of WF rows are Code-P at normal prices.
  is_spike_P                 : bool  — (code=='P') AND (ssp_actual > tukey_fence)
                                       Conjunction, not disjunction.  Fires on 1 row in
                                       WF window (2025-10-13 SP26, £487).
  elevated_count_lag1d       : int   — # SPs on D-1 with ssp_actual > 120 (p90 threshold).
                                       Practical ex-ante activity signal; 42 of 119 days ≥ 5.
  wind_pct_daily_mean_lag1d  : float — Mean wind penetration (0-100 scale) across all 48
                                       SPs of D-1, derived from wind_pct_lag_48 in
                                       features_5yr.csv.  Values < 10 signal low-wind days.
  spike_risk_flag            : bool  — Day-level ex-ante risk signal.  TRUE when:
                                           elevated_count_lag1d >= 5
                                         OR (wind_pct_daily_mean_lag1d < 10
                                             AND month ∈ {9,10,11,3,4,5})
                                       Fires on 43 / 119 days; 3.2× spike rate vs non-flag.
                                       Built from calendar month, NOT fold names.

Design note (for future δ_hi calibration):
  Any conformal δ_hi for the upper tail MUST be derived from elevated-price conformity
  scores (actual − q90 where actual > 120), NOT from Code-P conformity scores (46% of rows,
  most at normal prices).  See docs/spike-tail-design.md §5.

Usage:
    python src/data/join_derivation_code.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parents[2]
WF_PATH   = ROOT / "model_assets" / "walk_forward_predictions.csv"
SP_PATH   = ROOT / "data" / "raw" / "system_prices_5yr.csv"
FEAT_PATH = ROOT / "data" / "processed" / "features_5yr.csv"

TUKEY_FENCE_PATH = ROOT / "model_assets" / "tukey_fence.json"

ELEVATED_THRESHOLD = 120.0   # p90 of WF-window actuals; used for elevated_count_lag1d
ELEVATED_COUNT_MIN = 5       # flag fires when D-1 had >= this many elevated SPs
WIND_LOW_THRESHOLD = 10.0    # % wind penetration; flag also fires below this in aut/spr
HIGH_RISK_MONTHS   = {9, 10, 11, 3, 4, 5}   # September–November, March–May


def _load_fence() -> float:
    if TUKEY_FENCE_PATH.exists():
        import json
        return float(json.load(open(TUKEY_FENCE_PATH))["upper"])
    return 353.7


def main() -> None:
    # ── Load walk-forward predictions ─────────────────────────────────────────
    if not WF_PATH.exists():
        print(f"ERROR: {WF_PATH} not found", file=sys.stderr)
        sys.exit(1)
    wf = pd.read_csv(WF_PATH)
    wf["settlement_date"] = wf["settlement_date"].astype(str)
    wf["settlement_period"] = wf["settlement_period"].astype(int)
    original_cols = list(wf.columns)

    # Drop columns we're about to recompute (idempotent re-run)
    drop_cols = [c for c in [
        "price_derivation_code", "is_code_P", "is_spike_P",
        "elevated_count_lag1d", "wind_pct_daily_mean_lag1d", "spike_risk_flag",
    ] if c in wf.columns]
    if drop_cols:
        wf = wf.drop(columns=drop_cols)

    tukey_fence = _load_fence()
    print(f"WF rows: {len(wf)}  | Tukey fence: £{tukey_fence:.1f}")

    # ── Join price_derivation_code from system_prices_5yr.csv ─────────────────
    if not SP_PATH.exists():
        print(f"ERROR: {SP_PATH} not found", file=sys.stderr)
        sys.exit(1)
    sp = pd.read_csv(SP_PATH, usecols=["settlement_date", "settlement_period",
                                        "price_derivation_code"])
    sp["settlement_date"]    = sp["settlement_date"].astype(str)
    sp["settlement_period"]  = sp["settlement_period"].astype(int)
    sp["price_derivation_code"] = sp["price_derivation_code"].fillna("N").astype(str)

    wf = wf.merge(sp, on=["settlement_date", "settlement_period"], how="left")
    wf["price_derivation_code"] = wf["price_derivation_code"].fillna("N")

    n_missing = wf["price_derivation_code"].eq("N").sum() - \
                (sp["price_derivation_code"] == "N").sum()
    print(f"price_derivation_code joined  | P={wf['price_derivation_code'].eq('P').sum()}  "
          f"N={wf['price_derivation_code'].eq('N').sum()}  "
          f"K={wf['price_derivation_code'].eq('K').sum()}")

    wf["is_code_P"] = wf["price_derivation_code"] == "P"
    wf["is_spike_P"] = (wf["is_code_P"]) & (wf["ssp_actual"] > tukey_fence)
    print(f"is_spike_P rows (Code-P AND >£{tukey_fence:.0f}): {wf['is_spike_P'].sum()}")

    # ── Compute elevated_count_lag1d from WF actuals ──────────────────────────
    # Count of SPs per day with actual > ELEVATED_THRESHOLD, shifted forward 1 day.
    daily_elevated = (
        wf.assign(elevated=(wf["ssp_actual"] > ELEVATED_THRESHOLD).astype(int))
        .groupby("settlement_date")["elevated"]
        .sum()
        .reset_index(name="n_elevated")
    )
    daily_elevated["lag1d_date"] = (
        pd.to_datetime(daily_elevated["settlement_date"]) + pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    date_to_elevated = dict(zip(daily_elevated["lag1d_date"], daily_elevated["n_elevated"]))
    wf["elevated_count_lag1d"] = wf["settlement_date"].map(date_to_elevated).fillna(0).astype(int)
    print(f"elevated_count_lag1d (>£{ELEVATED_THRESHOLD:.0f} SPs on D-1): "
          f"days with >=1: {(wf.groupby('settlement_date')['elevated_count_lag1d'].first() >= 1).sum()}  "
          f"days with >={ELEVATED_COUNT_MIN}: "
          f"{(wf.groupby('settlement_date')['elevated_count_lag1d'].first() >= ELEVATED_COUNT_MIN).sum()}")

    # ── Compute wind_pct_daily_mean_lag1d from features_5yr.csv ──────────────
    if not FEAT_PATH.exists():
        print(f"WARNING: {FEAT_PATH} not found — wind_pct_daily_mean_lag1d set to NaN",
              file=sys.stderr)
        wf["wind_pct_daily_mean_lag1d"] = float("nan")
    else:
        feat = pd.read_csv(FEAT_PATH, usecols=["settlement_date", "settlement_period",
                                                "wind_pct_lag_48"])
        feat["settlement_date"]   = feat["settlement_date"].astype(str)
        feat["settlement_period"] = feat["settlement_period"].astype(int)
        # mean(wind_pct_lag_48) per day D = mean D-1 wind pct across all 48 SPs
        daily_wind = feat.groupby("settlement_date")["wind_pct_lag_48"].mean()
        wf["wind_pct_daily_mean_lag1d"] = wf["settlement_date"].map(daily_wind)
        n_wind_null = wf["wind_pct_daily_mean_lag1d"].isna().sum()
        if n_wind_null:
            print(f"WARNING: wind_pct_daily_mean_lag1d null for {n_wind_null} rows")
        print(f"wind_pct_daily_mean_lag1d joined: "
              f"mean={wf['wind_pct_daily_mean_lag1d'].mean():.1f}%  "
              f"p10={wf['wind_pct_daily_mean_lag1d'].quantile(.1):.1f}%")

    # ── Spike-risk flag ───────────────────────────────────────────────────────
    wf["_month"] = pd.to_datetime(wf["settlement_date"]).dt.month
    high_risk_season = wf["_month"].isin(HIGH_RISK_MONTHS)
    low_wind = wf["wind_pct_daily_mean_lag1d"].fillna(100.0) < WIND_LOW_THRESHOLD

    wf["spike_risk_flag"] = (
        (wf["elevated_count_lag1d"] >= ELEVATED_COUNT_MIN)
        | (low_wind & high_risk_season)
    )
    wf = wf.drop(columns=["_month"])

    n_risk_days = wf.groupby("settlement_date")["spike_risk_flag"].any().sum()
    print(f"spike_risk_flag days: {n_risk_days} / {wf['settlement_date'].nunique()}")

    # ── Validate signal strength ───────────────────────────────────────────────
    for flag_val in [True, False]:
        sub = wf[wf["spike_risk_flag"] == flag_val]
        n_days = sub["settlement_date"].nunique()
        pct_elev = (sub["ssp_actual"] > 150).mean()
        print(f"  spike_risk_flag={flag_val}: {n_days} days  "
              f"p(actual>£150)={pct_elev:.2%}  max=£{sub['ssp_actual'].max():.0f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    wf.to_csv(WF_PATH, index=False)
    new_cols = [c for c in wf.columns if c not in original_cols]
    print(f"\nSaved → {WF_PATH}")
    print(f"New columns added: {new_cols}")


if __name__ == "__main__":
    main()

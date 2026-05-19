"""
Daily-level feature engineering for Phase 3 level-shape decomposition.

Aggregates the SP-level feature table to one row per day.  All lag/rolling
features reference days strictly BEFORE the target day (shift ≥ 1 day),
so the level model is guaranteed leakage-free.

Weather for day D (temp, wind, solar) is treated as a day-ahead forecast
input — available from Open-Meteo before day D starts.  In training we use
actual weather as a perfect-forecast proxy (the same approximation already
used in Phase 2 feature engineering).

Target column: ssp_raw_daily_mean (daily mean of raw SSP across all 48 SPs).

Usage:
    from level_features import build_level_dataset, get_level_feature_cols
    daily = build_level_dataset(sp_df)
    feat_cols = get_level_feature_cols(daily)
"""

import numpy as np
import pandas as pd

LEVEL_TARGET = "ssp_raw_daily_mean"

# Contemporaneous day-D aggregates that are NOT available before day D ends.
# These must never appear as model features.
_CONTEMPORANEOUS = {
    LEVEL_TARGET,
    "ssp_raw_daily_max",
    "ssp_raw_daily_min",
    "ssp_raw_daily_std",
    "ssp_daily_mean",     # winsorised daily mean — also contemporaneous
    "niv_daily_mean",
    "niv_daily_std",
    "is_spike_count",
    "date",
}


def build_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SP-level data to one row per calendar day for the level model.

    Parameters
    ----------
    df : SP-level DataFrame sorted by settlement_datetime, containing at
         minimum settlement_datetime, ssp, ssp_raw, is_spike,
         net_imbalance_volume.  Optionally _raw_temp_c, _raw_wind_ms,
         _raw_solar_wm2, _raw_precip_mm (from weather_features.py).

    Returns
    -------
    daily : daily DataFrame with leakage-free features and the level target.
            Only complete days (exactly 48 SPs) are included.
    """
    df = df.sort_values("settlement_datetime").copy()
    df["_date"] = df["settlement_datetime"].dt.date

    # ── Aggregate SP → daily ───────────────────────────────────────────────
    agg: dict = {
        LEVEL_TARGET:          ("ssp_raw", "mean"),
        "ssp_raw_daily_max":   ("ssp_raw", "max"),
        "ssp_raw_daily_min":   ("ssp_raw", "min"),
        "ssp_raw_daily_std":   ("ssp_raw", "std"),
        "ssp_daily_mean":      ("ssp",     "mean"),
        "niv_daily_mean":      ("net_imbalance_volume", "mean"),
        "niv_daily_std":       ("net_imbalance_volume", "std"),
        "is_spike_count":      ("is_spike", "sum"),
        "_n_sp":               ("ssp_raw",  "count"),
    }
    _weather_vars = []
    for raw_col, clean in [
        ("_raw_temp_c",    "temp_c"),
        ("_raw_wind_ms",   "wind_ms"),
        ("_raw_solar_wm2", "solar_wm2"),
        ("_raw_precip_mm", "precip_mm"),
    ]:
        if raw_col in df.columns:
            agg[f"{clean}_daily_mean"] = (raw_col, "mean")
            agg[f"{clean}_daily_max"]  = (raw_col, "max")
            _weather_vars.append(clean)

    daily = df.groupby("_date").agg(**agg).reset_index()
    daily = daily.rename(columns={"_date": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    # Keep only complete days (DST transitions produce 46 or 50 SPs)
    daily = daily[daily["_n_sp"] == 48].drop(columns=["_n_sp"]).reset_index(drop=True)

    # ── Calendar features for day D ───────────────────────────────────────
    # Properties of the day itself — never derived from D's prices.
    doy = daily["date"].dt.dayofyear.values
    dow = daily["date"].dt.dayofweek.values
    mon = daily["date"].dt.month.values

    daily["day_of_week"]     = dow
    daily["month"]           = mon
    daily["day_of_year"]     = doy
    daily["quarter"]         = daily["date"].dt.quarter
    daily["is_weekend"]      = (dow >= 5).astype(int)
    daily["is_business_day"] = (dow < 5).astype(int)
    daily["sin_dow"]         = np.sin(2 * np.pi * dow / 7)
    daily["cos_dow"]         = np.cos(2 * np.pi * dow / 7)
    daily["sin_month"]       = np.sin(2 * np.pi * mon / 12)
    daily["cos_month"]       = np.cos(2 * np.pi * mon / 12)
    daily["sin_annual"]      = np.sin(2 * np.pi * doy / 365)
    daily["cos_annual"]      = np.cos(2 * np.pi * doy / 365)
    daily["sin_half_annual"] = np.sin(4 * np.pi * doy / 365)
    daily["cos_half_annual"] = np.cos(4 * np.pi * doy / 365)

    # ── Lag features on daily SSP (all shift ≥ 1 day — no leakage) ───────
    for lag_d in [1, 2, 7, 14, 28]:
        daily[f"ssp_daily_mean_lag{lag_d}d"]    = daily["ssp_daily_mean"].shift(lag_d)
        daily[f"ssp_raw_daily_max_lag{lag_d}d"] = daily["ssp_raw_daily_max"].shift(lag_d)
        daily[f"ssp_raw_daily_min_lag{lag_d}d"] = daily["ssp_raw_daily_min"].shift(lag_d)

    for w in [7, 14, 28]:
        roll = daily["ssp_daily_mean"].shift(1).rolling(window=w, min_periods=3)
        daily[f"ssp_daily_roll_mean_{w}d"] = roll.mean()
        daily[f"ssp_daily_roll_std_{w}d"]  = roll.std()
        daily[f"ssp_daily_roll_max_{w}d"]  = roll.max()
        daily[f"ssp_daily_roll_min_{w}d"]  = roll.min()

        roll_raw = daily["ssp_raw_daily_max"].shift(1).rolling(window=w, min_periods=3)
        daily[f"ssp_raw_max_roll_mean_{w}d"] = roll_raw.mean()

    # ── Lag features on NIV ───────────────────────────────────────────────
    for lag_d in [1, 7]:
        daily[f"niv_daily_mean_lag{lag_d}d"] = daily["niv_daily_mean"].shift(lag_d)
    for w in [7, 14]:
        roll_niv = daily["niv_daily_mean"].shift(1).rolling(window=w, min_periods=3)
        daily[f"niv_daily_roll_mean_{w}d"] = roll_niv.mean()
        daily[f"niv_daily_roll_std_{w}d"]  = roll_niv.std()

    # ── Spike count lags ──────────────────────────────────────────────────
    for lag_d in [1, 7]:
        daily[f"spike_count_lag{lag_d}d"] = daily["is_spike_count"].shift(lag_d)
    roll_spk = daily["is_spike_count"].shift(1).rolling(window=7, min_periods=1)
    daily["spike_count_roll_7d"] = roll_spk.sum()

    # ── Weather features ──────────────────────────────────────────────────
    # Same-day weather (col without _lag) = day-ahead forecast at inference;
    # actual weather in training (acceptable proxy — same as Phase 2).
    # Yesterday's weather (lag1d) provides a clean lag feature.
    for var in _weather_vars:
        col_mean = f"{var}_daily_mean"
        col_max  = f"{var}_daily_max"
        if col_mean in daily.columns:
            daily[f"{var}_daily_mean_lag1d"] = daily[col_mean].shift(1)
            daily[f"{var}_daily_mean_lag7d"] = daily[col_mean].shift(7)
        if col_max in daily.columns:
            daily[f"{var}_daily_max_lag1d"]  = daily[col_max].shift(1)

    return daily


def get_level_feature_cols(daily: pd.DataFrame) -> list:
    """Return model feature columns for the level model (excludes target and
    contemporaneous day-D aggregates)."""
    return [c for c in daily.columns if c not in _CONTEMPORANEOUS]

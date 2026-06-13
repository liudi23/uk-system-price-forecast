"""
Weather features for UK electricity price forecasting.

Weather directly drives both supply and demand:
  temperature_2m   → heating/cooling demand (key annual modulation driver)
  wind_speed_10m   → wind generation supply (UK: ~30% of generation mix)
  shortwave_radiation → solar PV output (suppresses midday prices)
  precipitation    → cloud cover proxy (reduces solar output)

Feature groups:
  Contemporaneous lags : weather at t-1, t-48, t-336 (available at forecast time)
  Rolling means/std    : 6-SP (3h), 48-SP (24h) windows for trend context
  Derived indicators   : heating/cooling degree proxies, wind ramp rate

Merge key: settlement_datetime (UTC) matched to weather datetime_utc.

Reference:
  Ziel, F. & Weron, R. (2018). Day-ahead electricity price forecasting with
  high-dimensional structures: Univariate vs. multivariate modeling frameworks.
  Energy Economics, 70, 396–420.
"""

from pathlib import Path

import numpy as np
import pandas as pd

WEATHER_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather_uk.csv"
WEATHER_COLS = ["temp_c", "wind_ms", "solar_wm2", "precip_mm"]

# Heating/cooling thresholds (UK standard: 15.5°C base temperature)
HEATING_BASE = 15.5
COOLING_BASE = 22.0


def load_weather(path: Path = WEATHER_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime_utc"])
    df["datetime_utc"] = df["datetime_utc"].dt.tz_localize(None)  # strip tz for merge
    return df.sort_values("datetime_utc").reset_index(drop=True)


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame,
                  datetime_col: str = "settlement_datetime") -> pd.DataFrame:
    """Left-join weather onto the price DataFrame on datetime (nearest 30-min)."""
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    # Round to nearest 30-min to align with weather grid
    df["_merge_key"] = dt.dt.floor("30min")
    weather_indexed = weather.set_index("datetime_utc")

    merged = df.merge(
        weather_indexed[WEATHER_COLS],
        left_on="_merge_key", right_index=True, how="left",
    )
    merged = merged.drop(columns=["_merge_key"])
    n_missing = merged[WEATHER_COLS[0]].isna().sum()
    if n_missing > 0:
        import logging
        logging.getLogger(__name__).warning(
            "Weather merge: %d rows unmatched (%.1f%%) — forward-filling",
            n_missing, 100 * n_missing / len(merged)
        )
        merged[WEATHER_COLS] = merged[WEATHER_COLS].ffill()
    return merged


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged and rolling weather features. Assumes weather columns
    (temp_c, wind_ms, solar_wm2, precip_mm) are already present from merge_weather().
    All features use shift(≥1) so no contemporaneous leakage.
    """
    df = df.copy()

    # ── Heating / cooling degree proxies ──────────────────────────────────────
    # Use lag-48 (yesterday same period) so available at forecast time
    temp_lag48 = df["temp_c"].shift(48)
    df["heating_degree"] = (HEATING_BASE - temp_lag48).clip(lower=0)
    df["cooling_degree"] = (temp_lag48 - COOLING_BASE).clip(lower=0)

    # ── Lags for each weather variable ────────────────────────────────────────
    for col in WEATHER_COLS:
        df[f"{col}_lag_1"]   = df[col].shift(1)
        df[f"{col}_lag_48"]  = df[col].shift(48)
        df[f"{col}_lag_336"] = df[col].shift(336)

    # ── Rolling statistics ────────────────────────────────────────────────────
    for col in ["temp_c", "wind_ms", "solar_wm2"]:
        for w in [6, 48]:
            roll = df[col].shift(1).rolling(window=w, min_periods=1)
            df[f"{col}_roll_mean_{w}"] = roll.mean()
            df[f"{col}_roll_std_{w}"]  = roll.std()

    # ── Wind ramp rate (change in wind speed over last 6 SPs) ─────────────────
    df["wind_ramp_6"] = df["wind_ms"].shift(1) - df["wind_ms"].shift(7)

    # ── Drop raw contemporaneous weather columns (leakage if kept as features) ─
    # They remain in the df for lag computation above but must not be used
    # directly as model features. Rename with prefix so EXCLUDE_COLS catches them.
    df = df.rename(columns={c: f"_raw_{c}" for c in WEATHER_COLS})

    return df

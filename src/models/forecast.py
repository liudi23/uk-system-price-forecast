"""
Generate a 48-period (24-hour) day-ahead SSP forecast using the production HGBR model.

Strategy: recursive multi-step forecasting.
  - SSP lags ≥ 48 always reference actual history (never exceed the forecast horizon).
  - SSP lags 1 and 2 are filled recursively with previously predicted values.
  - NIV is unforecastable; values within the forecast window are proxied by the
    corresponding settlement period from the previous day.
  - Weather for the forecast date is fetched live from the Open-Meteo forecast API.

Output: model_assets/next_day_forecast.csv

Usage:
    python src/models/forecast.py
    python src/models/forecast.py --date 2026-05-18
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ASSETS_DIR        = Path(__file__).resolve().parents[2] / "model_assets"
FEATURES_FILE     = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_5yr.csv"
RAW_PRICES_FILE   = Path(__file__).resolve().parents[2] / "data" / "raw" / "system_prices.csv"
OUTPUT_FILE       = ASSETS_DIR / "next_day_forecast.csv"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
UK_LOCATIONS = [
    {"name": "England", "lat": 52.5, "lon": -1.5, "weight": 0.6},
    {"name": "Scotland", "lat": 56.5, "lon": -4.0, "weight": 0.2},
    {"name": "Wales",    "lat": 52.3, "lon": -3.7, "weight": 0.2},
]
WEATHER_VARS = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "precipitation"]
WEATHER_RENAME = {
    "temperature_2m":    "temp_c",
    "wind_speed_10m":    "wind_ms",
    "shortwave_radiation": "solar_wm2",
    "precipitation":     "precip_mm",
}
HEATING_BASE = 15.5
COOLING_BASE = 22.0
HISTORY_ROWS = 400   # must be > max lag (336) + forecast horizon (48)


# ── Weather fetch ─────────────────────────────────────────────────────────────

def _fetch_location_forecast(lat, lon, start, end):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "start_date": start, "end_date": end,
        "timezone": "UTC", "wind_speed_unit": "ms",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=30)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    return df


def fetch_forecast_weather(target_date: date, start_date: date = None) -> pd.DataFrame:
    """Fetch 30-min UK weather from Open-Meteo forecast API.

    start_date defaults to target_date - 1. Pass an earlier date to cover
    recent history gaps (e.g. when features_5yr.csv is a few days stale).
    """
    if start_date is None:
        start_date = target_date - timedelta(days=1)
    start = str(start_date)
    end = str(target_date)
    frames = []
    for loc in UK_LOCATIONS:
        df = _fetch_location_forecast(loc["lat"], loc["lon"], start, end)
        for v in WEATHER_VARS:
            df[v] = df[v] * loc["weight"]
        frames.append(df[["time"] + WEATHER_VARS])
    combined = frames[0].copy()
    for v in WEATHER_VARS:
        combined[v] = sum(f[v] for f in frames)
    combined = (
        combined.set_index("time")
        .resample("30min").ffill()
        .reset_index()
        .rename(columns={**WEATHER_RENAME, "time": "datetime_utc"})
    )
    log.info("Weather forecast: %d rows (%s → %s)",
             len(combined), combined["datetime_utc"].min().date(), combined["datetime_utc"].max().date())
    return combined


def _extract_day_weather(weather_df: pd.DataFrame, target_date: date) -> dict:
    """Return dict of 48-element arrays for each weather variable on target_date."""
    start = pd.Timestamp(target_date)
    end = start + pd.Timedelta(hours=23, minutes=30)
    mask = (weather_df["datetime_utc"] >= start) & (weather_df["datetime_utc"] <= end)
    day = weather_df[mask].reset_index(drop=True)
    # Pad if Open-Meteo returns fewer than 48 rows
    while len(day) < 48:
        day = pd.concat([day, day.iloc[[-1]]], ignore_index=True)
    return {col: day[col].values[:48] for col in ["temp_c", "wind_ms", "solar_wm2", "precip_mm"]}


# ── Calendar features ─────────────────────────────────────────────────────────

def _calendar(dt: pd.Timestamp) -> dict:
    sp = dt.hour * 2 + dt.minute // 30 + 1
    doy = dt.day_of_year
    return {
        "hour":           dt.hour,
        "day_of_week":    dt.dayofweek,
        "month":          dt.month,
        "week_of_year":   int(dt.isocalendar().week),
        "quarter":        dt.quarter,
        "day_of_year":    doy,
        "sin_sp":         np.sin(2 * np.pi * sp / 48),
        "cos_sp":         np.cos(2 * np.pi * sp / 48),
        "sin_hour":       np.sin(2 * np.pi * dt.hour / 24),
        "cos_hour":       np.cos(2 * np.pi * dt.hour / 24),
        "sin_dow":        np.sin(2 * np.pi * dt.dayofweek / 7),
        "cos_dow":        np.cos(2 * np.pi * dt.dayofweek / 7),
        "sin_month":      np.sin(2 * np.pi * dt.month / 12),
        "cos_month":      np.cos(2 * np.pi * dt.month / 12),
        "sin_annual":     np.sin(2 * np.pi * doy / 365),
        "cos_annual":     np.cos(2 * np.pi * doy / 365),
        "sin_half_annual": np.sin(4 * np.pi * doy / 365),
        "cos_half_annual": np.cos(4 * np.pi * doy / 365),
        "is_weekend":     int(dt.dayofweek >= 5),
        "is_peak":        int(13 <= sp <= 36),
        "is_evening_ramp": int(33 <= sp <= 40),
        "is_uk_holiday":  0,
        "is_business_day": int(dt.dayofweek < 5),
    }


# ── Main forecast ─────────────────────────────────────────────────────────────

def run_forecast(target_date: date = None, features_file: Path = FEATURES_FILE) -> pd.DataFrame:
    model = joblib.load(ASSETS_DIR / "hgbr_model.pkl")
    with open(ASSETS_DIR / "feature_cols.json") as f:
        feature_cols = json.load(f)

    # ── Load base history from features file ──────────────────────────────────
    log.info("Loading feature history from %s", features_file)
    base_hist = pd.read_csv(features_file, parse_dates=["settlement_datetime"])
    base_hist = base_hist.sort_values("settlement_datetime").reset_index(drop=True)
    features_last_dt = base_hist["settlement_datetime"].max()

    # ── Extend with newer rows from system_prices.csv if available ────────────
    extra = pd.DataFrame()
    if RAW_PRICES_FILE.exists():
        raw = pd.read_csv(RAW_PRICES_FILE, parse_dates=["settlement_date"])
        raw["settlement_datetime"] = (
            raw["settlement_date"]
            + pd.to_timedelta((raw["settlement_period"] - 1) * 30, unit="min")
        )
        extra = raw[raw["settlement_datetime"] > features_last_dt].sort_values(
            "settlement_datetime"
        )
        if not extra.empty:
            log.info(
                "Extending history with %d rows from system_prices.csv (%s → %s)",
                len(extra),
                extra["settlement_datetime"].min().date(),
                extra["settlement_datetime"].max().date(),
            )

    combined_last_dt = extra["settlement_datetime"].max() if not extra.empty else features_last_dt
    if target_date is None:
        target_date = (combined_last_dt + pd.Timedelta(minutes=30)).date()
    log.info("Forecasting %s (history ends %s)", target_date, combined_last_dt)

    # ── Fetch weather covering any gap + target date ──────────────────────────
    # The Open-Meteo forecast API supports ~7 days of recent history, which is
    # enough to fill any gap between features_5yr.csv and today.
    weather_start = (features_last_dt + pd.Timedelta(minutes=30)).date()
    log.info("Fetching Open-Meteo forecast/recent weather %s → %s…", weather_start, target_date)
    weather_df = fetch_forecast_weather(target_date, start_date=weather_start)

    # ── Fill weather into extended rows ───────────────────────────────────────
    if not extra.empty:
        w_idx = weather_df.set_index("datetime_utc")
        extra = extra.copy()
        merge_key = extra["settlement_datetime"].dt.floor("30min")
        for col, raw_col in [
            ("temp_c",    "_raw_temp_c"),
            ("wind_ms",   "_raw_wind_ms"),
            ("solar_wm2", "_raw_solar_wm2"),
            ("precip_mm", "_raw_precip_mm"),
        ]:
            mapped = merge_key.map(w_idx[col])
            extra[raw_col] = mapped.ffill().bfill()

    # ── Build combined history tail ───────────────────────────────────────────
    hist = (
        pd.concat([base_hist, extra], ignore_index=True)
        .sort_values("settlement_datetime")
        .tail(HISTORY_ROWS)
        .reset_index(drop=True)
    )

    # ── History buffers ───────────────────────────────────────────────────────
    ssp_buf     = list(hist["ssp"].values)
    niv_actual  = hist["net_imbalance_volume"].values
    niv_virtual = np.concatenate([niv_actual, niv_actual[-48:]])

    temp_hist   = hist["_raw_temp_c"].values
    wind_hist   = hist["_raw_wind_ms"].values
    solar_hist  = hist["_raw_solar_wm2"].values
    precip_hist = hist["_raw_precip_mm"].values

    # ── Weather for the forecast date ─────────────────────────────────────────
    fw = _extract_day_weather(weather_df, target_date)

    temp_virt  = np.concatenate([temp_hist,  fw["temp_c"]])
    wind_virt  = np.concatenate([wind_hist,  fw["wind_ms"]])
    solar_virt = np.concatenate([solar_hist, fw["solar_wm2"]])
    precip_virt = np.concatenate([precip_hist, fw["precip_mm"]])

    # ── Recursive forecast loop ───────────────────────────────────────────────
    base_dt = pd.Timestamp(target_date)
    records = []

    for h in range(1, 49):
        dt  = base_dt + pd.Timedelta(minutes=30 * (h - 1))
        idx = HISTORY_ROWS + h - 1   # "current" position in virtual arrays (0-indexed)

        row = _calendar(dt)

        # SSP lags — ssp_buf has HISTORY_ROWS + h - 1 elements before append
        ssp_arr = np.array(ssp_buf)
        row["ssp_lag_1"]   = float(ssp_arr[-1])
        row["ssp_lag_2"]   = float(ssp_arr[-2])
        row["ssp_lag_48"]  = float(ssp_arr[-48])   # always actual (h ≤ 48)
        row["ssp_lag_96"]  = float(ssp_arr[-96])
        row["ssp_lag_336"] = float(ssp_arr[-336])

        # SSP rolling — shift(1)+rolling(w) at position p = mean/std of last w values before p
        for w in [6, 48, 336]:
            win = ssp_arr[-w:]
            row[f"ssp_roll_mean_{w}"] = float(win.mean())
            row[f"ssp_roll_std_{w}"]  = float(win.std(ddof=1)) if len(win) > 1 else 0.0
            row[f"ssp_roll_min_{w}"]  = float(win.min())
            row[f"ssp_roll_max_{w}"]  = float(win.max())

        row["ssp_diff_1_48"]   = row["ssp_lag_1"] - row["ssp_lag_48"]
        row["ssp_diff_48_336"] = row["ssp_lag_48"] - row["ssp_lag_336"]

        # NIV lags — niv_virtual[0..399]=actual, [400..447]=proxy (yesterday's SPs)
        row["net_imbalance_volume_lag_1"]   = float(niv_virtual[idx - 1])
        row["net_imbalance_volume_lag_48"]  = float(niv_virtual[idx - 48])
        row["net_imbalance_volume_lag_336"] = float(niv_virtual[idx - 336])
        for w in [6, 48]:
            niv_win = niv_virtual[idx - w : idx]
            row[f"net_imbalance_volume_roll_mean_{w}"] = float(niv_win.mean())
            row[f"net_imbalance_volume_roll_std_{w}"]  = (
                float(niv_win.std(ddof=1)) if len(niv_win) > 1 else 0.0
            )

        # Weather features
        temp_lag48 = float(temp_virt[idx - 48])
        row["heating_degree"] = max(0.0, HEATING_BASE - temp_lag48)
        row["cooling_degree"] = max(0.0, temp_lag48 - COOLING_BASE)

        for col, virt in [
            ("temp_c",    temp_virt),
            ("wind_ms",   wind_virt),
            ("solar_wm2", solar_virt),
            ("precip_mm", precip_virt),
        ]:
            row[f"{col}_lag_1"]   = float(virt[idx - 1])
            row[f"{col}_lag_48"]  = float(virt[idx - 48])
            row[f"{col}_lag_336"] = float(virt[idx - 336])

        for col, virt in [
            ("temp_c",    temp_virt),
            ("wind_ms",   wind_virt),
            ("solar_wm2", solar_virt),
        ]:
            for w in [6, 48]:
                win = virt[idx - w : idx]
                row[f"{col}_roll_mean_{w}"] = float(win.mean())
                row[f"{col}_roll_std_{w}"]  = float(win.std(ddof=1)) if len(win) > 1 else 0.0

        row["wind_ramp_6"] = float(wind_virt[idx - 1] - wind_virt[idx - 7])

        # Predict and append
        X = np.array([[row.get(f, 0.0) for f in feature_cols]])
        pred = float(model.predict(X)[0])
        ssp_buf.append(pred)

        records.append({
            "settlement_datetime": dt,
            "settlement_date":     str(target_date),
            "settlement_period":   h,
            "ssp_predicted":       round(pred, 2),
        })

    result = pd.DataFrame(records)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    log.info("Forecast saved → %s", OUTPUT_FILE)

    # Archive by date so each day's forecast is preserved for later verification
    archive_dir = OUTPUT_FILE.parent / "forecasts"
    archive_dir.mkdir(exist_ok=True)
    archive_path = archive_dir / f"forecast_{target_date}.csv"
    result.to_csv(archive_path, index=False)
    log.info("Archived → %s", archive_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Day-ahead SSP forecast")
    parser.add_argument(
        "--date", default=None,
        help="Target date YYYY-MM-DD (default: day after last row in features_5yr.csv)",
    )
    parser.add_argument("--features", default=str(FEATURES_FILE))
    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else None
    result = run_forecast(target, Path(args.features))
    print(f"\nForecast for {result['settlement_date'].iloc[0]} ({len(result)} periods):")
    print(result[["settlement_datetime", "settlement_period", "ssp_predicted"]].to_string(index=False))


if __name__ == "__main__":
    main()

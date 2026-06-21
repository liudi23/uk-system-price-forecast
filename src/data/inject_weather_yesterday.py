"""
inject_weather_yesterday.py
───────────────────────────
Fetch actual hourly weather from the Open-Meteo archive API for yesterday
(or --date DATE) and write _raw_temp_c / _raw_wind_ms / _raw_solar_wm2 /
_raw_precip_mm into data/raw/system_prices.csv for the matching SPs.

Called from forecast_pipeline.yml before forecast_phase3.py runs, so the
extra-row extension mechanism in forecast_phase3.py gets weather-populated
rows and all 9 lag-weather features are non-NaN.

Train/serve alignment: same archive API, locations, variables, and 30-min
resampling as src/data/fetch_weather.py — no skew introduced.

Resilience:
  1. Archive API: 3 retries with exponential backoff, 60 s timeout.
  2. Fallback to Open-Meteo forecast endpoint (near-actual for yesterday).
  3. If both APIs fail: proceed WITHOUT injection (NaN weather).
     HGBR handles NaN natively (~£2.80 MAE penalty from backtest).
     Script always exits 0 — a weather outage must not kill the forecast run.
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO       = Path(__file__).resolve().parents[2]
RAW_PRICES = REPO / "data" / "raw" / "system_prices.csv"

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

TIMEOUT      = 60   # seconds per request attempt
MAX_RETRIES  = 3
BACKOFF_BASE = 5    # seconds; delays are 5, 10, 20

UK_LOCATIONS = [
    {"name": "England",  "latitude": 52.5,  "longitude": -1.5,  "weight": 0.6},
    {"name": "Scotland", "latitude": 56.5,  "longitude": -4.0,  "weight": 0.2},
    {"name": "Wales",    "latitude": 52.3,  "longitude": -3.7,  "weight": 0.2},
]

VARIABLES = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "precipitation"]
RENAME = {
    "temperature_2m":      "temp_c",
    "wind_speed_10m":      "wind_ms",
    "shortwave_radiation": "solar_wm2",
    "precipitation":       "precip_mm",
}
COL_MAP = {
    "temp_c":    "_raw_temp_c",
    "wind_ms":   "_raw_wind_ms",
    "solar_wm2": "_raw_solar_wm2",
    "precip_mm": "_raw_precip_mm",
}


def _request_with_retry(url: str, params: dict) -> requests.Response:
    """GET with exponential backoff. Raises on final failure."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"[inject_weather] attempt {attempt + 1}/{MAX_RETRIES} failed "
                  f"({type(exc).__name__}: {exc}); retrying in {wait}s")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    raise last_exc


def _fetch_one(lat: float, lon: float, date_str: str, url: str) -> pd.DataFrame:
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(VARIABLES),
        "start_date":      date_str,
        "end_date":        date_str,
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    }
    r = _request_with_retry(url, params)
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").rename(columns=RENAME)
    # Keep only the target date rows (forecast endpoint may return extra days)
    target_date = pd.Timestamp(date_str).date()
    df = df[df.index.date == target_date]
    return df


def _fetch_weighted(date_str: str, url: str) -> pd.DataFrame:
    """Weighted average across UK locations for given URL, resampled to 30-min."""
    combined = None
    for loc in UK_LOCATIONS:
        df = _fetch_one(loc["latitude"], loc["longitude"], date_str, url)
        weighted = df * loc["weight"]
        combined = weighted if combined is None else combined + weighted
    full_idx = pd.date_range(date_str, periods=48, freq="30min")
    combined = combined.resample("30min").ffill().reindex(full_idx).ffill()
    return combined  # index=UTC datetime, cols=temp_c/wind_ms/solar_wm2/precip_mm


def fetch_weather_with_fallback(date_str: str):
    """
    Try archive API, then forecast API, then give up gracefully.

    Returns (weather_df, source) where source is one of:
      'archive'  — Open-Meteo archive (ERA5, best accuracy)
      'forecast' — Open-Meteo forecast model (near-actual, slightly lower accuracy)
      'none'     — both APIs failed; caller should proceed with NaN weather
    """
    # 1. Archive API (with retry)
    try:
        df = _fetch_weighted(date_str, ARCHIVE_URL)
        return df, "archive"
    except Exception as exc:
        print(f"[inject_weather] WARNING: archive API unavailable after {MAX_RETRIES} retries: {exc}")

    # 2. Forecast API fallback (near-actual for yesterday, no retry needed here)
    print("[inject_weather] trying forecast endpoint as fallback …")
    try:
        df = _fetch_weighted(date_str, FORECAST_URL)
        return df, "forecast"
    except Exception as exc:
        print(f"[inject_weather] WARNING: forecast API also failed: {exc}")

    return None, "none"


def main():
    parser = argparse.ArgumentParser(description="Inject archive weather into system_prices.csv")
    parser.add_argument("--date", default=None,
                        help="Target date YYYY-MM-DD (default: yesterday UTC)")
    args = parser.parse_args()

    target = (
        date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    )
    date_str = str(target)
    print(f"[inject_weather] target = {date_str}")

    raw = pd.read_csv(RAW_PRICES, parse_dates=["settlement_date"])
    raw["_sdt"] = (
        raw["settlement_date"]
        + pd.to_timedelta((raw["settlement_period"] - 1) * 30, unit="min")
    )

    mask = raw["settlement_date"].dt.date == target
    n = mask.sum()
    if n == 0:
        print(f"[inject_weather] WARNING: no rows for {date_str} in system_prices.csv — skipping")
        return

    print(f"[inject_weather] {n} SP rows found for {date_str}")
    print("[inject_weather] fetching weather …")

    weather, source = fetch_weather_with_fallback(date_str)

    if weather is None:
        # Both APIs failed — NaN injection; HGBR handles missing weather natively.
        print("[inject_weather] WARNING: all weather sources failed — proceeding WITHOUT "
              "injection. Weather features will be NaN. "
              "HGBR handles NaN natively (~£2.80 MAE penalty per backtest).")
        raw = raw.drop(columns=["_sdt"])
        raw.to_csv(RAW_PRICES, index=False)
        return  # exit 0

    print(f"[inject_weather] {len(weather)} 30-min weather rows (source={source})")

    for raw_col in COL_MAP.values():
        if raw_col not in raw.columns:
            raw[raw_col] = np.nan

    for weath_col, raw_col in COL_MAP.items():
        vals = raw.loc[mask, "_sdt"].map(weather[weath_col])
        raw.loc[mask, raw_col] = vals.values

    n_ok = raw.loc[mask, "_raw_temp_c"].notna().sum()
    print(f"[inject_weather] filled {n_ok}/{n} rows  "
          f"(source={source}, temp_c mean={raw.loc[mask, '_raw_temp_c'].mean():.1f} °C)")

    raw = raw.drop(columns=["_sdt"])
    raw.to_csv(RAW_PRICES, index=False)
    print(f"[inject_weather] saved → {RAW_PRICES}")


if __name__ == "__main__":
    main()

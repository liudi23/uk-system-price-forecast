"""
inject_weather_yesterday.py
───────────────────────────
Fetch actual hourly weather from the Open-Meteo archive API for yesterday
(or --date DATE) and write _raw_temp_c / _raw_wind_ms / _raw_solar_wm2 /
_raw_precip_mm into data/raw/system_prices.csv for the matching SPs.

Called from early_forecast.yml before forecast_phase3.py runs, so the
extra-row extension mechanism in forecast_phase3.py gets weather-populated
rows and all 9 lag-weather features are non-NaN.

Train/serve alignment: same archive API, locations, variables, and 30-min
resampling as src/data/fetch_weather.py — no skew introduced.
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO       = Path(__file__).resolve().parents[2]
RAW_PRICES = REPO / "data" / "raw" / "system_prices.csv"

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

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


def _fetch_one(lat: float, lon: float, date_str: str) -> pd.DataFrame:
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(VARIABLES),
        "start_date":      date_str,
        "end_date":        date_str,
        "timezone":        "UTC",
        "wind_speed_unit": "ms",
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").rename(columns=RENAME)


def fetch_weighted_weather(date_str: str) -> pd.DataFrame:
    """Weighted average across UK locations, resampled to 30-min."""
    combined = None
    for loc in UK_LOCATIONS:
        df = _fetch_one(loc["latitude"], loc["longitude"], date_str)
        weighted = df * loc["weight"]
        combined = weighted if combined is None else combined + weighted
    # Resample to 30-min and ensure all 48 SPs are covered (23:30 needs ffill from 23:00)
    full_idx = pd.date_range(date_str, periods=48, freq="30min")
    combined = combined.resample("30min").ffill().reindex(full_idx).ffill()
    return combined  # index = UTC datetime, cols = temp_c / wind_ms / solar_wm2 / precip_mm


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
        sys.exit(0)
    print(f"[inject_weather] {n} SP rows found for {date_str}")

    print("[inject_weather] fetching archive weather …")
    weather = fetch_weighted_weather(date_str)
    print(f"[inject_weather] {len(weather)} 30-min weather rows")

    for raw_col in COL_MAP.values():
        if raw_col not in raw.columns:
            raw[raw_col] = np.nan

    for weath_col, raw_col in COL_MAP.items():
        vals = raw.loc[mask, "_sdt"].map(weather[weath_col])
        raw.loc[mask, raw_col] = vals.values

    n_ok = raw.loc[mask, "_raw_temp_c"].notna().sum()
    print(f"[inject_weather] filled {n_ok}/{n} rows  (temp_c sample: "
          f"{raw.loc[mask, '_raw_temp_c'].mean():.1f} °C mean)")

    raw = raw.drop(columns=["_sdt"])
    raw.to_csv(RAW_PRICES, index=False)
    print(f"[inject_weather] saved → {RAW_PRICES}")


if __name__ == "__main__":
    main()

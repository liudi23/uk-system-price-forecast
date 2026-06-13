"""
Fetch UK weather data from Open-Meteo historical archive API.

Uses three representative UK locations (England, Scotland, Wales) and
averages them to produce a single UK-representative weather series.

Variables fetched (hourly, then resampled to 30-min settlement periods):
  temperature_2m       — air temperature (°C): drives heating/cooling demand
  wind_speed_10m       — wind speed (m/s): proxy for wind generation output
  shortwave_radiation  — solar irradiance (W/m²): proxy for solar output
  precipitation        — rainfall (mm/h): cloud cover proxy, affects solar

API: Open-Meteo ERA5 archive (free, no API key required)
     https://archive-api.open-meteo.com/v1/archive

Output: data/raw/weather_uk.csv

Usage:
    python src/data/fetch_weather.py
    python src/data/fetch_weather.py --start 2021-05-18 --end 2026-05-17
"""

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "data" / "raw" / "weather_uk.csv"
BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Three UK locations weighted by approximate population/demand share
UK_LOCATIONS = [
    {"name": "England",  "latitude": 52.5,  "longitude": -1.5,  "weight": 0.6},
    {"name": "Scotland", "latitude": 56.5,  "longitude": -4.0,  "weight": 0.2},
    {"name": "Wales",    "latitude": 52.3,  "longitude": -3.7,  "weight": 0.2},
]

VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "precipitation",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_location(name: str, lat: float, lon: float,
                   start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"time": "datetime_utc"})
    log.info("  %s: %d hourly rows", name, len(df))
    return df


def resample_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill hourly weather to 30-min settlement period grid."""
    df = df.set_index("datetime_utc")
    df_30 = df.resample("30min").ffill()
    df_30 = df_30.reset_index()
    return df_30


def fetch_uk_weather(start: str, end: str) -> pd.DataFrame:
    frames = []
    for loc in UK_LOCATIONS:
        log.info("Fetching %s (%.1f°N, %.1f°E)…", loc["name"], loc["latitude"], loc["longitude"])
        df = fetch_location(loc["name"], loc["latitude"], loc["longitude"], start, end)
        df_30 = resample_to_30min(df)
        for col in VARIABLES:
            df_30[col] = df_30[col] * loc["weight"]
        frames.append(df_30[["datetime_utc"] + VARIABLES])

    # Weighted sum across locations
    combined = frames[0].copy()
    for col in VARIABLES:
        combined[col] = sum(f[col] for f in frames)

    # Rename for clarity
    combined = combined.rename(columns={
        "temperature_2m": "temp_c",
        "wind_speed_10m": "wind_ms",
        "shortwave_radiation": "solar_wm2",
        "precipitation": "precip_mm",
    })

    log.info("UK weather series: %d rows", len(combined))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    default_end   = date.today() - timedelta(days=2)   # archive has ~2-day lag
    default_start = date(default_end.year - 5, default_end.month, default_end.day)
    parser.add_argument("--start",  default=str(default_start))
    parser.add_argument("--end",    default=str(default_end))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()

    log.info("Fetching UK weather %s → %s", args.start, args.end)
    df = fetch_uk_weather(args.start, args.end)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved %d rows → %s", len(df), args.output)


if __name__ == "__main__":
    main()

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


def _last_date_in_csv(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["datetime_utc"], nrows=1, skipfooter=0)
    if df.empty:
        return None
    full = pd.read_csv(path, usecols=["datetime_utc"])
    return pd.to_datetime(full["datetime_utc"]).max().date()


def main() -> None:
    parser = argparse.ArgumentParser()
    default_end   = date.today() - timedelta(days=2)   # archive has ~2-day lag
    default_start = date(default_end.year - 5, default_end.month, default_end.day)
    parser.add_argument("--start",  default=None)
    parser.add_argument("--end",    default=str(default_end))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--append", action="store_true",
                        help="Append to existing file instead of overwriting")
    args = parser.parse_args()

    output_path = Path(args.output)
    end = date.fromisoformat(args.end)

    if args.start is None:
        if args.append and output_path.exists():
            last = _last_date_in_csv(output_path)
            start = (last + timedelta(days=1)) if last else default_start
        else:
            start = default_start
    else:
        start = date.fromisoformat(args.start)

    if start > end:
        log.info("Weather already up to date (last: %s). Nothing to fetch.", start - timedelta(days=1))
        return

    log.info("Fetching UK weather %s → %s", start, end)
    df = fetch_uk_weather(str(start), str(end))

    if args.append and output_path.exists():
        existing = pd.read_csv(output_path)
        existing["datetime_utc"] = pd.to_datetime(existing["datetime_utc"], utc=True)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        combined = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["datetime_utc"])
            .sort_values("datetime_utc")
            .reset_index(drop=True)
        )
        df = combined

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved %d rows → %s", len(df), output_path)


if __name__ == "__main__":
    main()

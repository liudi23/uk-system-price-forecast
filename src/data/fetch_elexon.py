"""
Fetch UK electricity system prices (SSP/SBP) from the Elexon Insights Solution API.

API:    https://data.elexon.co.uk/bmrs/api/v1
Auth:   none required
Output: data/raw/system_prices.csv

Usage:
    python src/data/fetch_elexon.py                          # last 30 days
    python src/data/fetch_elexon.py --start 2025-01-01 --end 2025-03-31
    python src/data/fetch_elexon.py --start 2025-01-01 --end 2025-03-31 --append
"""

import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "system_prices.csv"

# Map Elexon camelCase field names → snake_case column names used in the project
COLUMN_MAP = {
    "settlementDate": "settlement_date",
    "settlementPeriod": "settlement_period",
    "systemSellPrice": "ssp",
    "systemBuyPrice": "sbp",
    "netImbalanceVolume": "net_imbalance_volume",
    "sellPriceAdjustment": "sell_price_adjustment",
    "buyPriceAdjustment": "buy_price_adjustment",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_records(payload) -> list:
    """Extract the list of records from a variety of API response shapes."""
    if isinstance(payload, list):
        return payload
    for key in ("data", "items", "results"):
        if key in payload and isinstance(payload[key], list):
            return payload[key]
    return []


def fetch_day(settlement_date: str, session: requests.Session) -> pd.DataFrame:
    """Return a DataFrame of system prices for one settlement date (YYYY-MM-DD)."""
    url = f"{BASE_URL}/balancing/settlement/system-prices/{settlement_date}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    records = _parse_records(resp.json())
    if not records:
        log.warning("No records returned for %s", settlement_date)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    cols = [c for c in COLUMN_MAP.values() if c in df.columns]
    df = df[cols].copy()

    if "settlement_date" in df.columns:
        df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date.astype(str)

    return df.sort_values("settlement_period").reset_index(drop=True)


def fetch_range(start: date, end: date, delay: float = 0.2) -> pd.DataFrame:
    """Fetch system prices for [start, end] inclusive and return a single DataFrame."""
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        log.info("Fetching %s", date_str)
        try:
            df = fetch_day(date_str, session)
            if not df.empty:
                frames.append(df)
        except requests.HTTPError as exc:
            log.error("HTTP %s for %s — skipping", exc.response.status_code, date_str)
        except Exception as exc:
            log.error("Error fetching %s: %s — skipping", date_str, exc)
        current += timedelta(days=1)
        time.sleep(delay)

    if not frames:
        return pd.DataFrame(columns=list(COLUMN_MAP.values()))

    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        existing = pd.read_csv(path)
        existing["settlement_date"] = existing["settlement_date"].astype(str)
        df["settlement_date"] = df["settlement_date"].astype(str)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["settlement_date", "settlement_period"]
        ).sort_values(["settlement_date", "settlement_period"])
        df = combined

    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Elexon system prices (SSP/SBP)")
    default_end = date.today() - timedelta(days=1)
    default_start = default_end - timedelta(days=29)
    parser.add_argument("--start", default=str(default_start), help="Start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end", default=str(default_end), help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV, deduplicating on (date, period)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    if start > end:
        parser.error(f"--start ({start}) must be <= --end ({end})")

    df = fetch_range(start, end)
    save(df, Path(args.output), append=args.append)


if __name__ == "__main__":
    main()

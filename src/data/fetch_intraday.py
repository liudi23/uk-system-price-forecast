"""
Fetch today's partial settlement prices from Elexon BMRS (near real-time).

The same API endpoint used for historical data returns today's Initial Settlement
values as each 30-minute period closes.  By 12:30 UTC (pipeline run time),
SP 1–25 (00:00–12:30 BST) are available.

Output: data/raw/intraday_prices.csv
        Overwritten on every pipeline run with today's data.

Usage:
    python src/data/fetch_intraday.py
"""

import logging
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_elexon import fetch_day

RAW_DIR     = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "intraday_prices.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    today = str(date.today())
    log.info("Fetching intraday prices for %s", today)
    try:
        df = fetch_day(today, requests.Session())
    except Exception as exc:
        log.error("Failed to fetch intraday data: %s", exc)
        return
    if df.empty:
        log.warning("No intraday data returned for %s — skipping", today)
        return
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved %d SPs → %s", len(df), OUTPUT_FILE)


if __name__ == "__main__":
    main()

"""
Fetch 5 years of Elexon SSP data for annual modulation analysis.

Uses ThreadPoolExecutor to parallelise day-level API calls and reduce
wall-clock time from ~12 min (sequential) to ~30 s (20 workers).

Output: data/raw/system_prices_5yr.csv

Usage:
    python src/data/fetch_historical.py
    python src/data/fetch_historical.py --start 2021-01-01 --end 2026-05-17 --workers 20
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from fetch_elexon import COLUMN_MAP, fetch_day

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_FILE = RAW_DATA_DIR / "system_prices_5yr.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_range_parallel(start: date, end: date, workers: int = 20) -> pd.DataFrame:
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    log.info("Fetching %d days with %d workers…", len(dates), workers)
    frames = {}
    failed = []

    def _fetch(d: date):
        session = requests.Session()
        try:
            df = fetch_day(d.strftime("%Y-%m-%d"), session)
            return d, df
        except Exception as exc:
            log.warning("Failed %s: %s", d, exc)
            return d, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, d): d for d in dates}
        done = 0
        for fut in as_completed(futures):
            d, df = fut.result()
            done += 1
            if df is not None and not df.empty:
                frames[d] = df
            else:
                failed.append(d)
            if done % 100 == 0:
                log.info("  %d / %d days fetched", done, len(dates))

    if failed:
        log.warning("Failed dates (%d): %s … ", len(failed), sorted(failed)[:5])

    all_frames = [frames[d] for d in sorted(frames)]
    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    log.info("Fetched %d rows for %d days", len(result), len(frames))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    default_end   = date.today() - timedelta(days=1)
    default_start = date(default_end.year - 5, default_end.month, default_end.day)
    parser.add_argument("--start",   default=str(default_start))
    parser.add_argument("--end",     default=str(default_end))
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output",  default=str(OUTPUT_FILE))
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    t0 = time.time()

    df = fetch_range_parallel(start, end, workers=args.workers)
    df.to_csv(args.output, index=False)
    log.info("Saved %d rows → %s  (%.0f s)", len(df), args.output, time.time() - t0)


if __name__ == "__main__":
    main()

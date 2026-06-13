"""
Fetch UK electricity generation mix (wind %, gas %) from the Carbon Intensity API.

API:    https://api.carbonintensity.org.uk/generation/{from}/{to}
Auth:   none required
Output: data/raw/generation_mix.csv
Cols:   settlement_date, settlement_period, wind_pct, gas_pct

Carbon Intensity uses UTC half-hour periods; Elexon settlement periods use
UK local time (BST/GMT).  Each UTC period "from" time is converted to
Europe/London and the SP index computed as hour*2 + minute//30 + 1.

Usage:
    python src/data/fetch_generation.py                          # last 30 days
    python src/data/fetch_generation.py --start 2021-05-24 --end 2026-05-18
    python src/data/fetch_generation.py --start 2024-01-01 --append
"""

import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BASE_URL     = "https://api.carbonintensity.org.uk/generation"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_FILE  = RAW_DATA_DIR / "generation_mix.csv"
CHUNK_DAYS   = 28   # safe batch size; API handles 30-day ranges reliably

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_chunk(data: list) -> pd.DataFrame:
    """Convert Carbon Intensity generation records → SP-indexed rows."""
    rows = []
    for period in data:
        raw = period["from"]
        # The API returns strings like "2021-05-24T00:00Z"
        utc_ts = pd.Timestamp(raw.replace("Z", "+00:00"))
        uk_ts  = utc_ts.tz_convert("Europe/London")

        sp    = uk_ts.hour * 2 + uk_ts.minute // 30 + 1
        sdate = str(uk_ts.date())

        mix = {g["fuel"]: g["perc"] for g in period.get("generationmix", [])}
        rows.append({
            "settlement_date":   sdate,
            "settlement_period": sp,
            "wind_pct": float(mix.get("wind", 0.0)),
            "gas_pct":  float(mix.get("gas",  0.0)),
        })
    return pd.DataFrame(rows)


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_chunk(from_dt: str, to_dt: str, session: requests.Session) -> pd.DataFrame:
    url  = f"{BASE_URL}/{from_dt}/{to_dt}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        return pd.DataFrame()
    return _parse_chunk(data)


def fetch_range(start: date, end: date, delay: float = 0.4) -> pd.DataFrame:
    """Fetch generation mix for [start, end] (settlement dates) inclusive."""
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    current = start

    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)

        # Query UTC range 1 h wider on each side so BST SP1 (23:00Z prev day)
        # is always included and chunk boundaries don't lose a period.
        from_dt = f"{(current - timedelta(days=1)).strftime('%Y-%m-%d')}T22:30Z"
        to_dt   = f"{(chunk_end + timedelta(days=1)).strftime('%Y-%m-%d')}T01:30Z"

        log.info("Fetching %s → %s", current, chunk_end)
        try:
            df = fetch_chunk(from_dt, to_dt, session)
            if not df.empty:
                df = df[
                    (df["settlement_date"] >= str(current)) &
                    (df["settlement_date"] <= str(chunk_end))
                ]
                frames.append(df)
        except requests.HTTPError as exc:
            log.error("HTTP %s for chunk %s→%s — skipping",
                      exc.response.status_code, current, chunk_end)
        except Exception as exc:
            log.error("Error fetching %s→%s: %s — skipping", current, chunk_end, exc)

        current = chunk_end + timedelta(days=1)
        time.sleep(delay)

    if not frames:
        return pd.DataFrame(columns=["settlement_date", "settlement_period",
                                     "wind_pct", "gas_pct"])

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["settlement_date", "settlement_period"])
        .sort_values(["settlement_date", "settlement_period"])
        .reset_index(drop=True)
    )


# ── Convenience loader (used by build_features and forecast) ──────────────────

def load_generation(path: Path = OUTPUT_FILE) -> pd.DataFrame:
    """Load generation_mix.csv and return with typed columns."""
    df = pd.read_csv(path, dtype={"settlement_date": str,
                                  "settlement_period": int,
                                  "wind_pct": float,
                                  "gas_pct":  float})
    return df


# ── I/O ───────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        existing = load_generation(path)
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["settlement_date", "settlement_period"])
            .sort_values(["settlement_date", "settlement_period"])
        )
    df.to_csv(path, index=False)
    log.info("Saved %d rows → %s", len(df), path)


def _last_date(path: Path) -> Optional[date]:
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=["settlement_date"])
    return date.fromisoformat(df["settlement_date"].max()) if not df.empty else None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch UK generation mix (wind %, gas %) from Carbon Intensity API")
    default_end = date.today() - timedelta(days=1)
    parser.add_argument("--start",  default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=str(default_end), help="End date YYYY-MM-DD")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    path = Path(args.output)
    end  = date.fromisoformat(args.end)

    if args.start is None:
        last = _last_date(path)
        start = (last + timedelta(days=1)) if last else (end - timedelta(days=29))
        if last:
            args.append = True
    else:
        start = date.fromisoformat(args.start)

    if start > end:
        log.info("Already up to date (last=%s). Nothing to fetch.",
                 start - timedelta(days=1))
        return

    log.info("Fetching generation mix %s → %s", start, end)
    df = fetch_range(start, end)
    save(df, path, append=args.append)
    log.info("Done — %d rows, %d dates",
             len(df), df["settlement_date"].nunique() if not df.empty else 0)


if __name__ == "__main__":
    main()

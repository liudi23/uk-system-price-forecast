"""
Fetch UK CPIH (Consumer Prices Index including Housing) from the ONS API.

API:    https://api.ons.gov.uk/v1/dataset/cpih01/timeseries/L55O/data
Auth:   none required
Output: data/raw/cpi_uk.csv
Cols:   year_month (YYYY-MM), cpi_index (float)

Usage:
    python src/data/fetch_cpi.py
    python src/data/fetch_cpi.py --output data/raw/cpi_uk.csv
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

# D7BT = CPIH All items index (2015=100), monthly, updated ~monthly with ~1-month lag
ONS_URL     = "https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/d7bt/data"
RAW_DIR     = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "cpi_uk.csv"

TIMEOUT      = 60
MAX_RETRIES  = 3
BACKOFF_BASE = 5   # seconds; delays: 5, 10, 20

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _request_with_retry(url: str, params=None):
    """GET with exponential backoff. Raises on final failure."""
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            wait = BACKOFF_BASE * (2 ** attempt)
            log.warning("attempt %d/%d failed (%s: %s); retrying in %ds",
                        attempt + 1, MAX_RETRIES, type(exc).__name__, exc, wait)
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    raise last_exc


def fetch_cpi() -> pd.DataFrame:
    """Fetch monthly UK CPIH from ONS API and return as DataFrame."""
    log.info("Fetching UK CPIH from ONS API …")
    resp = _request_with_retry(ONS_URL)
    data = resp.json()

    months = data.get("months", [])
    if not months:
        raise ValueError("ONS API returned no monthly data")

    rows = []
    for m in months:
        try:
            # date format: "2024 JAN"
            dt = pd.to_datetime(m["date"], format="%Y %b")
            year_month = dt.strftime("%Y-%m")
            cpi_index  = float(m["value"])
            rows.append({"year_month": year_month, "cpi_index": cpi_index})
        except (KeyError, ValueError):
            continue

    df = (
        pd.DataFrame(rows)
        .drop_duplicates("year_month")
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    log.info("  %d monthly CPI observations (%s → %s)",
             len(df), df["year_month"].iloc[0], df["year_month"].iloc[-1])
    return df


def load_cpi(path: Path = OUTPUT_FILE) -> pd.DataFrame:
    """Load cpi_uk.csv and return with typed columns."""
    return pd.read_csv(path, dtype={"year_month": str, "cpi_index": float})


def get_latest_cpi(path: Path = OUTPUT_FILE) -> float:
    """Return the most recent CPI index value."""
    df = load_cpi(path)
    return float(df["cpi_index"].iloc[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch UK CPIH from ONS API")
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    args = parser.parse_args()

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = fetch_cpi()
        df.to_csv(path, index=False)
        log.info("Saved %d rows → %s", len(df), path)
    except Exception as exc:
        log.warning("ONS API failed after %d retries: %s", MAX_RETRIES, exc)
        if path.exists():
            log.warning(
                "Using stale CPI from %s — acceptable (CPI changes monthly, "
                "1-day-stale data has negligible impact on training)", path
            )
        else:
            log.error(
                "No stale CPI file at %s — downstream extend_dataset.py may fail "
                "if CPI join is required", path
            )
        # Always exit 0: a CPI API outage must not kill the forecast run.


if __name__ == "__main__":
    main()

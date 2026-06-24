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

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_elexon import fetch_day

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ for shared helpers
from timeutils import uk_today

RAW_DIR     = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "intraday_prices.csv"
# Committed heartbeat so a silent fetch failure (which writes nothing) is
# distinguishable from an upstream gap (fetch succeeds but returns sparse data).
STATUS_FILE = Path(__file__).resolve().parents[2] / "model_assets" / "intraday_fetch_status.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    # Settlement day is UK-local. Use Europe/London, not the runner's UTC date,
    # or in the 23:00–00:00 UTC window during BST we'd fetch the previous UK day.
    today = str(uk_today())
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

    # Record a success heartbeat (last_fetch_ok + count) so a sustained fetch
    # failure can be detected even though main() never crashes (exit-0 hardening).
    _max_sp = int(df["settlement_period"].max()) if "settlement_period" in df.columns else None
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "last_fetch_ok":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M"),
        "settlement_date": today,
        "n_sps":           len(df),
        "max_sp":          _max_sp,
    }))
    log.info("Saved %d SPs (max SP %s) → %s", len(df), _max_sp, OUTPUT_FILE)


if __name__ == "__main__":
    main()

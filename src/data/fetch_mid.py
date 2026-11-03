"""
Fetch Elexon Market Index Data (MID) — the volume-weighted GB short-term
market price per settlement period (APX + N2EX providers combined).

MID is an OUTTURN series (published shortly after each SP trades), so it may
only enter the models as LAGGED features (mid_price_lag_48 etc.) — it is NOT
known day-ahead. It supplies the market-price information the feature set
lacked (Phase 5b wave 1): the spread between SSP and the market index is the
imbalance premium, whose dynamics lead the level regime.

API:    https://data.elexon.co.uk/bmrs/api/v1/datasets/MID?from=...&to=...
Fields: dataset, startTime, dataProvider (APXMIDP | N2EXMIDP),
        settlementDate, settlementPeriod, price, volume
Output: data/vintages/mid_prices.csv
        settlement_date, settlement_period, mid_price (volume-weighted),
        mid_volume (summed)

Usage:
    python src/data/fetch_mid.py --backfill 2021-05-17 2026-11-02
    python src/data/fetch_mid.py --append          # top up from last stored date
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_bmrs_forecasts import _fetch_json, BMRS_BASE  # shared retry helpers

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "data" / "vintages" / "mid_prices.csv"
CHUNK_DAYS  = 7      # MID from/to range per request
SLEEP_S     = 0.2    # politeness between requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_mid_range(start: date, end: date) -> pd.DataFrame:
    """Fetch MID for [start, end] inclusive, volume-weighting across providers."""
    frames = []
    d = start
    while d <= end:
        chunk_end = min(d + timedelta(days=CHUNK_DAYS - 1), end)
        records = _fetch_json(f"{BMRS_BASE}/MID",
                              {"from": str(d), "to": str(chunk_end), "format": "json"})
        if records:
            frames.append(pd.DataFrame(records))
        log.info("MID %s → %s: %d records", d, chunk_end, len(records))
        d = chunk_end + timedelta(days=1)
        time.sleep(SLEEP_S)
    if not frames:
        return pd.DataFrame(columns=["settlement_date", "settlement_period",
                                     "mid_price", "mid_volume"])
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"settlementDate": "settlement_date",
                            "settlementPeriod": "settlement_period"})
    df["price"]  = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["price"])
    # Volume-weighted price across providers; zero-volume rows carry no weight.
    df["_pv"] = df["price"] * df["volume"]
    g = df.groupby(["settlement_date", "settlement_period"]).agg(
        _pv=("_pv", "sum"), mid_volume=("volume", "sum"),
        _p_mean=("price", "mean"))
    g["mid_price"] = (g["_pv"] / g["mid_volume"]).where(g["mid_volume"] > 0, g["_p_mean"])
    out = g.reset_index()[["settlement_date", "settlement_period", "mid_price", "mid_volume"]]
    out["settlement_period"] = out["settlement_period"].astype(int)
    return out.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


def _merge_save(new: pd.DataFrame) -> None:
    if OUTPUT_FILE.exists():
        old = pd.read_csv(OUTPUT_FILE)
        combined = (pd.concat([old, new], ignore_index=True)
                    .drop_duplicates(subset=["settlement_date", "settlement_period"],
                                     keep="last")
                    .sort_values(["settlement_date", "settlement_period"]))
    else:
        combined = new
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    log.info("Saved %d rows → %s (through %s)", len(combined), OUTPUT_FILE,
             combined["settlement_date"].max() if len(combined) else "—")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"),
                    help="Fetch full range START END (YYYY-MM-DD)")
    ap.add_argument("--append", action="store_true",
                    help="Fetch from last stored date through yesterday")
    args = ap.parse_args()

    if args.backfill:
        start, end = date.fromisoformat(args.backfill[0]), date.fromisoformat(args.backfill[1])
    elif args.append:
        if OUTPUT_FILE.exists():
            last = pd.read_csv(OUTPUT_FILE)["settlement_date"].max()
            start = date.fromisoformat(str(last)[:10]) - timedelta(days=1)  # re-fetch last day (late MID postings)
        else:
            start = date.today() - timedelta(days=30)
        end = date.today()
    else:
        ap.error("one of --backfill / --append is required")

    log.info("Fetching MID %s → %s", start, end)
    _merge_save(fetch_mid_range(start, end))


if __name__ == "__main__":
    main()

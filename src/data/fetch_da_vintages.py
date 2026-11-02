"""
Fetch DAY-AHEAD VINTAGES of NESO forecasts from Elexon BMRS: what the system
operator's forecasts for day D said at 12:00 UTC on D−1 — the information the
12:30 UTC daily pipeline actually has when it forecasts day D.

Phase 5b wave 1. Leakage discipline: for every target day D, only publications
with publishTime ≤ (D−1)T12:00Z are used; the latest such publication wins.
Training on these vintages (not on outturn, not on the latest revision) keeps
the zero-leakage guarantee intact.

Datasets
────────
  WINDFOR  — wind generation forecast, hourly MW, ~8 publications/day
  TSDF     — transmission system demand forecast, SP-level MW, boundary 'N'
  OCNMFD   — generating-plant operating-surplus (derated margin) forecast,
             daily MW per forecastDate, ~hourly publications

Output: data/vintages/da_vintages.csv
        settlement_date, settlement_period, windfor_fc_mw, tsdf_fc_mw,
        surplus_fc_mw (daily value replicated across the day's SPs)

Usage:
    python src/data/fetch_da_vintages.py --backfill 2021-05-17 2026-11-02
    python src/data/fetch_da_vintages.py --append
"""

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_bmrs_forecasts import _fetch_json, BMRS_BASE  # shared retry helpers

OUTPUT_FILE = Path(__file__).resolve().parents[2] / "data" / "vintages" / "da_vintages.csv"
SLEEP_S     = 0.15
CUTOFF_HOUR = 12   # vintage cutoff: (D-1)T12:00Z — what the 12:30 UTC run can see

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _publish_window(target: date) -> tuple[str, str]:
    """Publication window queried for target day D: (D−2)T12:00Z → (D−1)T12:00Z.

    TSDF/OCNMFD publish less often than WINDFOR; a 24 h window guarantees at
    least one publication of each while keeping responses small.
    """
    lo = f"{target - timedelta(days=2)}T{CUTOFF_HOUR:02d}:00Z"
    hi = f"{target - timedelta(days=1)}T{CUTOFF_HOUR:02d}:00Z"
    return lo, hi


def _latest_per_key(df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """Keep the row with the latest publishTime per key."""
    return (df.sort_values("publishTime")
              .drop_duplicates(subset=key_cols, keep="last"))


def fetch_day(target: date) -> pd.DataFrame:
    """Return the D−1 12:00Z vintage rows (48 SPs) for one target day."""
    lo, hi = _publish_window(target)
    base = {"publishDateTimeFrom": lo, "publishDateTimeTo": hi, "format": "json"}

    # ── WINDFOR: hourly UTC startTime → two SPs per hour ──────────────────
    wind = pd.DataFrame()
    try:
        rec = _fetch_json(f"{BMRS_BASE}/WINDFOR", dict(base))
        if rec:
            w = pd.DataFrame(rec)
            w["startTime"] = pd.to_datetime(w["startTime"], utc=True)
            w = _latest_per_key(w, ["startTime"])
            rows = []
            for _, r in w.iterrows():
                for off in (0, 30):
                    uk = (r["startTime"] + pd.Timedelta(minutes=off)).tz_convert("Europe/London")
                    rows.append({"settlement_date": str(uk.date()),
                                 "settlement_period": uk.hour * 2 + uk.minute // 30 + 1,
                                 "windfor_fc_mw": float(r["generation"])})
            wind = pd.DataFrame(rows)
            wind = wind[wind["settlement_date"] == str(target)]
    except Exception as exc:
        log.warning("WINDFOR vintage failed for %s: %s", target, exc)
    time.sleep(SLEEP_S)

    # ── TSDF: SP-level, boundary N ────────────────────────────────────────
    dem = pd.DataFrame()
    try:
        rec = _fetch_json(f"{BMRS_BASE}/TSDF", dict(base, boundary="N"))
        if rec:
            t = pd.DataFrame(rec)
            if "boundary" in t.columns:
                t = t[t["boundary"] == "N"]
            t = t[t["settlementDate"] == str(target)]
            t = _latest_per_key(t, ["settlementDate", "settlementPeriod"])
            dem = t.rename(columns={"settlementDate": "settlement_date",
                                    "settlementPeriod": "settlement_period",
                                    "demand": "tsdf_fc_mw"})[
                ["settlement_date", "settlement_period", "tsdf_fc_mw"]]
    except Exception as exc:
        log.warning("TSDF vintage failed for %s: %s", target, exc)
    time.sleep(SLEEP_S)

    # ── OCNMFD: daily surplus for forecastDate == target ──────────────────
    surplus = np.nan
    try:
        rec = _fetch_json(f"{BMRS_BASE}/OCNMFD", dict(base))
        if rec:
            o = pd.DataFrame(rec)
            o = o[o["forecastDate"] == str(target)]
            if len(o):
                surplus = float(_latest_per_key(o, ["forecastDate"])["surplus"].iloc[0])
    except Exception as exc:
        log.warning("OCNMFD vintage failed for %s: %s", target, exc)
    time.sleep(SLEEP_S)

    # ── Combine on the 48-SP grid ─────────────────────────────────────────
    out = pd.DataFrame({"settlement_date": str(target),
                        "settlement_period": range(1, 49)})
    if not wind.empty:
        out = out.merge(wind.drop_duplicates(["settlement_date", "settlement_period"]),
                        on=["settlement_date", "settlement_period"], how="left")
    else:
        out["windfor_fc_mw"] = np.nan
    if not dem.empty:
        dem["settlement_period"] = dem["settlement_period"].astype(int)
        out = out.merge(dem, on=["settlement_date", "settlement_period"], how="left")
    else:
        out["tsdf_fc_mw"] = np.nan
    out["surplus_fc_mw"] = surplus
    return out


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


def _existing_dates() -> set:
    if not OUTPUT_FILE.exists():
        return set()
    df = pd.read_csv(OUTPUT_FILE)
    # a date counts as done only if its windfor OR tsdf vintage is present
    has_data = df.groupby("settlement_date")[["windfor_fc_mw", "tsdf_fc_mw"]].count().sum(axis=1)
    return set(has_data[has_data > 0].index.astype(str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", nargs=2, metavar=("START", "END"))
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--flush-every", type=int, default=30,
                    help="Write CSV every N days during backfill (resumable)")
    args = ap.parse_args()

    if args.backfill:
        start, end = date.fromisoformat(args.backfill[0]), date.fromisoformat(args.backfill[1])
    elif args.append:
        if OUTPUT_FILE.exists():
            last = pd.read_csv(OUTPUT_FILE)["settlement_date"].max()
            start = date.fromisoformat(str(last)[:10])  # re-fetch last stored day
        else:
            start = date.today() - timedelta(days=30)
        end = date.today() + timedelta(days=1)  # tomorrow's vintage exists after 12:00Z today
    else:
        ap.error("one of --backfill / --append is required")

    done = _existing_dates() if args.backfill else set()
    buf, d, n = [], start, 0
    while d <= end:
        if str(d) not in done:
            buf.append(fetch_day(d))
            n += 1
            if n % args.flush_every == 0:
                _merge_save(pd.concat(buf, ignore_index=True)); buf = []
                log.info("…progress: %s (%d new days)", d, n)
        d += timedelta(days=1)
    if buf:
        _merge_save(pd.concat(buf, ignore_index=True))
    log.info("Done — %d day(s) fetched", n)


if __name__ == "__main__":
    main()

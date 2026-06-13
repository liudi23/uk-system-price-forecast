"""
Fetch day-ahead wind and demand forecasts from Elexon BMRS Insights API.

Products fetched
───────────────
  WINDFOR  — Wind Power Generation Forecast (MW, hourly UTC)
             Published by NGESO; available ~10:00 D-1 for all 48 SPs of day D.
             Endpoint: /datasets/WINDFOR?settlementDate=YYYY-MM-DD

  TSDF     — Transmission System Demand Forecast (MW, settlement-period level)
             Boundary 'N' = National total GB demand.
             Endpoint: /datasets/TSDF?settlementDate=YYYY-MM-DD

Derived output
──────────────
  wind_pct_forecast[sp] = WINDFOR[sp] / TSDF[sp] * 100
  This mirrors the Carbon Intensity wind_pct definition (% of total generation)
  but uses genuine day-ahead forecasts rather than post-hoc actuals.

Usage
─────
  from fetch_bmrs_forecasts import get_wind_pct_forecast
  arr = get_wind_pct_forecast(date(2026, 6, 5))  # np.array, 48 values

  python src/data/fetch_bmrs_forecasts.py --date 2026-06-05
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BMRS_BASE = "https://data.elexon.co.uk/bmrs/api/v1/datasets"
UK_WIND_CAPACITY_MW = 32_000   # fallback denominator when TSDF unavailable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _utc_to_uk_sp(utc_ts: pd.Timestamp) -> tuple[str, int]:
    """Convert UTC timestamp → (UK settlement date string, settlement period)."""
    uk = utc_ts.tz_convert("Europe/London")
    sp = uk.hour * 2 + uk.minute // 30 + 1
    return str(uk.date()), sp


def _fetch_json(url: str, params: dict, timeout: int = 20) -> list:
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json().get("data", [])


# ── WINDFOR ───────────────────────────────────────────────────────────────────

def fetch_windfor(target_date: date) -> pd.DataFrame:
    """
    Fetch WINDFOR for target_date.  Returns a DataFrame indexed by settlement
    period (1-48) with column wind_mw.  WINDFOR is hourly UTC; each hour is
    forward-filled into the two 30-min SPs that span it.
    """
    records = _fetch_json(
        f"{BMRS_BASE}/WINDFOR",
        {"settlementDate": str(target_date), "format": "json"},
    )
    if not records:
        log.warning("WINDFOR: no data for %s", target_date)
        return pd.DataFrame()

    rows = []
    for r in records:
        try:
            utc = pd.Timestamp(r["startTime"]).tz_convert("UTC")
            sdate, sp = _utc_to_uk_sp(utc)
            # Also cover the +30 min slot within the same hour
            utc30 = utc + pd.Timedelta(minutes=30)
            sdate30, sp30 = _utc_to_uk_sp(utc30)
            mw = float(r["generation"])
            rows.append({"settlement_date": sdate,  "settlement_period": sp,   "wind_mw": mw})
            rows.append({"settlement_date": sdate30, "settlement_period": sp30, "wind_mw": mw})
        except (KeyError, ValueError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df[df["settlement_date"] == str(target_date)]
          .drop_duplicates("settlement_period")
          .sort_values("settlement_period")
          .set_index("settlement_period"))
    return df


# ── TSDF (National demand forecast) ──────────────────────────────────────────

def fetch_tsdf(target_date: date) -> pd.DataFrame:
    """
    Fetch TSDF boundary='N' (National total demand forecast) for target_date.
    Returns a DataFrame indexed by settlement period (1-48) with column demand_mw.
    Takes the latest publish run available.
    """
    records = _fetch_json(
        f"{BMRS_BASE}/TSDF",
        {"settlementDate": str(target_date), "format": "json"},
    )
    if not records:
        log.warning("TSDF: no data for %s", target_date)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df[df["boundary"] == "N"]
    if df.empty:
        return pd.DataFrame()

    df["publishTime"] = pd.to_datetime(df["publishTime"])
    # Take the latest published forecast run
    latest = df[df["publishTime"] == df["publishTime"].max()]
    latest = (latest[["settlementPeriod", "demand"]]
              .drop_duplicates("settlementPeriod")
              .rename(columns={"demand": "demand_mw", "settlementPeriod": "settlement_period"})
              .sort_values("settlement_period")
              .set_index("settlement_period"))
    return latest


# ── Combined: wind % forecast ─────────────────────────────────────────────────

def get_wind_pct_forecast(target_date: date) -> np.ndarray:
    """
    Return a 48-element array of day-ahead wind % forecast for target_date.

    wind_pct_forecast[sp-1] = WINDFOR_SP / TSDF_SP * 100

    Falls back gracefully:
      - If TSDF unavailable: use UK_WIND_CAPACITY_MW as denominator
      - If WINDFOR unavailable: return array of NaN (caller handles)
      - Missing SPs filled by linear interpolation then edge-pad
    """
    windfor = fetch_windfor(target_date)
    tsdf    = fetch_tsdf(target_date)

    # Build SP-indexed arrays (1-indexed → 0-indexed)
    wind_mw   = np.full(48, np.nan)
    demand_mw = np.full(48, np.nan)

    if not windfor.empty:
        for sp, row in windfor.iterrows():
            if 1 <= sp <= 48:
                wind_mw[sp - 1] = row["wind_mw"]

    if not tsdf.empty:
        for sp, row in tsdf.iterrows():
            if 1 <= sp <= 48:
                demand_mw[sp - 1] = row["demand_mw"]
    else:
        log.warning("TSDF unavailable — using fixed capacity denominator %d MW", UK_WIND_CAPACITY_MW)
        demand_mw[:] = UK_WIND_CAPACITY_MW

    # Interpolate missing SPs
    s = pd.Series(wind_mw)
    s = s.interpolate("linear").ffill().bfill()
    wind_mw = s.values

    d = pd.Series(demand_mw)
    d = d.interpolate("linear").ffill().bfill()
    demand_mw = d.values

    # Compute %
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(demand_mw > 0, wind_mw / demand_mw * 100, np.nan)

    n_valid = np.isfinite(pct).sum()
    log.info("WINDFOR/TSDF for %s: %d/48 SPs valid, wind %% range %.1f–%.1f%%",
             target_date, n_valid, np.nanmin(pct), np.nanmax(pct))
    return pct


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Fetch BMRS WINDFOR + TSDF day-ahead forecasts")
    parser.add_argument("--date", default=str(date.today() + timedelta(days=1)),
                        help="Target settlement date YYYY-MM-DD (default: tomorrow)")
    args = parser.parse_args()
    d = date.fromisoformat(args.date)
    pct = get_wind_pct_forecast(d)
    print(f"\nDay-ahead wind % forecast for {d}:")
    for i, v in enumerate(pct, 1):
        print(f"  SP{i:2d}: {v:.1f}%")


if __name__ == "__main__":
    main()

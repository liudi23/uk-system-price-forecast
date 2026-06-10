"""
Fetch UK electricity system prices (SSP/SBP) from the Elexon Insights API.

Usage:
    python src/data/fetch_elexon.py                          # last 12 months
    python src/data/fetch_elexon.py --start 2024-01-01       # from date to today
    python src/data/fetch_elexon.py --start 2024-01-01 --end 2024-06-30
    python src/data/fetch_elexon.py --months 6               # last N months

Output:
    data/raw/system_prices.csv
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

COLUMNS = {
    "settlementDate": "settlement_date",
    "settlementPeriod": "settlement_period",
    "systemSellPrice": "ssp",
    "systemBuyPrice": "sbp",
    "netImbalanceVolume": "net_imbalance_volume",
    "sellPriceAdjustment": "sell_price_adjustment",
    "buyPriceAdjustment": "buy_price_adjustment",
    "totalSystemAcceptedOfferVolume": "total_accepted_offer_mwh",
    "totalSystemAcceptedBidVolume": "total_accepted_bid_mwh",
}


def fetch_system_prices_for_date(session: requests.Session, settlement_date: date) -> list[dict]:
    url = f"{BASE_URL}/balancing/settlement/system-prices/{settlement_date.isoformat()}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_range(start: date, end: date, delay: float = 0.3) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    records = []
    current = start
    total_days = (end - start).days + 1

    print(f"Fetching system prices: {start} to {end} ({total_days} days)")

    while current <= end:
        try:
            daily = fetch_system_prices_for_date(session, current)
            records.extend(daily)
            print(f"  {current}  {len(daily)} periods")
        except requests.HTTPError as e:
            print(f"  {current}  HTTP error: {e}")
        except requests.RequestException as e:
            print(f"  {current}  request failed: {e}")

        current += timedelta(days=1)
        time.sleep(delay)

    if not records:
        raise RuntimeError("No data returned for the requested date range.")

    df = pd.DataFrame(records)

    rename = {k: v for k, v in COLUMNS.items() if k in df.columns}
    df = df.rename(columns=rename)

    keep = list(rename.values())
    df = df[[c for c in keep if c in df.columns]]

    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    df = df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Elexon system prices.")
    parser.add_argument("--start", type=date.fromisoformat, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--months", type=int, default=12, help="Lookback months if --start not given (default: 12)")
    return parser.parse_args()


def main():
    args = parse_args()
    today = date.today()
    end = args.end or today
    start = args.start or (today - timedelta(days=args.months * 30))

    df = fetch_range(start, end)

    out_path = RAW_DIR / "system_prices.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()

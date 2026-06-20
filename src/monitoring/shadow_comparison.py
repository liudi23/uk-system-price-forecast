"""
shadow_comparison.py
────────────────────
Compare YESTERDAY's shadow forecast vs production forecast against actuals.

Run daily from forecast_pipeline.yml (shadow mode) at 01:00 UTC on day X+1:
  shadow file : model_assets/forecasts/forecast_phase3_{X}_shadow.csv
  prod file   : model_assets/forecasts/forecast_phase3_{X}.csv
  actuals     : data/raw/system_prices.csv  (IIS, available by X+1 01:00 UTC)

Both archive files are pure model predictions (all is_actual=False) because
forecast_phase3.py skips the archive write when is_actual rows are present
(commit 7407648 — intraday leaves archives untouched).

q10/q90 in forecast files are already PI-calibrated (applied by forecast_phase3.py
when pi_calibration_v1.json exists); no further δ adjustment is needed here.

Output: reports/shadow_validation/log.csv  (one row appended per run)

Pass criterion over 2-week shadow period:
  MAE delta      |shadow_mae − prod_mae|  ≤ £0.50
  Coverage delta |shadow_cov − prod_cov| ≤ 3 pp
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO      = Path(__file__).resolve().parents[2]
ASSETS    = REPO / "model_assets"
FORECASTS = ASSETS / "forecasts"
SP_CSV    = REPO / "data" / "raw" / "system_prices.csv"
LOG       = REPO / "reports" / "shadow_validation" / "log.csv"

MAE_THRESHOLD  = 0.50   # £/MWh
COV_THRESHOLD  = 0.03   # 3 pp as fraction


def _load_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "is_actual" in df.columns:
        df = df[~df["is_actual"].astype(bool)]
    return df[["settlement_date", "settlement_period", "ssp_q10", "ssp_q50", "ssp_q90"]]


def _metrics(df: pd.DataFrame) -> dict:
    actual = df["actual"].values
    q50    = df["ssp_q50"].values
    q10    = df["ssp_q10"].values
    q90    = df["ssp_q90"].values
    mae      = float(np.mean(np.abs(actual - q50)))
    coverage = float(np.mean((actual >= q10) & (actual <= q90)))
    width    = float(np.mean(q90 - q10))
    return {"mae": mae, "coverage": coverage, "width": width}


def main(target_date_str=None) -> None:
    if target_date_str is None:
        target_date = date.today() - timedelta(days=1)
        target_date_str = target_date.isoformat()
    else:
        target_date = date.fromisoformat(target_date_str)

    shadow_path = FORECASTS / f"forecast_phase3_{target_date_str}_shadow.csv"
    prod_path   = FORECASTS / f"forecast_phase3_{target_date_str}.csv"

    # Both files must exist — day 1 of shadow has no shadow archive yet
    if not shadow_path.exists():
        print(f"[shadow_comparison] No shadow archive for {target_date_str}: "
              f"{shadow_path.name} — skipping (expected on day 1)")
        return
    if not prod_path.exists():
        print(f"[shadow_comparison] No production archive for {target_date_str}: "
              f"{prod_path.name} — skipping")
        return

    shadow_df = _load_forecast(shadow_path)
    prod_df   = _load_forecast(prod_path)

    # Actuals for target_date from system_prices.csv
    actuals = pd.read_csv(SP_CSV, parse_dates=["settlement_date"])
    actuals = actuals[actuals["settlement_date"].dt.date == target_date].copy()

    if actuals.empty:
        print(f"[shadow_comparison] No actuals in system_prices.csv for {target_date_str} — skipping")
        return

    actuals = actuals[["settlement_period", "ssp"]].rename(columns={"ssp": "actual"})

    # Join forecasts to actuals; use intersection of settlement_periods
    shadow_j = shadow_df.merge(actuals, on="settlement_period", how="inner")
    prod_j   = prod_df.merge(actuals,   on="settlement_period", how="inner")

    common_sps = set(shadow_j["settlement_period"]) & set(prod_j["settlement_period"])
    shadow_j = shadow_j[shadow_j["settlement_period"].isin(common_sps)]
    prod_j   = prod_j[prod_j["settlement_period"].isin(common_sps)]

    n = len(common_sps)
    if n == 0:
        print(f"[shadow_comparison] No overlapping settlement periods for {target_date_str} — skipping")
        return

    sm = _metrics(shadow_j)
    pm = _metrics(prod_j)

    mae_delta = sm["mae"] - pm["mae"]
    cov_delta = sm["coverage"] - pm["coverage"]

    pass_mae = abs(mae_delta) <= MAE_THRESHOLD
    pass_cov = abs(cov_delta) <= COV_THRESHOLD

    row = {
        "date":         target_date_str,
        "n_sps":        n,
        "shadow_mae":   round(sm["mae"],      3),
        "prod_mae":     round(pm["mae"],      3),
        "mae_delta":    round(mae_delta,      3),
        "shadow_cov":   round(sm["coverage"], 4),
        "prod_cov":     round(pm["coverage"], 4),
        "cov_delta_pp": round(cov_delta * 100, 2),
        "shadow_width": round(sm["width"],    2),
        "prod_width":   round(pm["width"],    2),
        "pass_mae":     pass_mae,
        "pass_cov":     pass_cov,
    }

    LOG.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not LOG.exists() or LOG.stat().st_size == 0
    df_row.to_csv(LOG, mode="a", index=False, header=write_header)

    print(f"[shadow_comparison] {target_date_str}  (n={n} SPs):")
    print(f"  MAE    — shadow £{sm['mae']:.2f}  prod £{pm['mae']:.2f}  Δ={mae_delta:+.2f}  "
          f"{'PASS' if pass_mae else 'FAIL ⚠️'} (threshold ±£{MAE_THRESHOLD:.2f})")
    print(f"  Cov    — shadow {sm['coverage']:.1%}  prod {pm['coverage']:.1%}  "
          f"Δ={cov_delta*100:+.1f}pp  {'PASS' if pass_cov else 'FAIL ⚠️'} (threshold ±{COV_THRESHOLD*100:.0f}pp)")
    print(f"  Width  — shadow £{sm['width']:.1f}  prod £{pm['width']:.1f}")
    print(f"  → {LOG}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday UTC)")
    args = p.parse_args()
    main(args.date)

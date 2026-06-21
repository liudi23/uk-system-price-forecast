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

log.csv schema
──────────────
  date            YYYY-MM-DD being compared (yesterday)
  n_sps           number of SP-rows with both forecast and actuals
  shadow_mae      shadow model MAE vs actuals (£/MWh)
  prod_mae        production model MAE vs actuals; NaN when prod archive absent
  mae_delta       shadow_mae − prod_mae; NaN when prod archive absent
  shadow_cov      shadow 80%-band coverage (fraction)
  prod_cov        production 80%-band coverage; NaN when prod archive absent
  cov_delta_pp    (shadow_cov − prod_cov) × 100 pp; NaN when prod archive absent
  shadow_width    shadow mean PI width (£)
  prod_width      production mean PI width; NaN when prod archive absent
  pass_mae        |mae_delta| ≤ £0.50; False when NaN
  pass_cov        |cov_delta| ≤ 3 pp; False when NaN
  shadow_only     True when prod archive absent (delta columns are NaN)

Pass criteria (full comparison): |mae_delta| ≤ £0.50 · |cov_delta| ≤ 3 pp

Every run emits a one-line status to stdout so GitHub Actions run logs
always show what happened, even on a skip:

  [shadow_comparison] 2026-06-21 · shadow=✓ · prod=✓ · actuals=✓(48 SPs) → wrote row (full)
  [shadow_comparison] 2026-06-21 · shadow=✓ · prod=✗ · actuals=✓(48 SPs) → wrote row (shadow-only)
  [shadow_comparison] 2026-06-20 · shadow=✗ · prod=✓ · actuals=✓         → SKIPPED (no shadow archive — expected on day 1)
  [shadow_comparison] 2026-06-21 · shadow=✓ · prod=✓ · actuals=✗         → SKIPPED (no actuals in system_prices.csv)
"""

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _append_log(row: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    write_header = not LOG.exists() or LOG.stat().st_size == 0
    df_row.to_csv(LOG, mode="a", index=False, header=write_header)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(target_date_str=None) -> None:
    if target_date_str is None:
        target_date = date.today() - timedelta(days=1)
        target_date_str = target_date.isoformat()
    else:
        target_date = date.fromisoformat(target_date_str)

    shadow_path = FORECASTS / f"forecast_phase3_{target_date_str}_shadow.csv"
    prod_path   = FORECASTS / f"forecast_phase3_{target_date_str}.csv"

    shadow_ok = shadow_path.exists()
    prod_ok   = prod_path.exists()

    # ── Load actuals ──────────────────────────────────────────────────────────
    actuals_n = 0
    actuals   = pd.DataFrame()
    if SP_CSV.exists():
        raw = pd.read_csv(SP_CSV, parse_dates=["settlement_date"])
        raw = raw[raw["settlement_date"].dt.date == target_date].copy()
        if not raw.empty:
            actuals = raw[["settlement_period", "ssp"]].rename(columns={"ssp": "actual"})
            actuals_n = len(actuals)

    actuals_ok = actuals_n > 0

    # ── Status header (always printed, visible in GitHub Actions log) ─────────
    shadow_sym  = "✓" if shadow_ok  else "✗"
    prod_sym    = "✓" if prod_ok    else "✗"
    actuals_sym = f"✓({actuals_n} SPs)" if actuals_ok else "✗"
    status_pfx  = f"[shadow_comparison] {target_date_str} · shadow={shadow_sym} · prod={prod_sym} · actuals={actuals_sym}"

    # ── Guard: need shadow + actuals to write anything ────────────────────────
    if not shadow_ok:
        reason = "no shadow archive — expected on day 1 of shadow period"
        print(f"{status_pfx} → SKIPPED ({reason})")
        return

    if not actuals_ok:
        reason = "no actuals in system_prices.csv (Elexon publication lag)"
        print(f"{status_pfx} → SKIPPED ({reason})")
        return

    # ── Load shadow forecast and join to actuals ──────────────────────────────
    shadow_df = _load_forecast(shadow_path)
    shadow_j  = shadow_df.merge(actuals, on="settlement_period", how="inner")
    n         = len(shadow_j)

    if n == 0:
        reason = "no overlapping settlement periods"
        print(f"{status_pfx} → SKIPPED ({reason})")
        return

    sm = _metrics(shadow_j)

    # ── Full comparison (shadow + production) ─────────────────────────────────
    if prod_ok:
        prod_df  = _load_forecast(prod_path)
        prod_j   = prod_df.merge(actuals, on="settlement_period", how="inner")
        common   = set(shadow_j["settlement_period"]) & set(prod_j["settlement_period"])
        shadow_j = shadow_j[shadow_j["settlement_period"].isin(common)]
        prod_j   = prod_j[prod_j["settlement_period"].isin(common)]
        n        = len(common)

        sm = _metrics(shadow_j)
        pm = _metrics(prod_j)

        mae_delta = sm["mae"] - pm["mae"]
        cov_delta = sm["coverage"] - pm["coverage"]
        pass_mae  = abs(mae_delta) <= MAE_THRESHOLD
        pass_cov  = abs(cov_delta) <= COV_THRESHOLD

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
            "shadow_only":  False,
        }
        _append_log(row)

        print(f"{status_pfx} → wrote row (full comparison)")
        print(f"  MAE    shadow £{sm['mae']:.2f}  prod £{pm['mae']:.2f}  "
              f"Δ={mae_delta:+.2f}  {'PASS' if pass_mae else 'FAIL ⚠️'}"
              f"  (threshold ±£{MAE_THRESHOLD:.2f})")
        print(f"  Cov    shadow {sm['coverage']:.1%}  prod {pm['coverage']:.1%}  "
              f"Δ={cov_delta * 100:+.1f}pp  {'PASS' if pass_cov else 'FAIL ⚠️'}"
              f"  (threshold ±{COV_THRESHOLD * 100:.0f}pp)")
        print(f"  Width  shadow £{sm['width']:.1f}  prod £{pm['width']:.1f}")
        print(f"  → {LOG}")

    else:
        # ── Shadow-only row (prod archive absent) ─────────────────────────────
        row = {
            "date":         target_date_str,
            "n_sps":        n,
            "shadow_mae":   round(sm["mae"],      3),
            "prod_mae":     float("nan"),
            "mae_delta":    float("nan"),
            "shadow_cov":   round(sm["coverage"], 4),
            "prod_cov":     float("nan"),
            "cov_delta_pp": float("nan"),
            "shadow_width": round(sm["width"],    2),
            "prod_width":   float("nan"),
            "pass_mae":     False,
            "pass_cov":     False,
            "shadow_only":  True,
        }
        _append_log(row)

        print(f"{status_pfx} → wrote row (shadow-only — prod archive absent)")
        print(f"  Shadow MAE £{sm['mae']:.2f}  Coverage {sm['coverage']:.1%}"
              f"  Width £{sm['width']:.1f}  (n={n} SPs)")
        print(f"  prod archive {prod_path.name} missing — delta metrics NaN")
        print(f"  → {LOG}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: yesterday UTC)")
    args = p.parse_args()
    main(args.date)

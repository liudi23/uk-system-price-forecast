"""
Phase 4 PI calibration: split-conformal symmetric widening.

Computes a per-SP calibration offset δ(sp) such that widening the base-model
PI by [q10 − δ, q90 + δ] achieves ~80% empirical coverage.

Method
------
For each settlement period s, the conformity score is:
    score_i = max(q10_i − actual_i,  actual_i − q90_i)
      > 0  ⟹ actual is OUTSIDE the PI (band too narrow)
      < 0  ⟹ actual is inside the PI
δ(s) = 80th percentile of {score_i : SP=s} across all calibration rows.

Widening applied at inference time:
    q10_new(s) = q10(s) − δ(s)
    q90_new(s) = q90(s) + δ(s)

This guarantees ≥80% empirical coverage on the calibration set (by construction
of the conformal prediction guarantee).  δ is clipped at 0 so we never NARROW.

Input  : model_assets/walk_forward_predictions.csv  (all 4 folds, 119 days × 48 SP)
Output : model_assets/pi_calibration_v1.json
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WF_PATH = ROOT / "model_assets" / "walk_forward_predictions.csv"
OUT_PATH = ROOT / "model_assets" / "pi_calibration_v1.json"

TARGET_COVERAGE = 0.80
QUANTILE_LEVEL  = 0.80


def compute_pi_calibration(wf: pd.DataFrame) -> dict:
    """Return the calibration artifact dict from a walk-forward predictions frame."""
    required = {"settlement_period", "ssp_q10", "ssp_q90", "ssp_actual"}
    missing = required - set(wf.columns)
    if missing:
        raise ValueError(f"walk_forward_predictions.csv missing columns: {missing}")

    wf = wf.dropna(subset=["ssp_q10", "ssp_q90", "ssp_actual"]).copy()

    # Conformity score: positive ⟹ actual outside band
    wf["score"] = np.maximum(
        wf["ssp_q10"] - wf["ssp_actual"],
        wf["ssp_actual"] - wf["ssp_q90"],
    )

    # Coverage before calibration
    inside_mask = (wf["ssp_actual"] >= wf["ssp_q10"]) & (wf["ssp_actual"] <= wf["ssp_q90"])
    coverage_before = inside_mask.mean()

    # Per-SP δ: 80th percentile of conformity scores
    sp_delta = (
        wf.groupby("settlement_period")["score"]
        .quantile(QUANTILE_LEVEL)
        .clip(lower=0.0)
    )

    # Global δ (for reference / fallback)
    delta_global = float(np.quantile(wf["score"], QUANTILE_LEVEL))
    delta_global = max(delta_global, 0.0)

    # Implied in-sample coverage after applying per-SP δ
    wf["q10_cal"] = wf["ssp_q10"] - wf["settlement_period"].map(sp_delta)
    wf["q90_cal"] = wf["ssp_q90"] + wf["settlement_period"].map(sp_delta)
    inside_cal = (wf["ssp_actual"] >= wf["q10_cal"]) & (wf["ssp_actual"] <= wf["q90_cal"])
    coverage_after = inside_cal.mean()

    # Implied coverage with global δ (diagnostic)
    wf["q10_gbl"] = wf["ssp_q10"] - delta_global
    wf["q90_gbl"] = wf["ssp_q90"] + delta_global
    inside_gbl = (wf["ssp_actual"] >= wf["q10_gbl"]) & (wf["ssp_actual"] <= wf["q90_gbl"])
    coverage_global = inside_gbl.mean()

    n_days = wf["settlement_date"].nunique() if "settlement_date" in wf.columns else None
    n_rows  = len(wf)

    return {
        "version":              "v1",
        "target_coverage":      TARGET_COVERAGE,
        "quantile_level":       QUANTILE_LEVEL,
        "method":               "split_conformal_symmetric",
        "n_calibration_days":   int(n_days) if n_days is not None else None,
        "n_calibration_rows":   int(n_rows),
        "folds_used":           sorted(wf["fold"].unique().tolist()) if "fold" in wf.columns else None,
        "coverage_before":      round(float(coverage_before), 4),
        "coverage_after_insample_per_sp": round(float(coverage_after), 4),
        "coverage_after_insample_global": round(float(coverage_global), 4),
        "delta_global":         round(float(delta_global), 4),
        "delta_by_sp":          {str(sp): round(float(d), 4) for sp, d in sp_delta.items()},
        "delta_stats": {
            "min":    round(float(sp_delta.min()), 2),
            "p25":    round(float(sp_delta.quantile(0.25)), 2),
            "median": round(float(sp_delta.median()), 2),
            "p75":    round(float(sp_delta.quantile(0.75)), 2),
            "max":    round(float(sp_delta.max()), 2),
        },
        "created": str(date.today()),
    }


def main() -> None:
    if not WF_PATH.exists():
        print(f"ERROR: {WF_PATH} not found", file=sys.stderr)
        sys.exit(1)

    wf = pd.read_csv(WF_PATH)
    result = compute_pi_calibration(wf)

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved → {OUT_PATH}")
    print(f"  Folds used          : {result['folds_used']}")
    print(f"  Calibration rows    : {result['n_calibration_rows']}  "
          f"({result['n_calibration_days']} days)")
    print(f"  Coverage before     : {result['coverage_before']:.1%}")
    print(f"  Coverage after (SP) : {result['coverage_after_insample_per_sp']:.1%}  "
          f"(target {TARGET_COVERAGE:.0%})")
    print(f"  Coverage after (gbl): {result['coverage_after_insample_global']:.1%}")
    print(f"  δ global (p80)      : £{result['delta_global']:.2f}")
    print(f"  δ per-SP range      : £{result['delta_stats']['min']:.1f} – "
          f"£{result['delta_stats']['max']:.1f}  "
          f"(median £{result['delta_stats']['median']:.1f})")


if __name__ == "__main__":
    main()

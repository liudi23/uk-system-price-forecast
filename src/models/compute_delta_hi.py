"""
Compute δ_hi for the high-risk afternoon SP block (33–40), targeting the
upper tail of elevated-price exceedances that the calibrated PI band misses.

Algorithm
---------
1. q90_cal(sp) = ssp_q90_raw(sp) + delta_sp(sp)   [post-PI-calibration q90]
2. Exceedance score: score_hi = max(actual - q90_cal, 0)
   computed on rows where actual > ELEVATED_THRESHOLD (£120) only.
   (≈ 76% of elevated rows already within calibrated band → score_hi = 0.)
3. Identify HIGH-RISK SPs: afternoon block {33..40} ∩ spike_propensity ≥ 2%.
4. Pool exceedance rows (score_hi > 0) from the high-risk SP block.
5. δ_hi = p80 of pooled exceedance scores — single value applied uniformly
   to all high-risk SPs.
   Applied uniformly because per-SP has only 1–6 exceedance events: insufficient
   for a reliable 80th-percentile estimate.
6. All other SPs receive δ_hi_eff = 0 (no widening).

Why exceedances only, not all elevated rows?
   p80 of ALL elevated-row scores is ≈ £2.88 (trivial) because PI calibration
   already covers 76.4% of elevated rows. Using exceedance rows gives a
   δ_hi that covers 80% of the "hard" events PI calibration fails on.

Design notes
------------
- δ_hi measured against POST-PI-calibration q90. At inference: apply AFTER PI
  calibration and BEFORE Kalman. Q10 and Q50 unchanged (asymmetric widening).
- Code-P NOT used: 46% of SPs at normal prices; conformity dominated by non-spikes.
- Season split deferred: ~27 afternoon-block exceedance rows in WF window.
- SP 14, 15 excluded despite propensity ≥ 2% because they are morning SPs outside
  the afternoon peak mechanism documented in docs/spike-tail-design.md.

Usage
-----
    python src/models/compute_delta_hi.py
"""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parents[2]
WF_PATH  = ROOT / "model_assets" / "walk_forward_predictions.csv"
PI_PATH  = ROOT / "model_assets" / "pi_calibration_v1.json"
OUT_PATH = ROOT / "model_assets" / "delta_hi_v1.json"

ELEVATED_THRESHOLD   = 120.0          # £/MWh — filter for elevated-price rows
SPIKE_THRESHOLD      = 150.0          # £/MWh — propensity = P(actual > this)
PROPENSITY_THRESHOLD = 0.02           # min P(actual>150) for SP to be high-risk
AFTERNOON_SP_RANGE   = range(33, 41)  # SP 33–40: afternoon/evening peak block
QUANTILE_LEVEL       = 0.80           # p80 of pooled exceedance scores
MIN_EXCEEDANCES      = 10             # require ≥ this many exceedance rows for estimate


def main() -> None:
    if not WF_PATH.exists():
        print(f"ERROR: {WF_PATH} not found", file=sys.stderr)
        sys.exit(1)

    # ── Load WF predictions and PI calibration ────────────────────────────────
    wf = pd.read_csv(WF_PATH)
    wf["settlement_period"] = wf["settlement_period"].astype(int)
    wf["ssp_actual"] = pd.to_numeric(wf["ssp_actual"], errors="coerce")
    wf["ssp_q90"]    = pd.to_numeric(wf["ssp_q90"],    errors="coerce")
    wf = wf.dropna(subset=["ssp_actual", "ssp_q90"])

    pi = json.load(open(PI_PATH)) if PI_PATH.exists() else {}
    delta_sp_map = {int(k): float(v) for k, v in pi.get("delta_by_sp", {}).items()}
    if not delta_sp_map:
        print("WARNING: pi_calibration_v1.json missing — q90_cal = q90_raw", file=sys.stderr)

    # ── Calibrated q90 per row ────────────────────────────────────────────────
    wf["q90_cal"] = wf["ssp_q90"] + wf["settlement_period"].map(delta_sp_map).fillna(0.0)

    # ── Spike propensity per SP ──────────────────────────────────────────────
    n_wf_days = wf["settlement_date"].nunique()
    spike_prop = (
        wf.groupby("settlement_period")["ssp_actual"]
        .apply(lambda x: (x > SPIKE_THRESHOLD).mean())
    )

    # ── High-risk SP set: afternoon block ∩ propensity filter ────────────────
    afternoon_set  = set(AFTERNOON_SP_RANGE)
    propensity_set = {sp for sp in spike_prop.index if spike_prop[sp] >= PROPENSITY_THRESHOLD}
    high_risk_sps  = sorted(afternoon_set & propensity_set)
    print(f"Afternoon block (SP 33–40) ∩ propensity ≥ {PROPENSITY_THRESHOLD:.0%}: {high_risk_sps}")

    for sp in high_risk_sps:
        print(f"  SP {sp:2d}: propensity = {spike_prop[sp]:.1%}")

    # ── Elevated-price exceedance rows from high-risk block ───────────────────
    elevated = wf[(wf["ssp_actual"] > ELEVATED_THRESHOLD)].copy()
    elevated["score_hi"] = (elevated["ssp_actual"] - elevated["q90_cal"]).clip(lower=0.0)

    high_risk_elevated  = elevated[elevated["settlement_period"].isin(high_risk_sps)]
    exceedances         = high_risk_elevated[high_risk_elevated["score_hi"] > 0]

    n_elevated_block  = len(high_risk_elevated)
    n_exceedances     = len(exceedances)
    print(f"\nHigh-risk block elevated rows (actual > £{ELEVATED_THRESHOLD:.0f}): {n_elevated_block}")
    print(f"  Exceedances (score_hi > 0): {n_exceedances} ({n_exceedances/max(n_elevated_block,1):.1%})")

    # ── Compute δ_hi for the block ────────────────────────────────────────────
    if n_exceedances >= MIN_EXCEEDANCES:
        delta_hi_block = float(exceedances["score_hi"].quantile(QUANTILE_LEVEL))
        print(f"\nδ_hi (p{int(QUANTILE_LEVEL*100)} of {n_exceedances} exceedance rows): £{delta_hi_block:.2f}")
        for pct in [0.50, 0.60, 0.70, 0.80, 0.90]:
            print(f"  p{int(pct*100)}: £{exceedances['score_hi'].quantile(pct):.2f}")
    else:
        delta_hi_block = 0.0
        print(f"\nWARNING: only {n_exceedances} exceedance rows (< {MIN_EXCEEDANCES}) — δ_hi = 0 (deferred)",
              file=sys.stderr)

    # Per-SP breakdown (diagnostic only — NOT used for widening due to data sparsity)
    per_sp_base = {}
    per_sp_nexceed = {}
    for sp in range(1, 49):
        sub = elevated[elevated["settlement_period"] == sp]
        exc = sub[sub["score_hi"] > 0]
        per_sp_base[sp] = float(sub["score_hi"].quantile(QUANTILE_LEVEL)) if len(sub) >= 5 else float("nan")
        per_sp_nexceed[sp] = len(exc)

    # ── Global diagnostics ────────────────────────────────────────────────────
    all_exceedances  = elevated[elevated["score_hi"] > 0]
    delta_hi_global  = float(all_exceedances["score_hi"].quantile(QUANTILE_LEVEL)) if len(all_exceedances) >= 5 else 0.0
    print(f"\nGlobal exceedance δ_hi (all SPs, informational): £{delta_hi_global:.2f}")

    # ── Apply block delta uniformly to high-risk SPs ──────────────────────────
    all_sps = list(range(1, 49))
    delta_hi_eff = {sp: (delta_hi_block if sp in high_risk_sps else 0.0) for sp in all_sps}
    n_widened = sum(1 for v in delta_hi_eff.values() if v > 0)
    print(f"\nSPs receiving widening (δ_hi = £{delta_hi_block:.2f}): {high_risk_sps}  ({n_widened}/48)")

    # ── Save artifact ─────────────────────────────────────────────────────────
    artifact = {
        "version":              "v1",
        "source":               "elevated_price_conformity_afternoon_block",
        "elevated_threshold":   ELEVATED_THRESHOLD,
        "spike_threshold":      SPIKE_THRESHOLD,
        "propensity_threshold": PROPENSITY_THRESHOLD,
        "afternoon_sp_range":   [33, 40],
        "quantile_level":       QUANTILE_LEVEL,
        "min_exceedances":      MIN_EXCEEDANCES,
        "n_wf_days":            int(n_wf_days),
        "n_elevated_rows":      int(len(elevated)),
        "n_elevated_block":     int(n_elevated_block),
        "n_exceedances_block":  int(n_exceedances),
        "high_risk_sps":        high_risk_sps,
        "delta_hi_block":       round(float(delta_hi_block), 4),
        "delta_hi_global":      round(float(delta_hi_global), 4),
        "delta_hi_effective_by_sp": {str(sp): round(v, 4) for sp, v in delta_hi_eff.items()},
        "spike_propensity_by_sp": {
            str(sp): round(float(spike_prop.get(sp, 0.0)), 4) for sp in all_sps
        },
        "per_sp_n_exceedances": {str(sp): per_sp_nexceed.get(sp, 0) for sp in all_sps},
        "per_sp_delta_hi_base_diagnostic": {
            str(sp): round(v, 4) if not (isinstance(v, float) and np.isnan(v)) else None
            for sp, v in per_sp_base.items()
        },
        "delta_hi_stats": {
            "n_widened_sps":  n_widened,
            "delta_hi_value": round(float(delta_hi_block), 2),
        },
        "created": str(date.today()),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()

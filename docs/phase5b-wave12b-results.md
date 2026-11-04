# Phase 5b — wave 1.2b results: NOT ADOPTED

**Verdict: KILL.** Gate failed on both primary endpoints. Branch `phase-5-wave1.2b`
is not merged; production (`streamlit-data`) is unchanged.

Validated: 2026-11-04 · 148 days · 5 seasonal folds incl. autumn-2026 ·
A = production set (wave 1 on), B = A + six within-day relative margin/solar features.
Pre-registration: `docs/phase5b-wave12b-prereg.md` (written before these numbers).

## Gate result

| | ΔMAE | ΔPinball | Coverage | Gate needs |
|---|---|---|---|---|
| ALL | **+1.0 %** | **+2.0 %** | 0.431 → 0.426 (−0.005, OK) | ≤ −3 % or ≤ −5 % |

No fold improved more than −0.6 % MAE, so the regime-conditional clause
(≥ −5 % in targeted folds) does not apply either. Level columns were bit-identical
A/B — clean isolation; this is a pure shape-head read.

## Secondary (diagnostic) endpoints vs pre-registered expectations

1. **Shape corr 0.45 → low 0.50s**: NO. Slipped 0.437 → 0.431 overall.
2. **Pit median 14 → single digits**: NO overall movement (6 → 6), and the targeted
   fold, autumn-2026, stayed at 14 — the defect the features were built for is untouched.
3. **Peak median ≤ 3 SPs**: held (4 → 3 overall; spring-2026 remains pathological at ~18
   for both arms — a different problem, not timing but a level/shape regime issue).
4. The one real effect: **pit precision** — Pit ≤ 2 SP rose 28 % → 35 % overall
   (winter 33 → 50 %, spring 40 → 60 %). The solar mechanism exists but is weak, and it
   came bundled with +2 % pinball and +3.1 % winter MAE.

## Interpretation (matches a pre-named failure mode)

The within-day margin/solar information is already largely absorbed through the wave-1
per-SP vintage levels (`margin_fc_mw`, `tsdf_fc_mw`, `windfor_fc_mw`, solar_wm2_lag_48
with forecast substitution); the derived relatives (z, rank, argmax distances) mostly
added variance. The in-sample signal was strong (margin_fc_z ranked #1 by permutation
importance in the smoke fit) — the walk-forward shows that was overfitting, which is
exactly what the gate is for.

## Disposition

- Branch kept unmerged as the experimental record (prereg + implementation + this verdict).
- The `--no-wave12b` flag and harness upgrades (pit-timing metric in
  `decomposition_metrics`, shape-diagnostics table in the walk-forward comparison) are
  worth cherry-picking into future gates — the diagnostics table is model-agnostic.
- Remaining lead for the autumn-2026 pit problem: pits there may be driven by actual
  solar *generation* (embedded PV estimates) rather than irradiance shape, or by
  negative-price bidding behaviour — a candidate for the parked Phase 6a spike/negative
  classifier work, not for more shape features.

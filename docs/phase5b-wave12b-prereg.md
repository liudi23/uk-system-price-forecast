# Phase 5b — wave 1.2b pre-registration

**Status: PRE-REGISTERED (written before any validation numbers were seen)**
Date: 2026-11-03 · Branch: `phase-5-wave1.2b` · Baseline: `streamlit-data` @ 09c0496

## Diagnosis (established empirically on 669 days, 2025–26, before this experiment)

- The D−1 residual-demand (margin = TSDF − WINDFOR) profile alone explains a mean
  within-day correlation of **r ≈ 0.42** with the realised 48-SP price shape; the full
  HGBR shape head currently reaches **r ≈ 0.45**.
- **Peak timing is already trackable**: median |price-argmax − margin-argmax| = 3 SPs
  (≤ 2 SPs on 44 % of days).
- **Pit timing fails**: median |price-argmin − margin-argmin| = 14 SPs. Pits are midday
  solar-duck events, and solar is absent from margin (TSDF − WINDFOR nets off wind only).
- Raw MW levels are non-stationary under capacity drift (this is what killed the wave-1
  *level* features). Within-day **relative** position is regime-free.

## Intervention (H+1 shape head only)

New features in `build_features.py` (gated by `--no-wave12b` for the A build):

| Feature | Source | Meaning |
|---|---|---|
| `margin_fc_z` | D−1 vintage | margin's z-score within day D's 48-SP profile |
| `margin_fc_rank` | D−1 vintage | margin's within-day percentile rank |
| `margin_fc_dist_max` | D−1 vintage | signed SP distance to the day's margin maximum |
| `margin_fc_dist_min` | D−1 vintage | signed SP distance to the day's margin minimum |
| `solar_wm2_lag_48_rel` | lag-48 irradiance | solar height relative to the day's max (0–1) |
| `solar_wm2_lag_48_distmax` | lag-48 irradiance | signed SP distance to the day's solar maximum |

Leakage: `margin_fc_*` derive from the D−1 12:00Z vintage (legitimately known day-ahead
for H+1; the existing `"_fc"` pattern excludes them from H+2). `solar_wm2_lag_48_*` use
yesterday's observed irradiance at training and the day-ahead Open-Meteo solar forecast
at inference — the same substitution convention already used for `solar_wm2_lag_48`; the
`"_lag_48"` pattern excludes them from H+2. The level model's feature list is explicit
(`get_level_feature_cols`), so nothing new reaches it. All six names fall under existing
`OPTIONAL_FEATURE_PREFIXES`, so they NaN-degrade rather than dropping rows.

## A/B definition

- **A (baseline)** = current production feature set: full-history level window + wave-1
  MID/vintage shape features (vintages **not** hidden — this gate is incremental on 5b wave 1).
- **B (candidate)** = A + the six wave-1.2b features.
Both runs: `--walk-forward --level-train-years 0`, 5 seasonal folds incl. autumn-2026.

## Pre-registered expectations and gate

Primary endpoints (unchanged adoption gate): **ΔMAE ≤ −3 % or ΔPinball ≤ −5 % overall,
with no P10–P90 coverage drop > 0.02**. Regime-conditional clause available as
pre-registered in `phase5-upgrade-plan.md` (named diagnosis → targeted folds ≥ −5 %,
no other fold > +2 %).

Secondary (diagnostic, not sufficient for adoption on their own):

1. Within-day shape correlation rises from ≈ 0.45 toward the low 0.50s.
2. **Pit-timing median error improves materially** (the targeted defect; 14 → single digits
   would confirm the solar mechanism).
3. Peak-timing median stays ≤ 3 SPs (no regression).
4. Overall MAE impact is expected to be **small** (shape error is the minor term vs level
   error), so the likely honest outcomes are: (a) pass on pinball, (b) fail the primary
   gate but clearly improve the timing diagnostics → park or iterate, (c) no effect → kill.

Failure modes named in advance: if B improves pit timing but worsens shape correlation or
coverage, the features are overfitting the argmin and should not be adopted; if nothing
moves, the margin/solar information is already absorbed via the wave-1 per-SP vintage
levels and the relatives add nothing.

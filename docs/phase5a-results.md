# Phase 5a Results — Error Autopsy, Window/Weighting Sweep, Target Transform

*Executed 2026-11-02 (Claude session). Branch: `phase-5`. All experiments offline; nothing deployed.*

## 1. Error autopsy — what actually degraded

As-published H+1 forecasts (dated archive; prod where available, early/shadow otherwise; 166 days, May 18 – Oct 31 2026) scored against settled actuals:

| Period | n | P50 MAE £ | abs level err £/day | shape corr | 80% cov | actual mean £ |
|---|---|---|---|---|---|---|
| May–Jun | 43 | 32.4 | 15.8 | 0.40 | 0.68 | 99 |
| Jul–Aug | 62 | 38.1 | 20.3 | 0.40 | 0.59 | 119 |
| Sep | 30 | 59.0 | 30.4 | 0.40 | 0.50 | 129 |
| Oct 1–15 | 15 | 55.2 | 35.4 | 0.45 | 0.55 | 138 |
| Oct 16–31 | 16 | 66.9 | 47.2 | 0.43 | 0.43 | 168 |

**Findings.** The degradation is entirely **level-driven**: absolute level error tripled (£16 → £47/day) while shape correlation stayed flat (~0.40–0.45). The level error is a systematic **under-prediction** (October mean signed error **+£35/day** — actuals above forecast) that grew as the market climbed from ~£100 to ~£170/MWh. Band coverage collapsed (0.68 → 0.43) as a direct consequence: the bands are centred too low, not merely too narrow. Seven of the eight worst days are high-price days under-forecast at the level stage. Every October day exceeded £150, so the fixed "spike day" threshold no longer discriminates in this regime.

## 2. Window / weighting / transform sweep (level model)

Walk-forward harness: 13 folds, fold start every 14 days (2026-05-01 → 2026-10-19), 182 evaluation days total. Per fold, P10/P50/P90 level HGBR retrained on data strictly before the fold (2-day gap), window / weights / transform per config. Metrics: daily-level q50 MAE, mean pinball(q10, q90), raw 80% band coverage (pre-conformal).

**Caveat:** run on the reduced local feature set (56 features — no weather, generation-mix or CPI columns, hence no CPI deflation), so absolute numbers are not comparable to production. The *relative ranking* is the result.

| Window | Weighting | Transform | Level MAE £ | Bias £ | Pinball £ | 80% cov |
|---|---|---|---|---|---|---|
| all (5.5y) | uniform | linear | 20.78 | -0.07 | 5.50 | 0.60 |
| all (5.5y) | uniform | asinh | 21.28 | -1.92 | 5.46 | 0.64 |
| all (5.5y) | hl 365d | linear | 22.18 | -0.77 | 5.86 | 0.57 |
| all (5.5y) | hl 182d | linear | 23.20 | -3.57 | 6.37 | 0.56 |
| all (5.5y) | hl 182d | asinh | 23.33 | -4.91 | 6.38 | 0.54 |
| 24 mo | hl 365d | linear | 24.29 | -8.76 | 6.61 | 0.56 |
| 36 mo | hl 365d | linear | 24.30 | -8.71 | 6.57 | 0.54 |
| 36 mo | uniform | linear | 24.73 | -9.48 | 6.92 | 0.51 |
| 24 mo | hl 182d | linear | 24.75 | -7.97 | 6.67 | 0.56 |
| 12 mo | hl 91d | linear | 24.75 | -9.02 | 7.08 | 0.50 |
| 24 mo | uniform | linear | 24.77 | -9.75 | 6.47 | 0.58 |
| 36 mo | hl 182d | linear | 25.06 | -9.29 | 6.77 | 0.55 |
| 24 mo | hl 91d | linear | 25.20 | -6.09 | 7.20 | 0.46 |
| 36 mo | hl 182d | asinh | 25.26 | -9.97 | 7.02 | 0.51 |
| 36 mo | uniform | asinh | 25.28 | -10.84 | 7.39 | 0.45 |
| 12 mo | hl 365d | linear | 25.29 | -12.54 | 6.81 | 0.52 |
| 12 mo | hl 182d | linear | 25.33 | -10.85 | 6.84 | 0.54 |
| all (5.5y) | hl 91d | linear | 25.45 | -8.33 | 6.76 | 0.55 |
| 18 mo | hl 182d | linear | 25.51 | -9.97 | 6.92 | 0.55 |
| 18 mo | hl 365d | linear | 25.65 | -11.14 | 6.85 | 0.56 |
| 18 mo | uniform | linear | 25.73 | -11.26 | 6.75 | 0.54 |
| 18 mo | hl 182d | asinh | 26.04 | -11.25 | 6.68 | 0.52 |
| 36 mo | hl 91d | linear | 26.07 | -10.48 | 7.05 | 0.54 |
| 18 mo | hl 91d | linear | 26.22 | -8.71 | 7.29 | 0.56 |
| 12 mo | uniform | linear | 26.33 | -14.60 | 7.14 | 0.46 |
| 18 mo | uniform | asinh | 26.75 | -13.80 | 7.01 | 0.51 |


**Findings.**

1. **The 3-year training truncation is the root cause of the October bias.** Every truncated window (12–36 mo) under-predicts by £6–15/day. The full 5.5-year history — which includes the 2021–23 high-price era — is essentially **unbiased (−£0.07)** and beats the best truncated config by **−14% MAE** (£20.78 vs £24.29). The model had progressively forgotten what high-price regimes look like, precisely as prices returned to them.
2. **Recency weighting hurts once full history is available** (hl 365d: +£1.4 MAE; hl 91d: +£4.7). The trees need the old high-price support at full weight.
3. **asinh target transform** is MAE-neutral-to-slightly-worse (£21.28 vs £20.78 on the winning window) but gives the **best raw coverage (0.64 vs 0.60)** and comparable pinball. Optional; decide at the CI gate alongside re-fitted conformal δ.

## 3. Recommendations

- **Adopt (pending CI gate): full-history training for the level model** — `TRAIN_YEARS` 3 → unbounded, uniform weights. Validate on the full 119-day CI walk-forward with the complete feature set and CPI deflator before merging to `streamlit-data`. (Deflation makes old nominal prices comparable, so full history should hold or improve there; "inflation drift", the original rationale for truncation, is already handled by the deflator.)
- **Hold: asinh** as a candidate paired with re-fitted PI calibration; adopt only if the gate shows the coverage gain without MAE regression.
- **Reject for the level model: short windows and recency weighting.**
- Shape model untouched this phase (autopsy shows it stable); a separate cheap window test for shape is optional follow-up.
- Re-base the £150 spike threshold (and the Phase 6a spike-classifier plan) to a regime-relative definition, e.g. a rolling percentile — October made a fixed £150 meaningless.

## 4. Reproduction

- Autopsy inputs: `model_assets/forecasts/*` vs `data/processed/dataset_5yr.csv` (per-day results: `reports/phase5a/autopsy_daily.csv`).
- Sweep: 26 configs × 13 folds (`reports/phase5a/sweep_5a.py`, results `reports/phase5a/sweep_results.csv`).
- Adoption gate: walk-forward MAE ≥3% better with no PI-coverage regression, per `docs/phase5-upgrade-plan.md`.

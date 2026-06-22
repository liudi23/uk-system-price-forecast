# UK System Sell Price (SSP) Forecasting — One-Page Summary

*Companion to [docs/technical-report.md](technical-report.md). All figures sourced from that report.*

**Objective.** Probabilistic day-ahead + intraday forecast of the UK System Sell Price across all 48 half-hourly settlement periods (SPs), with calibrated 80% prediction intervals (PIs).

**Headline:** WF MAE **£28.96**/MWh (StaticBase) · **£27.68** (Kalman) · 80% PI coverage **37.99% → 79.82%** · 7-day MAE **£32.03** · shape corr **0.4275**

## Model — two-stage level–shape HGBR
- **Stage 1 (level):** daily price level as quantiles P10/P50/P90 — pinball-loss `HistGradientBoostingRegressor`, 84 features, CPIH-deflated targets.
- **Stage 2 (shape):** per-SP deviation from the daily level (separate H+1 and H+2 heads). Day-ahead SP forecast = `q50_level + q50_shape`; bands from level quantiles + per-SP conformal widening.
- **Training:** weekly age-gated retrain on a rolling 3-year, CPIH-adjusted window; strict zero-leakage — every feature references data before day D begins, valid for all 48 day-ahead SPs.

## Intraday correction — every 30 min
Inference-only (frozen HGBR): fetch newly-settled SPs from Elexon BMRS → splice actuals into the live curve → scalar Kalman level-bias `x̂` (random-walk prior). Correction decays with horizon: `correction(h) = x̂·γ^h`, **γ = 0.966**/SP (halves by ≈SP+20). Uncertainty propagates to the P10–P90 bands via `±1.28·√P·γ^h` — widening when few SPs have settled, shrinking as posterior variance `P` falls. Kalman adds **+4.5 pp** PI coverage over the static base.

## Calibrated uncertainty — split-conformal per-SP δ
Per settlement position, `δ(sp)` = 80th-percentile conformity score over the calibration set; bands widened by `δ(sp)`. Lifts 80%-band coverage **37.99% → 79.82%** in-sample over the 119-day, 4-fold walk-forward (5,709 SP-rows). This was a calibration gain, not a better point model. *(Live DA coverage 66.0% over 30 spring/summer days, N=1,418 — under active monitoring.)*

## Short-horizon nowcast — persistence
SSP is near-AR(1) (ACF lag-1 = 0.828), so persistence (`SP[t] = SP[t−1]`) is the Bayes-optimal h+1 predictor. DA+Kalman adds no value through ≈h+4 and only overtakes persistence at ≈h+4.5 (≈5–6% better by h+5); an HGBR nowcast prototype was 17% worse than persistence at h+1 and was not shipped. Production serves persistence for h+1–h+3 with regime-asymmetric empirical 80% bands (live coverage 79.4–79.8%).

## Results

| Metric | Value |
|---|---|
| WF MAE — StaticBase (119 days, 4 seasonal folds) | £28.96/MWh |
| WF MAE — deployed Kalman corrector | £27.68/MWh |
| 7-day holdout MAE (RMSE £42.46, n=336) | £32.03/MWh |
| Stage-1 level MAE | £18.41/day |
| Stage-2 shape correlation (mean) | 0.4275 |
| Peak-timing MAE | 6.71 SPs |
| 80% PI coverage (in-sample WF) | 37.99% → 79.82% |

Seasonal WF MAE (Kalman): Summer £25.77 · Autumn £30.50 · Winter £20.27 · Spring £34.28. Baselines: naive t−48 £36.34 · seasonal naive t−336 £29.40 · rolling-mean-24h £26.78.

## Status
Day-ahead 48-SP profile published early (~01:00 UTC) to a live Streamlit dashboard; intraday Kalman layer updates every 30 min. CI-tested (pytest on push). The early-publish pipeline is validating against the prior 12:30 UTC daily pipeline before full cut-over. **Live dashboard:** <https://uk-system-price-forecast.streamlit.app/>

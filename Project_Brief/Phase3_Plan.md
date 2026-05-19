# UK Electricity System Price Forecasting — Phase 3 Plan

**Status:** Planning
**Follows:** Phase 2 (Quantile HGBR + Spike Classifier · May 2026)

---

## Phase 2 recap

Phase 2 upgraded the forecasting architecture with:

- Quantile HGBR — P10/P50/P90 uncertainty bands replacing the single point forecast
- Binary spike classifier (`class_weight="balanced"`) for per-period spike risk
- Training on raw (un-winsorised) prices (`ssp_raw`) to remove the artificial £354 prediction ceiling
- 91 features including spike memory (`ssp_raw_lag_{48,96,336}`, `is_spike_lag_{48,336}`) and NIV stress indicators
- `evaluate_dayahead_recursive()` — honest evaluation matching the deployment loop exactly

**Key finding:** the honest recursive P50 MAE is **£25.40/MWh**, approximately £11 higher than the leaky batch metric from Phase 1. The gap arises because `ssp_lag_1` — the dominant feature (importance = 22.0) — carries actual within-day prices in batch evaluation but must be approximated by the model's own previous prediction at day-ahead time.

**Root cause of remaining error:** recursive error propagation. Each settlement period's prediction depends on the previous period's prediction. Errors introduced early in the day compound across all 48 steps. The model learns the intra-day *shape* well via `sin_sp`/`cos_sp` and `ssp_lag_48`, but the *level* drifts as the recursion deepens.

---

## Phase 3 core insight

The daily SSP time series can be decomposed into two independent problems:

1. **Level** — what is the day's average price? (a single number per day)
2. **Shape** — how does the price deviate from that average across 48 settlement periods?

Both can be estimated using only lag-48+ features — data that is unambiguously available at day-ahead dispatch time. This eliminates recursive error propagation entirely.

---

## Phase 3 objectives

### 1. Level-shape decomposition model (high priority)

**Problem:** Recursive inference accumulates prediction errors across 48 steps. A drift of £10 in SP5 contaminates every subsequent prediction that day.

**Architecture:**

**Stage 1 — Daily level model**
- Target: mean SSP across all 48 SPs of the forecast day
- Features: lag-48+ only (rolling daily averages, weekly averages, NIV history, calendar, daily-average weather forecast)
- Model: quantile HGBR for P10/P50/P90 of the daily level
- Output: one number per day — the predicted average price and its uncertainty band
- No recursion; no error propagation

**Stage 2 — Intra-day shape model**
- Target: `ssp_sp_h − daily_mean` for each settlement period h = 1…48
- Features: `sin_sp`/`cos_sp`, predicted daily level, hourly weather forecast (solar irradiance, wind), day-type profile (weekday/weekend/bank holiday), `ssp_lag_48` for the same SP
- Model: either a single model taking SP index as a feature, or 48 independent direct models (one per SP)
- Output: deviation from the predicted level for each SP

**Final forecast:** `predicted_level + predicted_deviation_sp_h`

**Acceptance criteria:**
- Daily level MAE reported as a standalone metric
- Shape correlation (predicted vs actual intra-day profile shape, independent of level) reported separately
- Total MAE decomposed into level error and shape error
- End-to-end P50 MAE improves on Phase 2's £25.40/MWh benchmark

---

### 2. Direct multi-step forecasting (DIRECT strategy)

**Problem:** Even with decomposition, there may be per-SP structure not captured by a shared shape model.

**Proposed approach:**
- Train 48 independent HGBR models — one per settlement period
- Each model predicts `ssp_sp_h` directly from lag-48+ features, with no dependency on other periods' predictions
- Compare against the two-stage decomposition on the same test week

**Trade-offs:**

| Approach | Pros | Cons |
|---|---|---|
| Recursive (Phase 2) | One model, simple | Error propagation across 48 steps |
| Level + shape (Phase 3) | Separable diagnostics, no propagation | Requires two training stages |
| DIRECT (48 models) | Maximum flexibility per SP | 48× training cost, harder to maintain |

Both level-shape and DIRECT will be evaluated; the better approach becomes the production model.

---

### 3. Exogenous day-ahead inputs

**Problem:** The current model relies entirely on lagged price/NIV history and weather. Day-ahead generation mix and demand forecasts are publicly available and directly price-relevant.

**Proposed additions:**

- **Day-ahead wind generation forecast** — NESO publishes day-ahead wind output forecasts; high wind → low marginal cost → lower prices
- **Day-ahead solar generation forecast** — Open-Meteo day-ahead GHI already integrated; aggregate to daily/SP-level expected solar output
- **Day-ahead demand forecast** — NESO National Demand Forecast (NDF); higher demand → higher prices, especially in winter mornings/evenings
- **Day-ahead gas price** — TTF or NBP day-ahead; dominant marginal cost driver for gas-fired CCGT plants that set the price in most periods

Each input is available before day-ahead dispatch (typically by 10:00 the preceding day), so no leakage is introduced.

---

### 4. Calibration and conformal prediction intervals

**Problem:** Phase 2's P10/P90 quantile bands are nominal — there is no guarantee they achieve 80% empirical coverage on out-of-sample data. Miscalibrated intervals mislead risk decisions.

**Proposed approach:**
- Implement split-conformal prediction on top of the quantile models
- Use a calibration holdout (e.g. the most recent 30 days before the test week) to compute coverage-correcting residual quantiles
- Report empirical coverage at P10/P90 on the test set; target ≥ 80% actual coverage
- Dashboard addition: calibration plot (predicted quantile vs empirical hit rate) in the model accuracy panel

---

### 5. Error decomposition dashboard panel

**Problem:** The current model accuracy panel shows aggregate MAE, RMSE, sMAPE. It is not possible to tell whether errors come from getting the day's level wrong or getting the intra-day shape wrong.

**Proposed additions:**
- **Level error** — daily mean predicted vs actual; how often the model misjudges the day's average price level
- **Shape correlation** — per-day Pearson correlation between predicted and actual 48-SP profile; catches days where the level is right but the timing of peaks is wrong
- **Peak timing error** — how many settlement periods off is the predicted daily peak vs actual daily peak
- **Cumulative error by SP** — average abs error by settlement period index across the test week; shows whether error concentrates in morning ramp, evening peak, or overnight periods

---

### 6. Automated retraining pipeline (carry forward from Phase 2 plan)

**Problem:** The model is a static artefact trained once on May 2021–2026 data. Accuracy drifts as market conditions evolve.

**Proposed approach:**
- Automate the full pipeline: `fetch_elexon → fetch_weather → build_dataset → build_features → train → forecast`
- Weekly retraining on a rolling 5-year window via GitHub Actions or cron
- Version model artefacts by training date in `model_assets/`
- Track weekly test MAE over time in the verification panel to surface drift

---

## Suggested Phase 3 delivery order

| Priority | Item | Effort | Impact |
|---|---|---|---|
| 1 | Level-shape decomposition model | High | High — removes recursive error propagation |
| 2 | DIRECT 48-model comparison | Medium | High — establishes best architecture |
| 3 | Exogenous day-ahead inputs | Medium | High — wind/demand most impactful |
| 4 | Error decomposition dashboard panel | Low | Medium — diagnostic value |
| 5 | Conformal prediction calibration | Medium | Medium — honest uncertainty |
| 6 | Automated retraining pipeline | Medium | High — production readiness |

---

## Evaluation framework

All Phase 3 models are evaluated on the same test week (May 11–17 2026) using `evaluate_dayahead_recursive()` or its direct-model equivalent. Additional metrics introduced in Phase 3:

| Metric | Description |
|---|---|
| Daily level MAE | Mean absolute error of predicted vs actual daily average SSP |
| Shape correlation | Mean Pearson r between predicted and actual 48-SP intra-day profile |
| Peak timing error | Mean absolute SP offset between predicted and actual daily peak |
| Empirical P10/P90 coverage | Fraction of test periods where actual falls within predicted interval |
| Error by SP index | Average abs error broken down by settlement period (1–48) |

---

## Open questions

1. Does the level-shape decomposition outperform DIRECT 48 models? Theory favours DIRECT for flexibility, but in practice 48 models may overfit the training set individually.
2. How much lift does each exogenous input add in isolation? (wind vs demand vs gas price)
3. Is the daily average SSP predictable to better than the rolling-7-day-mean naive baseline — i.e., is there genuine signal in the level model beyond recent history?
4. What calibration holdout size is needed for reliable conformal intervals given the non-stationarity of electricity prices?
5. Deployment target: local pipeline or cloud-scheduled (GitHub Actions / AWS Lambda)?

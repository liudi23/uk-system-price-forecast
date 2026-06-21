# Hourly Forecast Calibration Design

**Branch:** `streamlit-data`  
**Status:** Design only — no implementation yet  
**Date:** 2026-06-16

---

## 1. Pipeline Audit

### 1.1 Intraday SSP fetch

**Module:** `src/data/fetch_intraday.py`

**Schedule:** Hourly at :30 past, 08:30–17:30 UTC, via `intraday_update.yml` on the `main` branch
(which checks out `streamlit-data` to run). The cron is `'30 8-17 * * *'`.

**Mechanism:** Calls `fetch_day(today, session)` imported from `src/data/fetch_elexon.py`, which
hits the Elexon Insights Solution REST API:

```
GET https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{settlement_date}
```

No authentication required.

**Data shape:** One row per settlement period (SP). Each row:

| Column | Type | Description |
|---|---|---|
| `settlement_date` | str YYYY-MM-DD | Calendar date |
| `settlement_period` | int 1–48 | 30-min slot (SP 1 = 00:00–00:30) |
| `ssp` | float £/MWh | System Sell Price |
| `net_imbalance_volume` | float MWh | NIV |
| `sell_price_adjustment` | float | Sell price adjustment |
| `buy_price_adjustment` | float | Buy price adjustment |
| `price_derivation_code` | str P/N/K | How the price was set |
| `replacement_price` | float | Replacement reference |

Note: SBP (`systemBuyPrice`) is **not** in the current `COLUMN_MAP` and is not fetched. The model
forecasts SSP only.

**Publication delay (Elexon Initial Settlement):** Each SP's price is published approximately
30 minutes after the settlement period closes. SP 1 (ending 00:30 UTC) is available ~01:00 UTC;
SP 17 (ending 08:30 UTC) is available by ~09:00 UTC. By the first hourly run at 08:30 UTC,
approximately SPs 1–16 are settled (~33% of the day). By 12:30 UTC (daily pipeline),
SPs 1–25 are available (~52% of the day). The docstring of `fetch_intraday.py` confirms:
> "By 12:30 UTC (pipeline run time), SP 1–25 (00:00–12:30 BST) are available."

Output: `data/raw/intraday_prices.csv` — overwritten on every run.

---

### 1.2 Forecasting model

**Model type:** `HistGradientBoostingRegressor` (HGBR) from scikit-learn — **not** "HGP".
The code comment in `train_phase3.py` says: *"same base as Phase 2 for fair comparison."*

**Architecture:** Phase 3 two-stage level-shape decomposition (`src/models/train_phase3.py`).

**Stage 1 — Level model (daily HGBR P10/P50/P90):**
- **Target:** `ssp_raw_daily_mean` = daily mean of raw (pre-winsorisation) SSP across all 48 SPs
- **Features:** Daily-aggregated lag stats from before day D — rolling means/stds of SSP,
  NIV, spike counts; day-ahead weather (Open-Meteo); wind % from BMRS WINDFOR; calendar
  harmonics; neg-day classifier output (`neg_price_risk_prob`); up to 50 features at lag ≥ 1 day
- **Guarantee:** All inputs reference data before day D starts — zero leakage
- **Artifacts:** `model_assets/level_q{10,50,90}.pkl`, `model_assets/level_feature_cols.json`

**Stage 2 — Shape model (SP-level HGBR P50):**
- **Target:** `ssp_raw_h − actual_daily_mean_D` = deviation of each SP from the day's actual mean
- **Features:** Fixed-point lags ≥ 48 SPs only (`ssp_lag_48/96/336`, `niv_lag_48/336`,
  `weather_lag_48/336`, `wind_pct_lag_48`, `solar_wm2_lag_48`); daily-level lags; calendar.
  Rolling windows are **strictly excluded** — `shift(1).rolling(w)` contaminates SPs 2–48
- **Guarantee:** All features are settled ≥ 48 SPs before forecast SP — zero leakage for all 48
- **Artifacts:** `model_assets/shape_q50.pkl`, `model_assets/shape_feature_cols.json`

**H+2 shape model:** Same structure with lag ≥ 96 features only.

**Neg-day classifier:** Binary HGBR (P(≥3 negative-price SPs tomorrow)) used as a level feature
to handle renewable oversupply. Artifact: `model_assets/neg_day_classifier.pkl`.

**Retraining schedule:** Weekly via `daily_pipeline.yml` (model age gate — retrain only when the
live model is > 7 days old, at the 12:30 UTC run). Models are frozen between retrains. The intraday
pipeline (`intraday_update.yml`) does **inference only**.

**Holdout performance (last 7 days as of latest run):**

| Metric | Value |
|---|---|
| MAE (P50, all SPs) | £30.35/MWh |
| RMSE | £37.94/MWh |
| sMAPE | 41.3% |
| Level MAE | £13.17/MWh/day |
| Shape correlation | 0.405 |
| Peak timing error | 6.86 SPs |

---

### 1.3 How the dashboard currently "adjusts the position"

The adjustment lives in `src/models/forecast_phase3.py`, lines 706–752, in the
`run_forecast()` function. It runs every intraday call when `intraday_prices.csv` exists.
The relevant block (quoted verbatim):

```python
_SHAPE_ALPHA = 0.4   # dampening: don't over-commit to morning signal
if not _id_today_all.empty:
    try:
        ...
        _settled = set(_actual_map)
        _unsettled_mask = ~result["settlement_period"].isin(_settled)

        # Option 2: compute correction BEFORE overwriting with actuals
        _settled_rows  = result[result["settlement_period"].isin(_settled)]
        _actual_vals   = [_actual_map[sp] for sp in _settled_rows["settlement_period"]]
        _fc_vals       = _settled_rows["ssp_q50"].tolist()
        _mean_err      = float(np.mean([a - f for a, f in zip(_actual_vals, _fc_vals)]))
        _correction    = _SHAPE_ALPHA * _mean_err
        if _unsettled_mask.any() and abs(_correction) > 0.5:
            for _col in ("ssp_predicted", "ssp_q10", "ssp_q50", "ssp_q90"):
                result.loc[_unsettled_mask, _col] = (
                    result.loc[_unsettled_mask, _col] + _correction
                ).round(2)
        log.info("Shape correction: mean_err=£%.1f  applied=£%.1f  (%d unsettled SPs)",
                 _mean_err, _correction, int(_unsettled_mask.sum()))

        # Option 1: replace settled SPs with actuals
        for _sp, _av in _actual_map.items():
            _m = result["settlement_period"] == _sp
            if _m.any():
                for _col in ("ssp_predicted", "ssp_q10", "ssp_q50", "ssp_q90"):
                    result.loc[_m, _col] = round(_av, 2)
                result.loc[_m, "is_actual"] = True
```

**In plain terms:** Compute the mean forecast error (actual − predicted) over all settled SPs.
Apply 40% of that error as a flat bias shift to every unsettled SP. Threshold: no correction
if |shift| ≤ £0.50. The correction is **uniform** — SP 48 gets the same shift as SP 18.
No decay across horizon. No propagation of uncertainty. The quantile width (P10–P90 band) does
not widen despite the correction being uncertain.

**Limitation:** The current approach is a hand-tuned heuristic (`_SHAPE_ALPHA = 0.4`). It
conflates short-run random error with slowly-drifting systematic bias, has no memory across
hourly calls (recomputed from scratch each time), and provides no principled uncertainty
envelope for the correction itself.

---

## 2. Proposed Architecture

The frozen base model (HGBR level + shape) remains **unchanged**. A lightweight Kalman filter
sits on top, tracking the base model's slowly-varying forecast bias and updating it each hour
as new actuals arrive.

---

## 3. PRIMARY: Kalman Filter Residual-Correction Layer

### 3.1 Motivation

The current flat-α correction has three weaknesses:
1. Observation noise is ignored — a single bad SP contaminates the correction as much as many
2. No decay across horizon — SPs settled 12 hours from now get the same raw correction as SP+1
3. No memory — each hourly call discards the previous hour's signal

A scalar Kalman filter on the mean residual resolves all three in 10 lines of arithmetic
and imposes no new model training.

### 3.2 State vector

**Level-only model (recommended for Phase 4 v1):**

```
x_t  ∈ ℝ   — current slowly-varying bias of the base model (£/MWh)
```

The bias is modelled as a random walk: the level drifts, but has no persistent trend.
This is the correct prior for intraday price regime shifts (renewable dispatch changes,
network constraints) that resolve within hours.

**Level + trend extension (Phase 4 v2, if level-only insufficient):**

```
x_t = [b_t, ḃ_t]ᵀ ∈ ℝ²   — bias level and its drift rate (£/MWh, £/MWh per step)
```

State transition matrix: `A = [[1, 1], [0, 1]]`. Use this only if backtest shows that bias
is trending intra-day (e.g., ramping up all morning) rather than mean-reverting. The risk
is overfitting to short-term noise as a trend.

### 3.3 Observation equation

At each hourly update, after fetching settled SPs, compute:

```
z_t = mean( actual[sp] − forecast_q50[sp]  for sp in settled_SPs )
```

This is an unbiased estimate of the current state with observation noise:

```
z_t = x_t + v_t,    v_t ~ N(0, R_t)
R_t = σ²_SP / n_t
```

where `σ²_SP` is the per-SP residual variance (estimated from holdout: MAE ≈ £30 → σ ≈ £35)
and `n_t` is the number of settled SPs at time `t`. Note `R_t` shrinks as the day progresses
and more SPs settle — the filter gains confidence naturally.

### 3.4 Process noise Q

The random-walk variance `Q` controls how fast the bias is allowed to drift between hourly
steps. From the phase3 holdout:
- Level MAE of £13/day → daily level drift ≈ £13 RMS
- 8 hourly steps per day → per-step drift ≈ £13/√8 ≈ £4.6 RMS
- Q ≈ 21 £² (i.e., σ_process ≈ £4.6)

This is the initial estimate. It should be tuned from a backtest by minimising the
normalised innovation squared (NIS) over historical intraday runs.

### 3.5 Kalman recursion (scalar, per hourly call)

Initialise once per day at midnight: `x̂₀ = 0`, `P₀ = R₀` (uninformative prior).

**Predict step** (start of each hourly call, before actuals arrive):
```
x̂⁻_t = x̂_{t-1}
P⁻_t  = P_{t-1} + Q
```

**Update step** (after computing z_t from settled SPs):
```
K_t   = P⁻_t / (P⁻_t + R_t)          # Kalman gain (0 = ignore, 1 = fully trust)
x̂_t   = x̂⁻_t + K_t * (z_t − x̂⁻_t)   # posterior bias estimate
P_t   = (1 − K_t) * P⁻_t             # posterior covariance
```

Cost: 5 scalar operations. Trivially O(1) per hourly update. No matrix inversion.

### 3.6 Propagation and decay across the 48-SP horizon

The estimated bias `x̂_t` is the best point estimate of the correction **right now**.
Further ahead in the day, the correction becomes less reliable because:
- The base model's residuals are less autocorrelated over longer gaps
- The source of the bias (e.g., current grid constraint) may resolve

Apply a per-SP exponential decay:

```
correction(h) = x̂_t · γ^h     for h = 0, 1, …, (48 − n_t − 1)
```

where `h` is the number of SPs ahead of the last settled SP and `γ` is the decay factor.

**Suggested initial γ = 0.966 per SP**, so the correction halves by SP+20 (~10 hours) and
decays to ~20% of current by SP 48. Tune from backtest; a reasonable range is γ ∈ [0.95, 0.99].

**Uncertainty envelope:** Propagate the Kalman posterior variance to widen the PI band:

```
σ²_correction(h) = P_t · γ^{2h}   # variance of correction at step h
```

When writing the corrected Q50 and updating Q10/Q90:
```
corrected_q50(h) = base_q50(h) + correction(h)
corrected_q90(h) = base_q90(h) + correction(h) + z_{0.9} * sqrt(P_t) * γ^h
corrected_q10(h) = base_q10(h) + correction(h) − z_{0.9} * sqrt(P_t) * γ^h
```

where `z_{0.9} ≈ 1.28`. This widens the PI when the Kalman estimate is uncertain
(early in the day, few SPs settled, large `P_t`) and narrows it when the day is nearly complete.

### 3.7 Daily reset

At midnight UTC (or when a new forecast date is detected), reset `x̂ = 0`, `P = P₀`.
The filter has no memory across calendar days — each day's bias correction starts fresh.

### 3.8 Comparison to current approach

| Property | Current (α-correction) | Kalman filter |
|---|---|---|
| Memory across calls | None (recomputed each hour) | Yes (posterior carries forward) |
| Observation noise | Ignored | Modelled (`R_t = σ²/n_t`) |
| Horizon decay | None (uniform shift) | Exponential decay `γ^h` |
| PI update | None (widths unchanged) | Principled widening early-day |
| Tuning knobs | 1 (`_SHAPE_ALPHA`) | 3 (`Q`, `R₀`, `γ`) — all observable |
| Cost per hourly call | O(n_settled) | O(1) after O(n_settled) for `z_t` |
| Retraining required | No | No |
| Implementation lines | ~15 | ~25 |

### 3.9 Cost, drift risk, complexity

- **Cost per hourly update:** O(1) arithmetic after computing `z_t` (O(n_settled) mean). Total
  wall-clock addition: negligible (<1 ms)
- **Drift / overfitting risk:** Low. The random-walk prior means a single anomalous SP cannot
  permanently shift the estimate; it is down-weighted by `K_t < 1`. The daily reset prevents
  drift from accumulating across days. The decay `γ^h` prevents far-horizon SPs from being
  over-corrected based on near-term evidence
- **Implementation complexity:** Low. The scalar filter is 5 equations. State persistence
  requires writing `(x̂_t, P_t)` to a small JSON file (or in-memory between hourly calls if
  the runner is stateful, but in GitHub Actions each run is cold — file persistence needed)

---

## 4. FALLBACK: Online / Incremental Learning

**Document only — do not build yet.**

If the Phase 3 backtest shows that the Kalman layer's level-only correction cannot capture
SP-varying bias patterns (e.g., the base model systematically underestimates morning peak SPs
but overestimates evening), consider nudging the shape model's leaf values via warm-start or
partial_fit on the most recent day's residuals.

**Mechanism:** After a day's actuals are published (D+1 Initial Settlement, ~12:00 UTC),
compute SP-level residuals. Use `HistGradientBoostingRegressor` warm-start (`warm_start=True`,
increment `max_iter` by 10–20 trees) on a rolling window of the last 7–14 days, so the
model learns from recent structure without discarding long-run patterns.

**Overfitting-to-latest-hour risks:**
1. The SP distribution on any single day is 48 data points — far too few for stable gradient steps
2. If the recent day was unusual (price spike, network event), the model internalises that
   as a permanent pattern when it is not
3. `warm_start` for quantile regressors can destabilise quantile ordering (Q10 > Q50 crossings)

**Required guardrails before adoption:**
- Rolling window ≥ 14 days minimum; discard if window MAE improves by <5% vs frozen model
- Hard cap on number of warm-start iterations added per day (≤ 20 trees)
- Enforce quantile monotonicity post-update (`np.sort` across Q10/Q50/Q90 at each SP)
- Full nightly retrain remains the authority; warm-start is discarded and replaced each day
- Gate on Phase 3 walk-forward backtest: adopt only if Kalman shows flat or worsening NIS
  over ≥ 4 seasonal folds, indicating fixed correction strength is insufficient

**Cost per daily update:** O(n_train · n_new_trees) — on a 14-day window (~672 rows) and
20 new trees, roughly 1–3 seconds. Acceptable but much more than the Kalman path.

**Drift risk:** Moderate-to-high without guardrails (see above). With guardrails: similar to
Kalman, but harder to reason about because gradient descent updates many leaves simultaneously.

**Implementation complexity:** Medium. Requires managing the warm-start state across daily
runs, persisting partial models, and adding the quantile ordering fix.

---

## 5. REJECTED: Reinforcement Learning

Evaluated and rejected. RL offers no benefit for this forecasting task: the "action" (bias
correction) is already well-defined by the Kalman formulation, the reward signal (forecast error)
is noisy at the SP level, and training an RL policy requires exploration that would degrade
live forecasts during the learning phase. Reconsider only if Phase 3/5 backtests show that a
fixed correction-decay profile `γ^h` is leaving substantial accuracy on the table across
multiple seasonal folds and the residual autocorrelation structure is strongly SP-position-dependent.

---

## 6. Recommendation

**Build PRIMARY (Kalman filter) in Phase 4.**

The current `α = 0.4` flat correction is already close to a badly-initialised Kalman filter
(K = 0.4, Q → ∞, R → 0, γ = 1). Replacing it with the proper recursion costs ~10 extra lines,
adds no model training, delivers principled uncertainty propagation, and is backtest-auditable
via the NIS statistic. The `(x̂_t, P_t)` state can be persisted to
`model_assets/kalman_state.json` between hourly GitHub Actions calls with no infrastructure change.

**Do not build FALLBACK until:**
1. Kalman layer is live and NIS is measured across ≥ 4 seasonal folds
2. NIS shows consistent over-confidence (actual innovations >> predicted), indicating the
   single-state model is structurally misspecified — not just noisy

| Criterion | Primary (Kalman) | Fallback (online) |
|---|---|---|
| Cost per update | O(1) | O(n_train) |
| Drift / overfit risk | Low | Moderate–high |
| Implementation complexity | Low | Medium |
| Uncertainty quantification | Yes (P_t → PI widening) | No |
| Retraining required | No | No (but daily warm-start) |
| Prerequisite | Phase 3 backtest metrics | Kalman NIS backtest |

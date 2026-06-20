# Workflow Consolidation Design: 3 → 2 Workflows

**Date:** 2026-06-20  
**Status:** DESIGN PASS — no implementation  
**Goal:** Fold `daily_pipeline.yml` into `early_forecast.yml`, leaving two workflows: one consolidated forecast+retrain and the unchanged intraday updater.

---

## 1. Current State (3 Workflows)

| Workflow | Trigger | Duration | Purpose |
|---|---|---|---|
| `daily_pipeline.yml` | 12:30 UTC schedule | ~15 min | Full data fetch + retrain + IS-refine forecast |
| `early_forecast.yml` | 01:00 UTC dispatch | ~3 min | Early day-ahead forecast from IIS+archive-weather |
| `intraday_update.yml` | */30 min | ~1 min | Kalman correction splice (unchanged; stays) |

**Target state:** `daily_pipeline.yml` retired. `early_forecast.yml` promoted to a consolidated `forecast_pipeline.yml` that absorbs all data, retrain, and forecast duties at 01:00 UTC. `intraday_update.yml` unchanged.

---

## 2. What daily_pipeline Does Beyond early_forecast

The current `early_forecast.yml` adds X-1 Elexon rows, injects weather, and runs `forecast_phase3.py`. Daily pipeline does all of that **plus**:

### 2a. Data pipeline (run every day)
- `fetch_elexon.py --append` — already in early_forecast ✓
- `fetch_weather.py --append` — **missing from early_forecast**
- `fetch_generation.py --append` — **missing**
- `fetch_cpi.py` — **missing**
- `extend_dataset.py` — **missing** (builds `dataset_5yr.csv` from raw sources)
- `build_features.py` — **missing** (computes all 128 lag/calendar/weather features → `features_5yr.csv`)
- Save `features_recent.csv` (tail 50 days) — **missing**

**Why the data pipeline must run daily even in the consolidated design:** `features_recent.csv` is the base used by `forecast_phase3.py`'s extension mechanism. If it is N days stale, the extra rows from `system_prices.csv` cover X-N through X-1, but only X-1 has weather injected. The lag-2d through lag-Nd weather features (`temp_c_daily_mean_lag2d`, etc.) are NaN for all intermediate extra rows — the same family of NaN features the weather injection was designed to fix, now re-introduced for all but the newest day. Conclusion: `features_recent.csv` must be rebuilt daily to keep lag-weather features current.

### 2b. Model retrain (currently runs every day)
- `train_phase3.py` — trains 4 models: `level_q10/q50/q90.pkl`, `shape_q50.pkl` + `neg_day_classifier.pkl`; writes `phase3_metrics.json`, `test_predictions_phase3.csv`, feature importance CSVs

### 2c. Streamlit Cloud redeploy touch
- `sed` on `_LAST_PIPELINE_RUN` in `streamlit_app.py` — Streamlit Cloud only redeploys on `.py` changes

### 2d. NOT part of daily_pipeline (clarification)
- `calibrate_pi.py` — **not called by any workflow**; `pi_calibration_v1.json` is a one-time artifact, re-run manually after major model changes
- `tukey_fence.json` — written by `train_lgbm.py` (Phase 2 trainer), also static
- Both are committed artifacts treated as stable between major version bumps

---

## 3. Training Window Shift: IS-through-X−2 vs IS-through-X−1

At 01:00 UTC on day X, the most recent IS (Initial Settlement confirmed) data is through X−2. At 12:30 UTC, the daily pipeline picks up IS through X−1.

**Quantitative impact:**
- Training window: 3 years = 1,095 days = ~52,560 rows (48 SP × 1,095)
- One additional day = 48 rows = **0.09% of training data**
- HGBR with `max_iter=1000` and early stopping is insensitive to this: the loss surface over 52,560 rows changes by less than numerical noise when 48 rows (0.09%) shift in or out
- The 3-year rolling window is chosen to reduce inflation drift, not for precision at the margin

**Verdict:** Retraining on IS-through-X−2 vs IS-through-X−1 produces materially identical models. The window shift is negligible.

---

## 4. Cost of Dropping the 12:30 IS-Refine

The daily pipeline re-runs `forecast_phase3.py` at 12:30 UTC on IS data through X−1. This "IS-refine" gives the model IS prices for yesterday's SPs (vs the IIS prices used at 01:00 UTC). Specifically:

**IIS vs IS price revision (measured on 2,304 overlapping SPs, Apr–Jun 2026):**
- p50 diff: £0.00 (median revision is zero)
- p95 diff: £0.00 (95th percentile revision is also zero)
- SPs with diff > £1.00: 11 of 2,304 (0.5%)
- Mean among revised SPs: £30.19 (concentrated on spike days: Jun 4 negatives, Jun 8 and Jun 11 spikes)
- Overall mean diff: £0.18/MWh

**Impact on the level model:** `ssp_daily_mean_lag1d` has the highest permutation importance (0.598) of any feature in the level model. On spike days (Jun 8: IIS £141–157, IS £83–90 after settlement correction), the IIS lag-1d value misleads the level prediction by up to £60/day. However:
1. These events are rare (0.5% of SPs)
2. The Kalman filter corrects for systematic bias intraday: by the time the daily IS-refine would have run at 12:30, the Kalman has already absorbed SPs 1–25 (the settled portion of today) and partially corrected any level bias from the stale lag
3. For the early 01:00 UTC forecast (the primary consumer), the IS-refine would only improve H+2 (next-day) accuracy slightly; H+1 is already Kalman-corrected throughout the day

**Estimated MAE cost of dropping IS-refine:**
- Average MAE impact: 0.5% × £30 ≈ £0.15–0.18/MWh overall
- Concentrated on spike-event days; on normal days the cost is zero
- Kalman partial compensation: on spike days where IIS ≠ IS, the Kalman observes the actuals intraday and updates x̂ accordingly, absorbing part of the level error
- Net residual: likely £0.05–0.10/MWh after Kalman on typical days; potentially £1–3 on rare spike-settlement days

**Verdict:** Acceptable. The IS-refine catches rare large revisions but the Kalman handles the bulk of intraday correction. The cost is within the £1.50 MAE gate used for early-forecast validation.

---

## 5. Retrain Cadence Recommendation

### Option A: Daily retrain at 01:00 UTC (no change to cadence)
- **Pro:** Model always trained on most recent IS data; consistent with current daily_pipeline behavior
- **Con:** Adds ~6–10 min to the 01:00 UTC path (data fetch ~3 min + build_features ~2 min + retrain ~4–8 min = 12–16 min total); creates a single point of failure where one broken training run blocks both the early forecast and the model for the day
- **Con:** HGBR with early stopping is stable; daily retraining gains <0.1% on a 3-year window

### Option B: Weekly retrain (Monday 01:00 UTC), daily data pipeline
- **Pro:** Keeps daily 01:00 path fast (~4–5 min: fetch + build_features + inject + forecast); retrain only adds latency once per week
- **Pro:** Decouples model freshness from forecast availability; a failed retrain doesn't block Tuesday's early forecast
- **Con:** Model is up to 6 days old; on a 3-year window this is negligible but adds a monitoring responsibility
- **Gate:** Check `phase3_metrics.json` for a `train_date` field; retrain if model age ≥ 7 days

### Option C: Age-based gate (retrain when model > N days old)
- Same as B but N is configurable; N=7 matches B
- Handles gaps (e.g. if Monday's retrain fails, Tuesday checks again and retries)

**Recommendation: Option C (age-based gate, default N=7).** The gate is one `if` block in the consolidated workflow. It gives resilience against retrain failures (Tuesday re-attempts if Monday failed), keeps the normal fast path at ~4–5 min, and limits model staleness to 7 days at worst. On a 3-year HGBR window, 7-day staleness is immaterial.

---

## 6. Proposed Consolidated Architecture

### Workflows (2 total)

```
forecast_pipeline.yml       ← replaces daily_pipeline + promotes early_forecast
intraday_update.yml         ← unchanged
```

### `forecast_pipeline.yml` trigger block

```yaml
on:
  repository_dispatch:
    types: [early-forecast]       # primary: cron-job.org at 01:00 UTC
  schedule:
    - cron: '0 1 * * *'           # fallback (may fire late on free tier)
  workflow_dispatch:              # manual trigger
```

Same external trigger as current `early_forecast.yml`. cron-job.org wires unchanged.

### `forecast_pipeline.yml` step sequence (01:00 UTC)

```
1. Checkout streamlit-data (fetch-depth: 1)
2. Set up Python 3.11
3. Cache pip + restore raw data cache (ISO week key, same as daily_pipeline)
   ↳ cache miss: fetch 3yr full history (elexon + weather + generation)
   ↳ cache hit:  incremental append (elexon + weather + generation)
4. fetch_cpi.py (always; small monthly dataset)
5. extend_dataset.py
6. build_features.py (features_5yr.csv)
7. Save features_recent.csv (tail 50 days)
8. inject_weather_yesterday.py  ← Phase A.1 addition
9. [age gate] if model_age >= 7 days: train_phase3.py
10. forecast_phase3.py
11. Touch streamlit_app.py (_LAST_PIPELINE_RUN = today)
12. git add [all artifacts] + commit "Early forecast update <timestamp>"
    with autostash+retry push (same pattern as both current workflows)
```

### Artifacts committed

All artifacts committed by daily_pipeline + early_forecast:
```
src/dashboard/streamlit_app.py
data/processed/features_recent.csv
data/processed/dataset_5yr.csv (force-add; gitignored)
model_assets/kalman_state.json
model_assets/next_day_forecast_phase3.csv
model_assets/day2_forecast_phase3.csv
model_assets/forecasts/
model_assets/phase3_metrics.json        (only on retrain)
model_assets/test_predictions_phase3.csv (only on retrain)
model_assets/phase3_level_importance.csv (only on retrain)
model_assets/phase3_shape_importance.csv (only on retrain)
model_assets/level_q10/q50/q90.pkl      (only on retrain)
model_assets/shape_q50.pkl              (only on retrain)
model_assets/shape_h2_q50.pkl           (only on retrain)
model_assets/neg_day_classifier.pkl     (only on retrain)
```

### Concurrency

Same group `streamlit-data-commit`, `cancel-in-progress: false`. Unchanged from both existing workflows.

### `intraday_update.yml`

Unchanged. Still fires every 30 min, fetches today's IIS via `fetch_intraday.py`, runs `forecast_phase3.py` in Kalman-only mode, commits.

---

## 7. What Changes vs What Stays the Same

| Item | Current | After consolidation |
|---|---|---|
| 12:30 UTC retrain + IS-refine | `daily_pipeline.yml` | **Dropped** |
| 01:00 UTC data fetch + features | `early_forecast.yml` (missing) | Added to consolidated |
| 01:00 UTC forecast | `early_forecast.yml` | Stays |
| Retrain cadence | Daily (12:30 UTC) | Weekly age-gate (01:00 UTC) |
| PI calibration | Manual / not in any workflow | Unchanged (manual) |
| Streamlit redeploy touch | `daily_pipeline.yml` | Moved to consolidated |
| External trigger (cron-job.org) | `early-forecast` dispatch | Unchanged |
| Intraday Kalman update | `intraday_update.yml` | Unchanged |
| Concurrency group | `streamlit-data-commit` | Unchanged |
| GitHub Actions minutes/month | ~50 min/day + 48 min/day = ~98 min/day | ~5 min/day + 48 min/day = ~53 min/day (−46%) |

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 01:00 cron fires late (GitHub fallback) | High | Low (external cron is primary) | cron-job.org dispatch is primary; schedule is backup |
| Retrain fails on Monday, model goes stale | Low | Low (7-day-old model ≈ negligible MAE drift) | Age gate re-tries next day automatically |
| IS-refine loss on spike settlement day | Rare (0.5% of SPs) | Medium (£1–3 MAE on that day) | Kalman partial compensation; acceptable per §4 |
| Raw data cache miss on first run of week | Low (weekly) | Medium (adds 10–15 min for full 3yr fetch) | Cache miss handled by existing full-history fetch block (unchanged from daily_pipeline) |
| Single point of failure (01:00 run fails) | Low | High (no early forecast + no retrain) | Keep `workflow_dispatch` for manual recovery; intraday_update.yml continues operating from prior model artifacts |

---

## 9. Validation Required Before Cutover

1. **Dry-run the consolidated workflow steps locally:** run data fetch + build_features + inject_weather + forecast_phase3 in sequence; confirm `features_recent.csv` is rebuilt correctly and lag-weather features are non-NaN.

2. **Retrain comparison (IS-through-X−2 vs IS-through-X−1):** run `train_phase3.py` twice on features_5yr.csv ending one day apart; compare `phase3_metrics.json` MAE; confirm delta < £0.50 (expected < £0.10).

3. **Features_recent freshness check:** confirm that after the data pipeline runs at 01:00 UTC, `features_recent.csv` ends at X−1 (not X−2), so the extension mechanism only needs to add X−1's IIS rows (which are weather-injected).

4. **2-week shadow run:** run consolidated workflow manually via `workflow_dispatch` at 01:00 UTC for 5 consecutive days; compare next_day_forecast_phase3.csv to what daily_pipeline produced; confirm MAE difference is < £0.50.

5. **Monitor Kalman state for 2 weeks post-cutover:** watch `last_n_settled` and `x_hat` trajectory for systematic drift not present before; flag if |x_hat| > £5 on 3 consecutive days (possible IS-refine loss effect).

6. **Retire daily_pipeline.yml only after:** 2-week shadow passes AND first Monday weekly retrain succeeds AND monitoring shows no bias increase.

---

## 10. Decision

**Recommend consolidation with weekly retrain gate.** The IS-refine cost is £0.15–0.18/MWh overall (Kalman-compensated), training window shift is 0.09% (negligible), and the architecture simplification (3 → 2 workflows, −46% Actions minutes) is meaningful. The data pipeline (fetch + features_recent rebuild) must remain daily to prevent lag-weather NaN accumulation.

**Implementation scope:** ~60-line update to `early_forecast.yml` (rename to `forecast_pipeline.yml`, add data pipeline steps, add retrain age gate, add Streamlit touch). `daily_pipeline.yml` deleted after validation. No model or inference script changes.

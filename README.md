# UK Electricity System Price Forecasting Platform

Two-day-ahead forecasting of UK electricity System Sell Price (SSP) at 30-minute settlement period resolution. Built on public data from Elexon BMRS, Open-Meteo, Carbon Intensity API, ONS, and BMRS WINDFOR/TSDF.

**Phase 3 — Level-Shape Decomposition · Phase 4 — Kalman Correction + PI Calibration · H+1 and H+2 forecasts · June 2026**

**[Live dashboard →](https://uk-system-price-forecast.streamlit.app/)**

![UK SSP Dashboard](demo/uk-ssp-streamlit.gif)

---

## Architecture

Phase 3 splits the forecasting problem into two independent stages, eliminating recursive error propagation, and produces forecasts for **today (H+1)** and **tomorrow (H+2)** simultaneously. Phase 4 adds a real-time Kalman corrector and split-conformal PI calibration on top of Phase 3 outputs.

**Stage 1 — Level model** (quantile HGBR · P10/P50/P90)
- Predicts the daily mean SSP for the target date
- 85 features: SSP/NIV lags and rolling stats, calendar harmonics, day-ahead Open-Meteo weather, Carbon Intensity wind/gas generation lags, ONS CPIH index and YoY change, negative-price daily count lags, negative-price regime classifier output
- Training targets deflated to real (current-money) terms using monthly CPIH ratio
- 3-year rolling training window · test = 7 days · val = 5 days

**Stage 2H+1 — Shape model** (HGBR · P50, lag ≥ 48)
- Predicts each SP's deviation from the daily mean for today
- 76 features: fixed-point lags ≥ 48 SPs — leakage-free for all 48 forecast SPs simultaneously
- Includes ramp features, same-SP cross-day volatility (`same_sp_std_7d`), wind/gas lag-48/lag-336
- At inference: `solar_wm2_lag_48` → Open-Meteo day-ahead SP-level solar; `wind_pct_lag_48` → BMRS WINDFOR/TSDF

**Stage 2H+2 — Shape model** (HGBR · P50, lag ≥ 96)
- Predicts each SP's deviation for tomorrow — lag-48 excluded (today's prices not settled)
- 50 features: lag-96 replaces lag-48 as primary same-SP signal; all lag-48 references removed
- At inference: WINDFOR for tomorrow's date; Open-Meteo day+2 solar

**Negative-price regime classifier** (binary HGBR)
- Predicts P(≥3 negative-price SPs tomorrow) from wind/solar lags and recent negative-price history
- Output `neg_price_risk_prob` fed as a level model feature — improves daily mean prediction on high-renewable days

**Kalman corrector** (Phase 4 · scalar random-walk filter)
- Applied intraday as each settlement period is confirmed by Elexon
- Models level bias as a scalar random-walk state x̂; prior Q = 0.1, observation noise R = 1.0
- Horizon decay: correction attenuates as γ^h (γ = 0.85) so near-horizon SPs get full bias correction, far-horizon SPs regress toward the base forecast
- Updates q50 only; PI bands are shifted by the same amount (linear, commutes with calibration)
- State persisted in `model_assets/kalman_state.json`; reset on full retrain

**PI calibration** (Phase 4 · split-conformal symmetric widening)
- Applied to both H+1 and H+2 on every pipeline run
- Per-SP widening: `q10_cal[sp] = ssp_q10[sp] − δ(sp)`, `q90_cal[sp] = ssp_q90[sp] + δ(sp)`
- δ(sp) computed once from walk-forward residuals via conformal split; range £13.95–£39.74 across SPs (SP 1 narrowest, SP 37 widest — matches intraday volatility pattern)
- Achieved coverage (walk-forward reference): **79.8%** vs raw model coverage ~38%
- Artifact: `model_assets/pi_calibration_v1.json` · target coverage from `pi_calibration_v1.json` key `achieved_coverage`

**Intraday Nowcast** (Phase 5 · h+1/h+2/h+3 · persistence + empirical bands)
- Point forecast = last settled SP (pure persistence); 80% empirical bands = P10/P90 of residuals r_h = SSP[t+h−1] − SSP[t−1] from an 18-month trailing window (~26,000 SP pairs)
- Regime-aware: NP bands (N-code, normal auction — right-skewed, upside spike risk) vs EN bands (P/K-code, formula price — left-skewed, downside risk)
- Crossover analysis (`docs/persistence-ml-crossover.md`): persistence wins at h+1–h+4; DA+Kalman wins at h+5+; planned blend α·persistence + (1−α)·DA for h+4–h+6 once 6-month forecast archive accumulates (est. Oct 2026)
- Bands stored in `model_assets/nowcast_bands.json`, built by `src/models/build_nowcast_bands.py`, refreshed monthly

**Final forecast per SP:**
```
H+1:  ssp_q[X][sp] = level_P[X](today)    + shape_H1_deviation[sp]
                    + kalman_bias_correction × γ^h
                    ± δ(sp)  [PI calibration]

H+2:  ssp_q[X][sp] = level_P[X](tomorrow) + shape_H2_deviation[sp]
                    ± δ(sp)  [PI calibration; no Kalman on H+2]
```

---

## Results

### Holdout performance (test = 7 days, May 29 – Jun 4 2026)

Includes Jun 3–4 extreme renewable oversupply event (−£70 midday prices, 10 negative SPs on Jun 4).

| Model | MAE (£/MWh) | RMSE | sMAPE | Evaluation |
|---|---|---|---|---|
| Naive lag-48 | 36.34 | 47.73 | 41.6% | batch |
| Seasonal naive lag-336 | 29.40 | 38.29 | 34.7% | batch |
| Rolling mean (48 SP) | 26.78 | 33.07 | 28.5% | batch |
| Phase 2 recursive HGBR | 25.40 | 32.36 | 27.4% | 7-day holdout |
| **Phase 3 H+1 (current)** | **31.61** | **41.63** | **37.8%** | **7-day holdout** |

**Why naive baselines are the right benchmark.** The naive lag-48 model predicts each half-hour slot tomorrow using the same slot's actual price today (`forecast[SP, D] = actual[SP, D−1]`). Seasonal naive lag-336 does the same using the same slot one week ago, additionally capturing the weekday/weekend demand pattern. These are hard to beat because strong intraday and day-to-day autocorrelation is a genuine feature of electricity markets — any model that fails to outperform them is merely rediscovering the lag structure. Beating the seasonal naive on a 119-day walk-forward (£27.39 vs £29.40) with honest out-of-sample evaluation confirms the model extracts signal beyond simple repetition.

> The 7-day test window includes the Jun 4 extreme event (daily mean £36.8, −£70 midday). Excluding Jun 3–4, the model performs in line with the seasonal walk-forward average (£25–27).

### Seasonal walk-forward CV (119 days · 4 folds)

Each fold retrains on data before the fold window — honest out-of-sample evaluation across all seasons.

| Season | Period | MAE | sMAPE | Level MAE | Shape Corr |
|---|---|---|---|---|---|
| Summer | Jul 2025 | £25.42 | 36.2% | £12.51 | 0.291 |
| Autumn | Oct 2025 | £29.58 | 47.5% | £14.94 | 0.489 |
| Winter | Dec 2025 | £20.97 | 33.1% | £8.43 | 0.387 |
| Spring | Apr 2026 | £33.67 | 51.3% | £16.53 | 0.452 |
| **Aggregate** | **119 days** | **£27.39** | **42.0%** | **£13.09** | **0.404** |

**Why sMAPE varies by season:** Winter is most predictable — demand-dominated, gas sets marginal cost consistently. Summer is level-easy but shape-hard (flat profiles). Autumn has the best shape correlation (pronounced demand peaks). Spring is hardest — high renewable penetration and erratic dispatch order.

### PI coverage (Phase 4)

| Window | N (SP-rows) | Coverage | Target |
|---|---|---|---|
| Walk-forward reference (PI-calibrated) | 5,709 | 79.8% | 79.8% |
| Live — all data (26 days, Jun 2026) | 1,248 | 70.8% | 79.8% |

The live shortfall is being tracked in the weekly monitoring report. The live sample is small (26 days); the walk-forward baseline is the primary calibration reference. Raw model coverage before PI calibration was ~38%.

---

## Data sources

| Source | Data | Update |
|---|---|---|
| Elexon BMRS | Settlement prices, NIV | Daily incremental |
| Open-Meteo | Historical weather + day-ahead + day+2 forecast | Daily |
| Carbon Intensity API | Wind %, gas % generation mix (30-min) | Daily |
| ONS (series D7BT) | CPIH index 2015=100, monthly | Monthly |
| BMRS WINDFOR + TSDF | Day-ahead wind forecast (MW) + demand — H+1 and H+2 | Daily at inference |

---

## Project structure

```
src/
  data/
    fetch_elexon.py          Elexon BMRS prices — incremental append
    fetch_historical.py      One-shot 5-year bulk fetch
    fetch_weather.py         Open-Meteo archive
    fetch_generation.py      Carbon Intensity generation mix
    fetch_cpi.py             ONS CPIH index (D7BT)
    fetch_bmrs_forecasts.py  BMRS WINDFOR + TSDF day-ahead wind forecast
    build_dataset.py         Cleaning + Tukey winsorisation
    extend_dataset.py        Safe incremental append to dataset_5yr.csv
  features/
    lag_features.py          SSP/NIV lags, rolling stats, SP dynamic features
    calendar_features.py     Cyclic + annual harmonic encodings
    weather_features.py      Weather lags, degree-day features
    build_features.py        Full SP-level feature pipeline
    level_features.py        Daily aggregation for Stage 1 (incl. neg-price count)
  models/
    train_phase3.py          Two-stage training + H+2 model + seasonal walk-forward CV
    forecast_phase3.py       Non-recursive H+1 and H+2 inference; PI calibration guard
    correctors.py            Kalman corrector, AlphaCorrector, apply_pi_calibration
    build_nowcast_bands.py   Build P10/P90 empirical bands for intraday persistence nowcast
    train_spike_classifier.py  Spike classifier training (config-OFF in production)
    evaluate.py              MAE, RMSE, sMAPE, decomposition metrics
  monitoring/
    build_monitoring_report.py  Weekly PI coverage + Kalman health + Step-3 gate report
  dashboard/
    streamlit_app.py         Streamlit analytics + forecast dashboard

data/
  raw/
    system_prices.csv        Elexon SSP + NIV
    weather_uk.csv           30-min UK weather (3 locations, weighted)
    generation_mix.csv       Carbon Intensity wind %, gas %
    cpi_uk.csv               ONS CPIH monthly index
  processed/
    dataset_5yr.csv          Cleaned + denoised price history
    features_5yr.csv         SP-level feature matrix (128 columns)
    features_recent.csv      Rolling feature matrix used by intraday pipeline

model_assets/
  level_q10/q50/q90.pkl      Stage 1 quantile models (85 features)
  shape_q50.pkl              Stage 2 H+1 shape model (76 features, lag ≥ 48)
  shape_h2_q50.pkl           Stage 2 H+2 shape model (50 features, lag ≥ 96)
  neg_day_classifier.pkl     Negative-price regime classifier
  spike_classifier_v1.pkl    Spike classifier (config-OFF; gated by corrector_config.json)
  level_feature_cols.json    85 level features
  shape_feature_cols.json    76 H+1 shape features
  shape_h2_feature_cols.json 50 H+2 shape features
  neg_day_classifier_feats.json
  phase3_metrics.json        Current test-set metrics
  pi_calibration_v1.json     Per-SP PI widening deltas (δ by SP, achieved_coverage)
  corrector_config.json      Corrector flags: kalman=true, spike_widening=false
  kalman_state.json          Persisted Kalman state (x̂, P, last_updated)
  nowcast_bands.json         P10/P90 empirical bands for h+1/h+2/h+3 nowcast (NP + EN regimes)
  test_predictions_phase3.csv  Test window actuals vs predictions
  walk_forward_predictions.csv 119-day seasonal CV predictions (raw q10/q90)
  phase3_level_importance.csv  Level model Spearman importance
  phase3_shape_importance.csv  Shape model permutation importance
  next_day_forecast_phase3.csv H+1 forecast — today (48 SPs, PI-calibrated)
  day2_forecast_phase3.csv     H+2 forecast — tomorrow (48 SPs, PI-calibrated)
  forecasts/
    forecast_phase3_YYYY-MM-DD.csv  Archived daily forecasts (PI-calibrated from 2026-06-17)

tests/
  test_correctors.py           AlphaCorrector, Kalman, apply_pi_calibration
  test_kalman_corrector.py     Kalman state, horizon decay, edge cases
  test_pi_calibration_guard.py _assert_pi_calibrated scenario matrix (A–F)

reports/
  phase3_root_cause_analysis.md  Jun 4 2026 extreme event analysis + Phase 4 guidance
  annual_modulation_analysis.md
  monitoring/
    2026-W25.md                Weekly monitoring report (first run)
    plots/
      2026-W25_coverage.png    PI coverage timeline + bucket breakdown
      2026-W25_kalman.png      Kalman residual trajectory
      2026-W25_step3.png       Step-3 readiness gate timeline

demo/
  demo_phase3.py             Four-panel demonstration figure
  phase3_demo.png
  uk-ssp-streamlit.gif       Dashboard walkthrough (5-page animated GIF, 3 s/frame)

Project_Brief/
  phase-3-summary.md         Architecture and seasonal analysis
```

---

## Setup and usage

```bash
git clone <repo>
cd uk-system-price-forecast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Daily refresh pipeline

The dashboard **Refresh** button runs these steps automatically:

```bash
python src/data/fetch_elexon.py --append        # latest prices
python src/data/fetch_weather.py                 # latest weather
python src/data/fetch_generation.py --append     # latest wind/gas mix
python src/data/fetch_cpi.py                     # latest CPIH
python src/data/extend_dataset.py                # extend dataset_5yr.csv safely
python src/features/build_features.py \
    --input  data/processed/dataset_5yr.csv \
    --output data/processed/features_5yr.csv
python src/models/train_phase3.py                # retrain + H+2 model
python src/models/forecast_phase3.py             # H+1 (today) + H+2 (tomorrow); PI calibration applied
```

### Intraday updates

GitHub Actions runs every 30 minutes around the clock (48 runs/day), fetching the latest Elexon Initial Settlement prices, applying the Kalman corrector, and committing the updated forecast to the repo. The Streamlit dashboard reads these commits directly so it always shows the most current corrected view.

The intraday nowcast panel (h+1/h+2/h+3) is updated each run. Nowcast bands are rebuilt monthly: `python src/models/build_nowcast_bands.py`.

### Monitoring report

```bash
python src/monitoring/build_monitoring_report.py [--week 2026-W25] [--out reports/monitoring/]
```

Outputs `reports/monitoring/YYYY-WNN.md` with three plot PNGs. Sections: §1 PI coverage overall, §2 by price bucket, §3 ex-ante risk-flag split, §4 Kalman residuals, §5 spike widening (INACTIVE), §6 classifier calibration (INACTIVE), §7 Step-3 readiness gate.

### Evaluation

```bash
# Standard holdout (test = 7 days, val = 5 days)
python src/models/train_phase3.py

# Seasonal walk-forward CV (4 × 30-day folds, no model overwrite)
python src/models/train_phase3.py --walk-forward

# Retrodiction for a specific date
python src/models/forecast_phase3.py --date 2026-06-01
```

### Tests

```bash
pytest tests/ --ignore=tests/test_build_dataset.py -q
# → 55 passed
```

### Dashboard

```bash
.venv/bin/streamlit run src/dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Dashboard sections

| Section | Description |
|---|---|
| Today Forecast (H+1) | 48-SP P50 curve with PI-calibrated P10/P90 band; Stage 1 level prediction |
| Tomorrow Forecast (H+2) | 48-SP P50 with PI-calibrated P10/P90; WINDFOR for tomorrow's date |
| Intraday Nowcast | h+1/h+2/h+3 persistence nowcast; 80% empirical P10/P90 bands; NP/EN regime-aware; updates every 30 min as SPs settle |
| Forecast Verification | Archived Phase 3 forecasts vs Elexon actuals — MAE, RMSE, sMAPE, per-SP error |
| KPIs | Latest SSP, daily average, min/max, spike count |
| SSP Time Series | Daily average SSP with configurable spike threshold |
| Daily Heatmap | SP × date heatmap — intra-day and weekly price patterns |
| Net Imbalance Volume | Daily average NIV (green = long system, red = short) |
| SP Profile | Average 30-min price profile across selected date range |
| Price Derivation Code | P vs N split — when replacement price methodology applies |
| Model Accuracy | Phase 3 test-window series, scatter, error distribution, decomposition metrics |
| Feature Importance | Top-20 for Stage 1 (Spearman) and Stage 2 H+1 (permutation importance) |
| Raw Data | Filterable table with CSV export |

---

## Technical notes

**Two-day forecast horizon** — H+1 uses lag-48 as the primary same-SP signal (yesterday's actual, available at settlement). H+2 drops all lag-48 features (today's prices not settled) and uses lag-96 instead. Both share the same Stage 1 level model; separate shape models encode the different information sets.

**Negative-price regime detection** — a binary HGBR classifier is trained to predict P(≥3 negative-price SPs tomorrow) using wind/solar lags, recent negative-price history, and calendar. Its probability output is injected as a level model feature, enabling the model to anticipate low/negative daily means on high-renewable days.

**Exogenous day-ahead signals at inference** — three genuine day-ahead forecasts replace lag-proxy values at inference time:
- `solar_wm2_lag_48` → Open-Meteo hourly solar for target date (SP-level)
- `wind_pct_lag_48` → BMRS WINDFOR/TSDF wind % for target date (SP-level)
- `wind_pct_lag_336` → BMRS WINDFOR for H+2 date (tomorrow)

This follows the same convention as weather: historical actuals as training proxy, real forecasts at inference.

**Leakage prevention** — shape features use only fixed-point lags ≥ 48 SPs (H+1) or ≥ 96 SPs (H+2). `shift(1).rolling(w)` window features are excluded entirely. Contemporaneous wind/gas actual values are excluded — only lag versions are used.

**CPI deflation** — training targets are multiplied by `cpi_deflator = cpi_latest / cpi_month` so the model learns in real (current-money) terms. The deflator is ≈1.0 at inference.

**3-year training window** — drops data before `today − 1095 days` to reduce the influence of the 2022 gas-price crisis, which is structurally unlike the current market regime.

**Elexon settlement data** — prices are finalised at Initial Settlement on D+1; no intraday actuals are available via the public BMRS API.

**Kalman corrector** — the scalar filter addresses the systematic level bias that persists across the 119-day walk-forward (the model tends to over-predict at low prices, under-predict on high-demand days). The random-walk prior means the filter tracks slowly drifting regime bias rather than noise. Horizon decay γ^h = 0.85^h means the correction is full strength at h=0 (the current SP) and ~10% at h=27 (end of day). PI calibration commutes with Kalman correction (both are linear shifts) so the order of application doesn't affect the archived CSV values.

**PI calibration production guard** — `forecast_phase3.py` raises `RuntimeError` if (a) PI calibration fails at application time, or (b) `pi_calibration_v1.json` is absent when `walk_forward_predictions.csv` is present (the trained-model sentinel). Scenario (b) is the exact failure mode that caused raw ~38% bands to ship before 2026-06-17. A post-write assertion (`_assert_pi_calibrated`) verifies the W25 detector criterion — `std(spread − 2δ) < 1.5 AND std(spread) > 3.0` — on the output DataFrame before it is committed to disk. Both H+1 and H+2 paths are covered.

**Spike classifier** — a binary HGBR trained to flag days with SSP > £150. The classifier is built and evaluated (`spike_classifier_v1_eval.json`: Brier=0.124, AP=0.332) but held config-OFF (`corrector_config.json: spike_widening: false`). The wind-skew bias (training uses CI actuals, inference uses WINDFOR forecasts for `wind_pct_lag_1d`) makes the classifier over-flag on low-wind days; resolving this requires a dedicated training pipeline that mirrors the inference feature at training time.

---

## Approaches explored and parked

These were tested, found to not clearly improve on the current system, and documented here so they won't be re-tried without a specific new motivation.

| Approach | Verdict | Why parked |
|---|---|---|
| Phase 5a: asymmetric per-SP-bucket PI widening | No net gain | The per-bucket asymmetry was overwhelmed by the level-bias in the <£85 and £120-150 buckets; symmetric global calibration is cleaner and interpretable |
| Season-δ: seasonal PI deltas | Marginal at best | 119-day walk-forward is too short to estimate stable seasonal δ; adding 4× parameters with ~30 training days per season risks overfitting to single-season anomalies |
| Asymmetric-global: separate δ_lo and δ_hi | Dominated by level bias | In practice, q10 under-coverage (actual < q10) and q90 under-coverage (actual > q90) are both driven by the same directional level bias on a given day; asymmetric widening doesn't fix the root cause |
| RL-based dynamic PI width | Out of scope | Requires an environment with SSP dynamics close enough to simulate; the current data volume (119 days) is insufficient to train a policy with meaningful coverage guarantees |
| Spike widening (config-ON path) | Built, gated | Works mechanically; held OFF pending wind-skew fix in classifier training (see spike classifier note above) |

---

## Step-3 readiness gate

A gate on revisiting the P95/P99 upper-tail head. GREEN when ≥ 2 usable spike-bearing training autumns (Sep–Nov) are present outside the 2021–2022 energy crisis and the 2023 post-crisis transition year.

| Year | Annual spike rate | Autumn spike rate | Status |
|---|---|---|---|
| 2021 | 62.9% | 100.0% | EXCLUDED — energy crisis |
| 2022 | 96.7% | 95.6% | EXCLUDED — energy crisis |
| 2023 | 57.3% | 52.8% | EXCLUDED — post-crisis transition |
| 2024 | 7.6% | 17.6% | EXCLUDED — annual rate too quiet (<10%) |
| 2025 | 22.5% | 18.7% | USABLE — first qualifying autumn |
| 2026 | — | — | PENDING — autumn not yet settled |

**Current gate: RED (1/2 autumns). Trigger: autumn 2026 settled, est. Dec 2026.**

Tracked automatically in the weekly monitoring report §7.

---

## Known limitations and Phase 4 guidance

See `reports/phase3_root_cause_analysis.md` for a full diagnosis of the Jun 4 2026 extreme event (renewable oversupply, −£70 midday prices). Key remaining gaps:

| Gap | Suggested fix |
|---|---|
| No SP-level demand forecast | BMRS TSDF boundary='N' at SP level as H+1/H+2 shape feature |
| Gas % has no day-ahead product | Use CI lag-48/lag-336 proxy; explore gas futures as level feature |
| Negative-price recall still limited | Specialist model for high-renewable-oversupply days |
| H+2 shape weaker (50 vs 76 features) | Add NIV lag-96, weather lag-96, wind/gas lag-96 to training data |
| Live PI coverage 70.8% vs 79.8% target | Under investigation via monitoring report; 26-day sample still small |

---

## Model progression

| Phase | Architecture | MAE | PI coverage | Note |
|---|---|---|---|---|
| Phase 1 | Batch HGBR | £15.01 | — | Leaky — evaluated on training data |
| Phase 2 | Recursive HGBR (honest) | £25.40 | — | Recursive SP-by-SP, 7-day holdout |
| Phase 3 H+1 | Level-Shape + neg-price classifier | £31.61 | ~38% (raw) | 7-day holdout incl. extreme Jun event |
| Phase 3 walk-forward | Level-Shape (4-season CV) | £27.39 | ~38% (raw) | 119-day honest seasonal evaluation |
| Phase 4 | + Kalman corrector + PI calibration | £27.39 | **79.8%** | Intraday Kalman live; PI calibrated on walk-forward reference |

The 7-day holdout MAE (£31.61) is elevated by the Jun 3–4 renewable oversupply event. The 4-season walk-forward (£27.39) is the more representative performance estimate across normal market conditions. Phase 4 does not change point-forecast MAE — it corrects systematic level bias intraday and widens the PI to achieve the target conformal coverage.

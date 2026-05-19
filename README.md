# UK Electricity System Price Forecasting Platform

An end-to-end data science project for forecasting UK electricity system prices (SSP) at the settlement-period level (30-minute intervals). Built on public data from Elexon BMRS and Open-Meteo.

**Phase 3 — Level-Shape Decomposition · Wind/Gas exogenous features added May 2026**

---

## What it does

- **Ingests** 5 years of Elexon BMRS settlement data (May 2021 – May 2026) with smart incremental updates
- **Fetches** UK weather history and day-ahead forecasts from Open-Meteo (temperature, wind speed, solar irradiance, precipitation) across three representative UK locations
- **Engineers** features at two resolutions: 65 SP-level features (lag-48+ only for the shape model, includes wind/gas lag-48/lag-336) and 79 daily-level features for the level model (includes wind/gas daily lags + rolling mean)
- **Trains** a two-stage decomposition model with no recursive error propagation:
  - **Stage 1 — Level model**: quantile HGBR (P10/P50/P90) predicts the day's average SSP from daily-aggregated history
  - **Stage 2 — Shape model**: HGBR predicts each SP's deviation from the daily mean using only fixed lag-48+ features
- **Evaluates** with a non-recursive two-stage simulation: honest P50 MAE **£27.39/MWh · RMSE £35.13** on May 11–17 2026 (Level MAE £15.83/MWh/day), with wind/gas generation exogenous inputs now feeding both stages
- **Forecasts** tomorrow's 48 settlement periods without any within-day recursion — both models use only data available before the forecast day starts
- **Visualises** everything in a Streamlit dashboard: level vs shape decomposition metrics, P10/P90 uncertainty bands, historical analytics, model accuracy, and feature importance

---

## Results

| Model | MAE (£/MWh) | RMSE | sMAPE | Evaluation |
|---|---|---|---|---|
| Naive (lag-48) | 36.34 | 47.73 | 41.6% | batch |
| Seasonal naive (lag-336) | 29.40 | 38.29 | 34.7% | batch |
| Rolling mean (48 SP) | 26.78 | 33.07 | 28.5% | batch |
| ~~HGBR Phase 1 (batch/leaky)~~ | ~~15.01~~ | ~~22.91~~ | ~~17.9%~~ | ~~leaky batch~~ |
| Quantile HGBR P50 · Phase 2 | 25.40 | 32.36 | 27.4% | honest recursive |
| **Level-Shape HGBR P50 · Phase 3** | **27.39** | **35.13** | **30.3%** | **honest non-recursive** |

Test period: 7 days (May 11–17 2026), 336 settlement periods.

**Phase 3 decomposition diagnostics:**

| Metric | Value | Meaning |
|---|---|---|
| Level MAE | £15.83/MWh/day | Error in predicting the day's average price (Stage 1) |
| Shape correlation | 0.327 | Mean Pearson r between predicted and actual intra-day profiles |
| Peak timing error | 4.4 SPs | Mean absolute offset between predicted and actual daily peak (±2 h) |

> **Phase 3 vs Phase 2:** Phase 2's recursive loop propagates prediction errors across all 48 steps — a drift in SP5 contaminates every subsequent prediction that day. Phase 3 eliminates this by splitting the problem: Stage 1 predicts the day's price level from daily-aggregated history (no recursion), and Stage 2 predicts the intra-day shape using only fixed lag-48+ features (safe for every SP simultaneously). Wind generation % and gas % from the Carbon Intensity API are now included as exogenous features in both stages, capturing the low-wind → high-gas dispatch → high-price causal chain that drove the May 18 overnight anomaly (16.4% wind mean day). At inference, CI actual lag-48 data is fetched automatically for the same-day-yesterday and week-prior settlement periods.

> **Why rolling features are excluded from the shape model:** `shift(1).rolling(w)` features include within-day actual prices for SPs 2–48 of the forecast day (only SP1 is clean). Even `ssp_roll_mean_336` contains up to 47/336 ≈ 14% within-day contamination for late-day SPs. Phase 3 uses only fixed-point lags (`ssp_lag_48`, `ssp_lag_96`, `ssp_lag_336`) — guaranteed leakage-free for every SP in the 48-period forecast window. An earlier version of the code had a substring matching bug that silently included 32 leaky SP-level rolling features in the shape model; this inflated the reported MAE from £25.39 to £22.18 (the ~12% apparent gain was entirely leakage-driven). The bug has been fixed and all metrics reflect clean evaluation.

Top features by permutation importance (val set, Phase 2 P50 model): `ssp_lag_1` (22.0), `net_imbalance_volume_lag_1` (1.3), `ssp_lag_2` (0.34), `ssp_roll_mean_6` (0.25), `ssp_lag_48` (0.22), `solar_wm2_lag_1` (0.17), `cos_sp`/`sin_sp` (intra-day cycle).

Annual modulation confirmed statistically (Kruskal-Wallis p = 5.4 × 10⁻¹¹): prices peak in December/January, trough in May, with a secondary summer peak — driven by heating demand seasonality. The 2022 Russia-Ukraine energy crisis is visible as a structural outlier.

---

## Folder structure

```
data/
    raw/
        system_prices.csv         # Elexon BMRS — SSP, NIV, price derivation code
        weather_uk.csv            # Open-Meteo — 30-min UK weather (3 locations, weighted)
    processed/
        dataset_5yr.csv           # Cleaned + denoised (Tukey outer-fence winsorisation)
        features_5yr.csv          # 87,686 rows × 114 columns — full SP-level feature matrix (incl. wind/gas)

src/
    data/
        fetch_elexon.py           # Smart incremental Elexon ingest (concurrent, day-level)
        fetch_historical.py       # One-shot 5-year bulk fetch (ThreadPoolExecutor)
        fetch_weather.py          # Open-Meteo historical archive fetch
        fetch_generation.py       # Carbon Intensity API — wind %, gas % generation mix (30-min)
        build_dataset.py          # Cleaning, denoising, derived columns
    features/
        calendar_features.py      # Temporal + cyclic + annual harmonic features
        lag_features.py           # SSP/NIV lags, rolling stats, spike memory, NIV extremes
        weather_features.py       # Weather lags, rolling stats, degree/ramp features
        build_features.py         # Full SP-level feature engineering pipeline
        level_features.py         # Daily-level aggregation for Stage 1 level model
    models/
        evaluate.py               # MAE, RMSE, sMAPE + decomposition metrics
        train_baseline.py         # Three lag-based baselines
        train_lgbm.py             # Phase 2: quantile HGBR + spike classifier (recursive)
        train_phase3.py           # Phase 3: two-stage level-shape training + evaluation
        forecast.py               # Phase 2: recursive day-ahead inference (P10/P50/P90)
        forecast_phase3.py        # Phase 3: non-recursive two-stage inference
    dashboard/
        streamlit_app.py          # Streamlit analytics + forecast dashboard

model_assets/
    # Phase 3 models
    level_q10.pkl                 # Stage 1 level model — P10
    level_q50.pkl                 # Stage 1 level model — P50 (daily mean point forecast)
    level_q90.pkl                 # Stage 1 level model — P90
    shape_q50.pkl                 # Stage 2 shape model — P50 deviation forecast
    level_feature_cols.json       # 79 daily-level features for Stage 1 (incl. wind/gas lags)
    shape_feature_cols.json       # 65 SP-level lag-48+ features for Stage 2 (incl. wind/gas lag-48/lag-336)
    phase3_metrics.json           # Phase 3 test-set metrics + decomposition diagnostics
    test_predictions_phase3.csv   # Actuals vs Phase 3 predictions (May 11–17)
    next_day_forecast_phase3.csv  # Latest Phase 3 day-ahead forecast (48 SPs)
    # Phase 2 models (retained for comparison)
    hgbr_q10.pkl                  # Phase 2 quantile HGBR — P10
    hgbr_q50.pkl                  # Phase 2 quantile HGBR — P50
    hgbr_q90.pkl                  # Phase 2 quantile HGBR — P90
    spike_classifier.pkl          # Binary HGBR classifier (class_weight="balanced")
    tukey_fence.json              # Winsorisation bounds {"lower": -156.6, "upper": 353.7}
    feature_cols.json             # 91 SP-level features for Phase 2 models
    hgbr_feature_importance.csv   # Permutation importance (val set)
    hgbr_metrics.json             # Phase 2 honest recursive test-set metrics
    baseline_metrics.json         # Three baseline model metrics
    test_predictions.csv          # Phase 2 actuals vs predictions (May 11–17)
    next_day_forecast.csv         # Latest Phase 2 day-ahead forecast (48 SPs)
    forecasts/
        forecast_YYYY-MM-DD.csv         # Phase 2 archived daily forecasts
        forecast_phase3_YYYY-MM-DD.csv  # Phase 3 archived daily forecasts

reports/
    annual_modulation_analysis.md  # Statistical analysis of annual price seasonality
```

---

## Setup

```bash
git clone <repo>
cd uk-system-price-forecast
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the pipeline

### 1 — Fetch raw data

```bash
# Elexon BMRS (last 5 years, concurrent fetch — takes ~10s)
python src/data/fetch_historical.py

# Or incrementally update an existing file
python src/data/fetch_elexon.py --append

# Open-Meteo weather (last 5 years)
python src/data/fetch_weather.py
```

### 2 — Build dataset and features

```bash
python src/data/build_dataset.py --raw data/raw/system_prices_5yr.csv \
                                  --out data/processed/dataset_5yr.csv

python src/features/build_features.py --input  data/processed/dataset_5yr.csv \
                                       --output data/processed/features_5yr.csv
```

### 3 — Train models

**Phase 3 (recommended):**
```bash
python src/models/train_phase3.py
```
Trains a two-stage level-shape decomposition model. Outputs `level_q10/q50/q90.pkl`, `shape_q50.pkl`, `level_feature_cols.json`, `shape_feature_cols.json`, `phase3_metrics.json`, `test_predictions_phase3.csv`.

**Phase 2 (retained for comparison):**
```bash
python src/models/train_lgbm.py
```
Trains quantile HGBR P10/P50/P90 + spike classifier with recursive evaluation. Outputs `hgbr_q10/q50/q90.pkl`, `spike_classifier.pkl`, `hgbr_metrics.json`, `test_predictions.csv`.

### 4 — Generate day-ahead forecast

```bash
# Phase 3 (non-recursive, recommended)
python src/models/forecast_phase3.py
python src/models/forecast_phase3.py --date 2026-05-20

# Phase 2 (recursive, for comparison)
python src/models/forecast.py
```

Phase 3 fetches live weather from Open-Meteo, predicts the daily level, then predicts 48 SP deviations — no recursive loop. Saves `model_assets/next_day_forecast_phase3.csv`.

### 5 — Launch the dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

The **Refresh Data & Run Forecast** button in the sidebar runs data fetch + both Phase 2 and Phase 3 forecasts automatically.

---

## Dashboard sections

| Section | What it shows |
|---|---|
| Day-Ahead Forecast | 48-period P50 curve with P10/P90 band; predicted daily level from Stage 1 |
| KPI row | Latest SSP, average, min, max, spike count for the selected date range |
| SSP Time Series | Daily average SSP with configurable spike threshold overlay |
| Daily Heatmap | Settlement-period × date heat map — reveals intra-day and weekly patterns |
| Net Imbalance Volume | Daily average NIV bar chart (green = long, red = short) |
| Settlement Period Profile | Average 30-minute price profile across selected date range |
| Price Derivation Code | P vs N code breakdown — how often replacement price methodology triggers |
| Model Forecast vs Actual | Phase 3 test-week series, scatter, error histogram, daily error bars; decomposition metrics: level MAE, shape correlation, peak timing |
| Live Forecast Verification | Compares archived Phase 3 forecasts against Elexon actuals — MAE, RMSE, sMAPE, error by SP |
| Feature Importance | Top-20 features by permutation MAE reduction with uncertainty bars |
| Raw data | Filterable table with CSV download |

---

## Technical notes

**Level-shape decomposition** — the key Phase 3 innovation. Electricity prices on any given day can be decomposed into (1) a daily level (how expensive the day is on average) and (2) an intra-day shape (how prices vary across the 48 SPs relative to the level). Both are predictable from different information sets: level from daily-aggregated history, shape from yesterday's same-SP prices and calendar features. Separating them eliminates recursive error propagation entirely.

**Leakage prevention (shape features)** — rolling window features (`shift(1).rolling(w)`) include within-day actual prices for SPs 2–48 of the forecast day. Only `ssp_roll_mean_w` for SP1 is clean; for SP48 the window is 98% within-day contaminated. Phase 3 therefore uses only fixed-point lags (`ssp_lag_48`, `ssp_lag_96`, `ssp_lag_336`, `weather_lag_48`, `niv_lag_48`) — safe for every SP in the 48-period window simultaneously.

**Leakage prevention (evaluation)** — batch evaluation with pre-computed feature tables is leaky: `ssp_lag_1` contains actual within-day prices. Phase 2 fixed this with `evaluate_dayahead_recursive()`. Phase 3 eliminates the problem structurally: neither model stage uses any feature with lag < 48 SPs, so no feature overriding is needed during evaluation.

**Leakage prevention (features)** — contemporaneous columns excluded: `replacement_price` (corr = 0.9999 with SSP), `price_derivation_code_P` (corr = 0.69), `abs_imbalance_volume`, and contemporaneous `net_imbalance_volume`. Failure to exclude these produced MAE ≈ 0.72 — a near-perfect but fully leakage-driven result.

**Level model features (73)** — daily-aggregated lags of SSP/NIV/spike counts (shift ≥ 1 day), rolling 7/14/28-day stats, calendar (day-of-week, month, annual harmonics), and daily-average weather for the forecast day (from Open-Meteo day-ahead forecast). All reference data before the forecast day begins.

**Shape model features (90)** — fixed-point lag-48+ SP-level features: `ssp_lag_48/96/336`, `ssp_raw_lag_48/96/336`, `is_spike_lag_48/336`, `niv_lag_48/336`, `weather_lag_48`, `heating_degree`, `cooling_degree`, calendar (SP position, day-of-week, month, annual harmonics), and daily-level lag features merged from Stage 1 (`ssp_daily_mean_lag1d`, `ssp_lag48_deviation`). Rolling features entirely excluded.

**Shape target** — `ssp_raw_h − actual_daily_mean_D`. The actual daily mean is used only as the supervised learning target during training; it is never a feature. At inference time, the Stage 1 predicted level is added to the Stage 2 predicted deviation.

**Training target** — both stages train on `ssp_raw` (un-winsorised actual prices). The Tukey outer fence (`lower=-156.60`, `upper=353.70`) is retained for the Phase 2 recursive lag features (`ssp` column), not for the Phase 3 models.

**SSP = SBP** — confirmed by design, not a data error. ~50% of settlement periods use "P" (replacement price) methodology where SSP = SBP by definition. SBP is removed as redundant.

**Annual modulation** — statistically confirmed (p = 5.4 × 10⁻¹¹) but explains only R² = 2.9% of variance. Short-range autocorrelation (`ssp_lag_1`) dominates at SP level; at daily level, weekly and monthly harmonics contribute more meaningfully to the level model.

**Live verification loop** — every time `forecast_phase3.py` runs it archives the forecast to `model_assets/forecasts/forecast_phase3_YYYY-MM-DD.csv`. The dashboard's verification panel prefers Phase 3 archives and falls back to Phase 2 archives for dates not yet re-forecast.

**Model choice** — `HistGradientBoostingRegressor` (sklearn) used in place of LightGBM; same histogram-based algorithm, no external OpenMP dependency. Swap back with `brew install libomp` + LightGBM if needed.

---

## Motivation

UK electricity markets exhibit strong 30-minute periodicity, renewable intermittency, annual demand seasonality, and occasional extreme price spikes — a challenging forecasting environment that benefits from careful feature engineering over model complexity. This project demonstrates a realistic DS workflow: automated ingestion, denoising, systematic feature construction, leakage-aware model training and evaluation, decomposition-based inference without recursive error propagation, exogenous generation mix inputs (Carbon Intensity API), and interactive visualisation. The progression from Phase 1 (leaky batch, MAE £15.0) → Phase 2 (honest recursive, £25.4) → Phase 3 (non-recursive decomposition with wind/gas exogenous features, Level MAE £15.8/MWh/day) illustrates both the pitfalls of naive evaluation and the gains from architectural choices grounded in the causal structure of the problem.

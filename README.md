# UK Electricity System Price Forecasting Platform

An end-to-end data science project for forecasting UK electricity system prices (SSP) at the settlement-period level (30-minute intervals). Built on public data from Elexon BMRS and Open-Meteo.

**Phase 2 — Quantile forecasting + spike detection · In progress May 2026**

---

## What it does

- **Ingests** 5 years of Elexon BMRS settlement data (May 2021 – May 2026) with smart incremental updates
- **Fetches** UK weather history and day-ahead forecasts from Open-Meteo (temperature, wind speed, solar irradiance, precipitation) across three representative UK locations
- **Engineers** 91 features covering price lags/rolling statistics, spike memory, NIV stress indicators, calendar/annual harmonics, and weather-driven supply-demand signals
- **Trains** three quantile HGBR models (P10/P50/P90) on raw (un-winsorised) prices, plus a binary spike classifier — enabling calibrated uncertainty bands and explicit peak-risk signals
- **Evaluates** with a correct recursive day-ahead simulation that mirrors the deployment loop: honest P50 MAE **£25.40/MWh · RMSE £32.36** on May 11–17 2026 (gap of ~£11 vs prior leaky batch metric)
- **Forecasts** tomorrow's 48 settlement periods (00:00–23:30) using recursive multi-step inference with live weather
- **Visualises** everything in a Streamlit dashboard: P10/P90 uncertainty bands, spike probability chart, historical analytics, model accuracy, and feature importance

---

## Results

| Model | MAE (£/MWh) | RMSE | sMAPE | Evaluation |
|---|---|---|---|---|
| Naive (lag-48) | 36.34 | 47.73 | 41.6% | batch |
| Seasonal naive (lag-336) | 29.40 | 38.29 | 34.7% | batch |
| Rolling mean (48 SP) | 26.78 | 33.07 | 28.5% | batch |
| ~~HGBR Phase 1 (batch/leaky)~~ | ~~15.01~~ | ~~22.91~~ | ~~17.9%~~ | ~~leaky batch~~ |
| **Quantile HGBR P50 · Phase 2** | **25.40** | **32.36** | **27.4%** | **honest recursive** |

Test period: 7 days (May 11–17 2026), 336 settlement periods.

> **Why the gap?** The Phase 1 batch metric used pre-computed features where `ssp_lag_1` captured actual within-day prices — contaminating 47 of 48 settlement periods per test day. Phase 2 evaluation uses `evaluate_dayahead_recursive()`, which overrides all short-lag features with running model predictions, exactly matching the deployment loop. The honest MAE is £25.40/MWh — still 30% better than the seasonal-naive baseline.

Top features by permutation importance (val set): `ssp_lag_1` (22.0), `net_imbalance_volume_lag_1` (1.3), `ssp_lag_2` (0.34), `ssp_roll_mean_6` (0.25), `ssp_lag_48` (0.22), `solar_wm2_lag_1` (0.17), `cos_sp`/`sin_sp` (intra-day cycle).

Annual modulation confirmed statistically (Kruskal-Wallis p = 5.4 × 10⁻¹¹): prices peak in December/January, trough in May, with a secondary summer peak — driven by heating demand seasonality. The 2022 Russia-Ukraine energy crisis is visible as a structural outlier.

---

## Folder structure

```
data/
    raw/
        system_prices.csv       # Elexon BMRS — SSP, NIV, price derivation code
        weather_uk.csv          # Open-Meteo — 30-min UK weather (3 locations, weighted)
    processed/
        dataset_5yr.csv         # Cleaned + denoised (Tukey outer-fence winsorisation)
        features_5yr.csv        # 87,686 rows × 108 columns — full feature matrix

src/
    data/
        fetch_elexon.py         # Smart incremental Elexon ingest (concurrent, day-level)
        fetch_historical.py     # One-shot 5-year bulk fetch (ThreadPoolExecutor)
        fetch_weather.py        # Open-Meteo historical archive fetch
        build_dataset.py        # Cleaning, denoising, derived columns
    features/
        calendar_features.py    # Temporal + cyclic + annual harmonic features
        lag_features.py         # SSP/NIV lags, rolling stats, spike memory, NIV extremes
        weather_features.py     # Weather lags, rolling stats, degree/ramp features
        build_features.py       # Full feature engineering pipeline
    models/
        evaluate.py             # MAE, RMSE, sMAPE, metrics reporting
        train_baseline.py       # Three lag-based baselines
        train_lgbm.py           # Quantile HGBR + spike classifier training; honest recursive eval
        forecast.py             # Day-ahead recursive 48-period inference (P10/P50/P90 + spike prob)
    dashboard/
        streamlit_app.py        # Streamlit analytics + forecast dashboard

model_assets/
    hgbr_q10.pkl                # Quantile HGBR — P10 model
    hgbr_q50.pkl                # Quantile HGBR — P50 model (point forecast)
    hgbr_q90.pkl                # Quantile HGBR — P90 model
    spike_classifier.pkl        # Binary HGBR classifier (class_weight="balanced")
    tukey_fence.json            # Winsorisation bounds {"lower": -156.6, "upper": 353.7}
    feature_cols.json           # Exact 91-feature list used in training
    hgbr_feature_importance.csv # Permutation importance (val set)
    hgbr_metrics.json           # Honest recursive test-set evaluation metrics
    baseline_metrics.json       # Three baseline model metrics
    test_predictions.csv        # Actuals vs P10/P50/P90 predictions (May 11–17)
    next_day_forecast.csv       # Latest day-ahead forecast (48 SPs)
    forecasts/
        forecast_YYYY-MM-DD.csv # Archived daily forecasts for live verification

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

```bash
python src/models/train_lgbm.py
```

Trains three quantile HGBR models (P10/P50/P90) on raw prices plus a binary spike classifier. Outputs:
- `model_assets/hgbr_q10.pkl`, `hgbr_q50.pkl`, `hgbr_q90.pkl`
- `model_assets/spike_classifier.pkl`
- `model_assets/tukey_fence.json`
- `model_assets/feature_cols.json`
- `model_assets/hgbr_feature_importance.csv`
- `model_assets/hgbr_metrics.json` (honest recursive MAE)
- `model_assets/test_predictions.csv`

### 4 — Generate day-ahead forecast

```bash
python src/models/forecast.py
# Or for a specific date:
python src/models/forecast.py --date 2026-05-20
```

Fetches live weather from Open-Meteo and runs recursive 48-step inference, outputting P10/P50/P90 quantile bands and per-period spike probability. Saves `model_assets/next_day_forecast.csv`.

### 5 — Launch the dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

The **Refresh Data & Run Forecast** button in the sidebar runs steps 1 + 4 automatically and refreshes the display.

---

## Dashboard sections

| Section | What it shows |
|---|---|
| Day-Ahead Forecast | 48-period P50 price curve with P10/P90 uncertainty band; Min/Avg/Max/Peak-P90-risk metrics |
| Spike Probability | Per-settlement-period spike probability bars (red >30%, orange >10%) |
| KPI row | Latest SSP, average, min, max, spike count for the selected date range |
| SSP Time Series | Daily average SSP with configurable spike threshold overlay |
| Daily Heatmap | Settlement-period × date heat map — reveals intra-day and weekly patterns |
| Net Imbalance Volume | Daily average NIV bar chart (green = long, red = short) |
| Settlement Period Profile | Average 30-minute price profile across selected date range |
| Price Derivation Code | P vs N code breakdown — how often replacement price methodology triggers |
| Model Forecast vs Actual | Test-week time series with P10/P90 band, scatter, error histogram, daily error bars; high-price MAE separate metric |
| Live Forecast Verification | Compares each archived day-ahead forecast against Elexon actuals — MAE, RMSE, sMAPE, error by settlement period, error histogram |
| Feature Importance | Top-20 features by permutation MAE reduction with uncertainty bars |
| Raw data | Filterable table with CSV download |

---

## Technical notes

**Leakage prevention (features)** — three contemporaneous columns excluded from features: `replacement_price` (corr = 0.9999 with SSP), `price_derivation_code_P` (corr = 0.69), `abs_imbalance_volume`. Contemporaneous `net_imbalance_volume` is also excluded; only lag-1 (30-min-old) and lag-48 are used. Failure to exclude these produced MAE ≈ 0.72 — a near-perfect but fully leakage-driven result.

**Leakage prevention (evaluation)** — batch evaluation with pre-computed feature tables is leaky: `ssp_lag_1` records the actual price 30 minutes ago, which is unavailable at day-ahead dispatch. `evaluate_dayahead_recursive()` replays each test day end-to-end, overriding `ssp_lag_1/2` and all 6-period rolling SSP/NIV features with running model predictions — matching `forecast.py`'s deployment loop exactly. The honest P50 MAE is £25.40/MWh, approximately £11 worse than the leaky batch figure.

**Training target** — Phase 2 trains on `ssp_raw` (un-winsorised actual prices, max ~£4038) rather than the Tukey-clipped `ssp`. This removes the artificial £354 ceiling on predictions, allowing the model to forecast genuine spikes. The Tukey outer fence (`lower=-156.60`, `upper=353.70`) is still used to clip the `ssp` column used as autoregressive lag features, preventing spike contamination from propagating into the smooth price signal.

**Spike classifier** — `HistGradientBoostingClassifier` with `class_weight="balanced"` addresses the ~2.4% spike rate. Outputs per-period probability that `ssp_raw` exceeds the Tukey upper fence (£353.70). Used in the dashboard for risk visualisation; does not alter the P50 point forecast.

**Spike memory features** — `ssp_raw_lag_{48,96,336}`, `is_spike_lag_{48,336}`, `spike_count_roll_48`, `is_negative_lag_{48,336}`, `neg_count_roll_48`: all use lags ≥ 48 SPs so they always reference settled actual data at day-ahead inference time (beyond the 48-period forecast horizon).

**NIV extreme features** — `niv_roll_min/max_{6,48}`, `abs_niv_roll_mean_{6,48}`: capture how short or long the system has been in recent hours — direct antecedents of price spikes and crashes. Used with lag-1 shift so no contemporaneous leakage.

**SSP = SBP** — confirmed by design, not a data error. ~50% of settlement periods use "P" (replacement price) methodology where SSP = SBP by definition. SBP is removed as redundant.

**Recursive forecasting** — `ssp_lag_1` and `ssp_lag_2` are filled from the model's own running predictions. Lags ≥ 48 always reference actual history. NIV within the forecast window is proxied by yesterday's same settlement period. Quantile ordering is enforced post-prediction: `pred_q10 = min(q10, q50)`, `pred_q90 = max(q90, q50)`.

**Annual modulation** — statistically confirmed (p = 5.4 × 10⁻¹¹) but explains only R² = 2.9% of variance. Short-range autocorrelation (`ssp_lag_1`) dominates. Annual harmonic features are included but contribute marginal lift.

**Live verification loop** — every time `forecast.py` runs it archives the forecast to `model_assets/forecasts/forecast_YYYY-MM-DD.csv`. The dashboard's verification panel automatically detects which archived dates have Elexon actuals available and surfaces MAE, RMSE, sMAPE, and per-settlement-period errors for each verified day.

**Model choice** — `HistGradientBoostingRegressor` / `Classifier` (sklearn) used in place of LightGBM; same histogram-based algorithm, no external OpenMP dependency. Swap back with `brew install libomp` + LightGBM as noted in `train_lgbm.py`.

---

## Motivation

UK electricity markets exhibit strong 30-minute periodicity, renewable intermittency, annual demand seasonality, and occasional extreme price spikes — a challenging forecasting environment that benefits from careful feature engineering over model complexity. This project demonstrates a realistic DS workflow: automated ingestion, denoising, systematic feature construction, leakage-aware model training and evaluation, recursive inference with calibrated uncertainty, and interactive visualisation.

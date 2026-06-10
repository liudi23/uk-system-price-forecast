# UK Electricity System Price Forecasting Platform

An end-to-end data science project for forecasting UK electricity system prices (SSP) at the settlement-period level (30-minute intervals). Built on public data from Elexon BMRS and Open-Meteo.

**Phase 1 MVP — shipped May 2026**

---

## What it does

- **Ingests** 5 years of Elexon BMRS settlement data (May 2021 – May 2026) with smart incremental updates
- **Fetches** UK weather history and day-ahead forecasts from Open-Meteo (temperature, wind speed, solar irradiance, precipitation) across three representative UK locations
- **Engineers** 76 features covering price lags/rolling statistics, calendar/annual harmonics, and weather-driven supply-demand signals
- **Trains** a HistGradientBoosting (HGBR) model achieving **MAE £25.9/MWh · sMAPE 28.0%** on the May 11–17 2026 test week (honest recursive day-ahead evaluation)
- **Forecasts** tomorrow's 48 settlement periods (00:00–23:30) using recursive multi-step inference with live weather
- **Visualises** everything in a Streamlit dashboard: historical analytics, model accuracy, day-ahead forecast curve, and feature importance

---

## Results

| Model | MAE (£/MWh) | RMSE | sMAPE | Evaluation method |
|---|---|---|---|---|
| Naive (lag-48) | 36.34 | — | — | Direct |
| Seasonal naive (lag-336) | 29.40 | — | — | Direct |
| Rolling mean (48 SP) | 26.78 | — | — | Direct |
| **HGBR · 5yr + weather (production)** | **25.9** | **33.0** | **28.0%** | **Recursive day-ahead** |
| ~~HGBR (batch eval, leaky)~~ | ~~15.0~~ | — | — | Batch — inflated by `ssp_lag_1` |

Test period: 7 days (May 11–17 2026), 336 settlement periods.

**Why naive baselines are the right benchmark.** The naive lag-48 model predicts each half-hour slot tomorrow using the same slot's actual price today (`forecast[SP, D] = actual[SP, D−1]`). Seasonal naive lag-336 does the same using the same slot one week ago, additionally capturing the weekday/weekend demand pattern. These are hard to beat because strong intraday and day-to-day autocorrelation is a genuine feature of electricity markets — any model that fails to outperform them is merely rediscovering the lag structure. Beating the seasonal naive (£25.9 vs £29.40) with honest recursive evaluation confirms the model extracts signal beyond simple repetition.

> **Evaluation note:** batch prediction on pre-computed features is misleading for day-ahead forecasting because `ssp_lag_1` (the single most important feature, importance = 22.0) references the actual price from the previous 30-minute period — information that does not exist when the forecast is generated. The correct evaluation simulates the deployment loop: for each test day, short-lag features (`ssp_lag_1/2`, `ssp_roll_*_6`, `niv_lag_1`) are filled recursively from running model predictions, matching exactly how `forecast.py` runs. This reduces the apparent MAE gap between test and live performance.

Top features by permutation importance: `ssp_lag_1` (21.9), `net_imbalance_volume_lag_1` (1.7), `ssp_lag_2` (0.33), `sin_sp`/`cos_sp` (intra-day cycle), `solar_wm2_lag_1`, `wind_ms_lag_1`.

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
        features_5yr.csv        # 87,686 rows × 93 columns — full feature matrix

src/
    data/
        fetch_elexon.py         # Smart incremental Elexon ingest (concurrent, day-level)
        fetch_historical.py     # One-shot 5-year bulk fetch (ThreadPoolExecutor)
        fetch_weather.py        # Open-Meteo historical archive fetch
        build_dataset.py        # Cleaning, denoising, derived columns
    features/
        calendar_features.py    # Temporal + cyclic + annual harmonic features
        lag_features.py         # SSP/NIV lags, rolling stats, momentum diffs
        weather_features.py     # Weather lags, rolling stats, degree/ramp features
        build_features.py       # Full feature engineering pipeline
    models/
        evaluate.py             # MAE, RMSE, sMAPE, metrics reporting
        train_baseline.py       # Three lag-based baselines
        train_lgbm.py           # HGBR training with train/val/test split
        forecast.py             # Day-ahead recursive 48-period inference
    dashboard/
        streamlit_app.py        # Streamlit analytics + forecast dashboard

model_assets/
    hgbr_model.pkl              # Trained production model
    feature_cols.json           # Exact feature list used in training
    hgbr_feature_importance.csv # Permutation importance (val set)
    hgbr_metrics.json           # Test-set evaluation metrics
    test_predictions.csv        # Actuals vs predictions (May 11–17)
    next_day_forecast.csv       # Latest day-ahead forecast (48 SPs)
    forecasts/
        forecast_YYYY-MM-DD.csv # Archived daily forecasts for verification

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

### 3 — Train model

```bash
python src/models/train_lgbm.py
```

Outputs `model_assets/hgbr_model.pkl`, `feature_cols.json`, `hgbr_feature_importance.csv`, `hgbr_metrics.json`, and `test_predictions.csv`.

### 4 — Generate day-ahead forecast

```bash
python src/models/forecast.py
# Or for a specific date:
python src/models/forecast.py --date 2026-05-19
```

Fetches live weather from Open-Meteo and runs recursive 48-step inference. Saves `model_assets/next_day_forecast.csv`.

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
| Day-Ahead Forecast | 48-period price curve for the next day, with Min/Avg/Max metrics |
| KPI row | Latest SSP, average, min, max, spike count for the selected date range |
| SSP Time Series | Daily average SSP with configurable spike threshold overlay |
| Daily Heatmap | Settlement-period × date heat map — reveals intra-day and weekly patterns |
| Net Imbalance Volume | Daily average NIV bar chart (green = long, red = short) |
| Settlement Period Profile | Average 30-minute price profile across selected date range |
| Price Derivation Code | P vs N code breakdown — how often replacement price methodology triggers |
| Model Forecast vs Actual | Test-week time series, scatter, error histogram, daily error bars |
| Live Forecast Verification | Compares each archived day-ahead forecast against Elexon actuals once published — MAE, RMSE, sMAPE, error by settlement period, error histogram |
| Feature Importance | Top-20 features by permutation MAE reduction with uncertainty bars |
| Raw data | Filterable table with CSV download |

---

## Technical notes

**Leakage prevention (features)** — three contemporaneous columns excluded from features: `replacement_price` (corr = 0.9999 with SSP), `price_derivation_code_P` (corr = 0.69), `abs_imbalance_volume`. Failure to exclude these produced MAE = 0.72 — a near-perfect but fully leakage-driven result.

**Leakage prevention (evaluation)** — batch prediction on pre-computed test features inflates the reported MAE by ~£11/MWh because `ssp_lag_1` (importance = 22.0) uses the actual previous price from within the forecast window. The training script (`train_lgbm.py`) uses `evaluate_dayahead_recursive()`: for each test day it builds a running prediction buffer and overrides all short-lag features with model predictions, reproducing the true day-ahead information set.

**SSP = SBP** — confirmed by design, not a data error. ~50% of settlement periods use "P" (replacement price) methodology where SSP = SBP by definition. SBP is removed as redundant.

**Annual modulation** — statistically confirmed (p = 5.4 × 10⁻¹¹) but explains only R² = 2.9% of variance. Short-range autocorrelation (`ssp_lag_1`) dominates. Annual harmonic features are included but contribute marginal lift.

**Recursive forecasting** — SSP lags ≥ 48 always reference actual history (forecast horizon is 48 periods). Only lags 1 and 2 are filled recursively. NIV within the forecast window is proxied by the same settlement period on the previous day.

**Live verification loop** — every time `forecast.py` runs it archives the forecast to `model_assets/forecasts/forecast_YYYY-MM-DD.csv`. The dashboard's verification panel automatically detects which archived dates have Elexon actuals available and surfaces MAE, RMSE, sMAPE, and per-settlement-period errors for each verified day. Pending dates show a prompt to refresh once Elexon publishes the prices (typically next-day).

**Model choice** — `HistGradientBoostingRegressor` (sklearn) used in place of LightGBM; same histogram-based algorithm, no external OpenMP dependency. Swap back with `brew install libomp` + LightGBM as noted in `train_lgbm.py`.

---

## Motivation

UK electricity markets exhibit strong 30-minute periodicity, renewable intermittency, annual demand seasonality, and occasional extreme price spikes — a challenging forecasting environment that benefits from careful feature engineering over model complexity. This project demonstrates a realistic DS workflow: automated ingestion, denoising, systematic feature construction, leakage-aware model training, recursive inference, and interactive visualisation.

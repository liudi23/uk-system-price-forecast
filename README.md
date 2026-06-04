# UK Electricity System Price Forecasting Platform

Day-ahead forecasting of UK electricity System Sell Price (SSP) at 30-minute settlement period resolution. Built on public data from Elexon BMRS, Open-Meteo, Carbon Intensity API, ONS, and BMRS WINDFOR.

**Phase 3 — Level-Shape Decomposition · June 2026**

---

## Architecture

Phase 3 splits the day-ahead forecasting problem into two independent stages, eliminating recursive error propagation:

**Stage 1 — Level model** (quantile HGBR · P10/P50/P90)
- Predicts the daily mean SSP for the target date
- 81 features: SSP/NIV lags and rolling stats, calendar harmonics, day-ahead Open-Meteo weather, Carbon Intensity wind/gas generation lags, ONS CPIH index and YoY change
- Training targets deflated to real (current-money) terms using monthly CPI ratio
- 3-year rolling training window; test = last 2 days, val = 3 days before test

**Stage 2 — Shape model** (HGBR · P50)
- Predicts each SP's deviation from the daily mean
- 76 features: fixed-point lags ≥ 48 SPs only — leakage-free for all 48 forecast SPs simultaneously
- Includes ramp features (`ssp_ramp_48`, `niv_ramp_48`), same-SP cross-day statistics (`same_sp_mean_7d`, `same_sp_std_7d`), and wind/gas lag-48/lag-336
- At inference: `wind_pct_lag_48` replaced with BMRS WINDFOR/TSDF day-ahead wind % forecast

**Final forecast:** `ssp_q[X][sp] = level_P[X] + shape_deviation[sp]`

---

## Results

### Holdout performance (test = last 2 days)

| Model | MAE (£/MWh) | RMSE | sMAPE | Evaluation |
|---|---|---|---|---|
| Naive lag-48 | 36.34 | 47.73 | 41.6% | batch |
| Seasonal naive lag-336 | 29.40 | 38.29 | 34.7% | batch |
| Rolling mean (48 SP) | 26.78 | 33.07 | 28.5% | batch |
| Phase 2 recursive HGBR | 25.40 | 32.36 | 27.4% | 7-day holdout |
| **Phase 3 (current)** | **26.24** | **33.87** | **22.5%** | **2-day holdout** |

### Seasonal walk-forward CV (119 days · 4 folds)

Each fold retrains on data before the fold window — honest out-of-sample evaluation across all seasons.

| Season | Period | MAE | sMAPE | Level MAE | Shape Corr |
|---|---|---|---|---|---|
| Summer | Jul 2025 | £25.42 | 36.2% | £12.51 | 0.291 |
| Autumn | Oct 2025 | £29.58 | 47.5% | £14.94 | 0.489 |
| Winter | Dec 2025 | £20.97 | 33.1% | £8.43 | 0.387 |
| Spring | Apr 2026 | £33.67 | 51.3% | £16.53 | 0.452 |
| **Aggregate** | **119 days** | **£27.39** | **42.0%** | **£13.09** | **0.404** |

**Why sMAPE varies by season:** Winter is most predictable — demand-dominated, gas sets marginal cost consistently (Level MAE £8.43). Summer is level-easy but shape-hard (flat profiles). Autumn has the best shape correlation (demand peaks are pronounced). Spring is hardest — high renewable penetration and erratic dispatch.

---

## Data sources

| Source | Data | Update |
|---|---|---|
| Elexon BMRS | Settlement prices, NIV | Daily incremental |
| Open-Meteo | Historical weather + day-ahead forecast | Daily |
| Carbon Intensity API | Wind %, gas % generation mix (30-min) | Daily |
| ONS (series D7BT) | CPIH index 2015=100, monthly | Monthly |
| BMRS WINDFOR + TSDF | Day-ahead wind forecast (MW) + demand forecast | Daily at inference |

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
    level_features.py        Daily aggregation for Stage 1
  models/
    train_phase3.py          Two-stage training + seasonal walk-forward CV
    forecast_phase3.py       Non-recursive inference with WINDFOR substitution
    evaluate.py              MAE, RMSE, sMAPE, decomposition metrics
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

model_assets/
  level_q10/q50/q90.pkl      Stage 1 quantile models
  shape_q50.pkl              Stage 2 shape model
  level_feature_cols.json    81 level features
  shape_feature_cols.json    76 shape features
  phase3_metrics.json        Current test-set metrics
  test_predictions_phase3.csv  Test window actuals vs predictions
  walk_forward_predictions.csv 119-day seasonal CV predictions
  phase3_level_importance.csv  Level model Spearman importance
  phase3_shape_importance.csv  Shape model permutation importance
  next_day_forecast_phase3.csv Latest 48-SP day-ahead forecast
  forecasts/
    forecast_phase3_YYYY-MM-DD.csv  Daily archived forecasts (verification)

demo/
  demo_phase3.py             Four-panel demonstration figure
  phase3_demo.png            Pre-rendered output

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
python src/data/extend_dataset.py                # extend dataset_5yr.csv
python src/features/build_features.py \
    --input  data/processed/dataset_5yr.csv \
    --output data/processed/features_5yr.csv
python src/models/train_phase3.py                # retrain rolling window
python src/models/forecast_phase3.py            # tomorrow's forecast
```

Or step by step manually. The forecast script also fetches today's BMRS WINDFOR + TSDF day-ahead wind forecast automatically.

### Evaluation

```bash
# Standard holdout (test = last 2 days)
python src/models/train_phase3.py

# Seasonal walk-forward CV (4 × 30-day folds, no model overwrite)
python src/models/train_phase3.py --walk-forward

# Retrodiction for a specific date
python src/models/forecast_phase3.py --date 2026-06-01
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
| Day-Ahead Forecast | 48-SP P50 curve with P10/P90 band; Stage 1 level prediction |
| Forecast Verification | Archived Phase 3 forecasts vs Elexon actuals — MAE, RMSE, sMAPE, error by SP |
| KPIs | Latest SSP, daily average, min/max, spike count |
| SSP Time Series | Daily average SSP with configurable spike threshold |
| Daily Heatmap | SP × date heatmap — intra-day and weekly price patterns |
| Net Imbalance Volume | Daily average NIV (green = long system, red = short) |
| SP Profile | Average 30-min price profile across selected date range |
| Price Derivation Code | P vs N split — when replacement price methodology applies |
| Model Accuracy | Phase 3 test-window series, scatter, error distribution, decomposition metrics |
| Feature Importance | Top-20 for Stage 1 (Spearman correlation) and Stage 2 (permutation importance) |
| Raw Data | Filterable table with CSV export |

---

## Technical notes

**Leakage prevention** — shape features use only fixed-point lags ≥ 48 SPs. `shift(1).rolling(w)` window features contaminate SPs 2–48 with within-day actuals and are excluded entirely. Contemporaneous wind/gas (actual day-D values) are also excluded — only lag-48/lag-336 versions are used during training.

**WINDFOR substitution at inference** — the model's `wind_pct_lag_48` slot (trained on yesterday's CI actual as a proxy) is overridden with the genuine BMRS WINDFOR/TSDF day-ahead wind % at inference time. This is the same approach used for weather: actual weather during training, real day-ahead forecast at inference.

**CPI deflation** — training targets (daily level and shape deviations) are multiplied by `cpi_deflator = cpi_latest / cpi_month` before fitting, so the model learns in real (current-money) terms. The deflator is ≈1.0 at inference (tomorrow ≈ today in CPI terms).

**3-year training window** — drops data before `today − 1095 days` to reduce the influence of the 2022 gas-price crisis (SSP regularly exceeded £500/MWh), which is structurally unlike the current market regime.

**Elexon settlement data** — SSP = SBP for ~50% of periods (P-code replacement price methodology). Prices are finalised at Initial Settlement on D+1; no intraday actuals are available via the public BMRS API.

---

## Model progression

| Phase | Architecture | MAE | Note |
|---|---|---|---|
| Phase 1 | Batch HGBR | £15.01 | Leaky — evaluated on training data |
| Phase 2 | Recursive HGBR (honest) | £25.40 | Recursive SP-by-SP, 7-day holdout |
| Phase 3 | Level-Shape decomposition | £27.39 | Honest 4-season walk-forward CV |

The Phase 3 walk-forward MAE (£27.39) is measured without the leaky contemporaneous wind/gas features that inflated the earlier Phase 3 estimate of £25.15. The honest comparison with Phase 2 is approximate — both use small test windows in a volatile market.

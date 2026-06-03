# UK Electricity System Price Forecasting Platform

An end-to-end data science project for forecasting UK electricity system prices (SSP) at the settlement-period level (30-minute intervals). Built on public data from Elexon BMRS, Open-Meteo, Carbon Intensity API, and ONS.

**Phase 3 — Level-Shape Decomposition · CPI inflation adjustment · Seasonal walk-forward CV · June 2026**

---

## What it does

- **Ingests** 3 years of Elexon BMRS settlement data with smart incremental updates; training window rolls forward automatically each day
- **Fetches** UK weather history and day-ahead forecasts from Open-Meteo; wind/gas generation mix from the Carbon Intensity API; monthly UK CPIH index from ONS (series D7BT, 2015=100)
- **Engineers** features at two resolutions:
  - **81 daily-level features** for the level model — SSP/NIV lags and rolling stats, calendar harmonics, day-ahead weather, wind/gas daily lags, and CPI index + YoY inflation rate
  - **67 SP-level features** for the shape model — fixed-point lag-48+ only (lag-48/96/336 for SSP, NIV, weather, wind/gas), guaranteed leakage-free for all 48 forecast SPs
- **Trains** a two-stage decomposition model with no recursive error propagation:
  - **Stage 1 — Level model**: quantile HGBR (P10/P50/P90) predicts the day's average SSP; training targets deflated to real (current-money) terms using the monthly CPI ratio
  - **Stage 2 — Shape model**: HGBR predicts each SP's deviation from the daily mean using only fixed lag-48+ features; training targets also CPI-deflated
- **Evaluates** honestly with a non-recursive two-stage simulation on held-out data; seasonal walk-forward CV across 4 × 30-day windows spanning all seasons
- **Forecasts** tomorrow's 48 SPs without any within-day recursion — both models use only data available before the forecast day starts
- **Refreshes** automatically: the dashboard's **Refresh** button runs the full 8-step pipeline (fetch → build → retrain → forecast) in one click

---

## Results

### Current holdout (test = last 2 days, Jun 1–2 2026)

| Model | MAE (£/MWh) | RMSE | sMAPE | Evaluation |
|---|---|---|---|---|
| Naive (lag-48) | 36.34 | 47.73 | 41.6% | batch |
| Seasonal naive (lag-336) | 29.40 | 38.29 | 34.7% | batch |
| Rolling mean (48 SP) | 26.78 | 33.07 | 28.5% | batch |
| ~~HGBR Phase 1 (batch/leaky)~~ | ~~15.01~~ | ~~22.91~~ | ~~17.9%~~ | ~~leaky batch~~ |
| Quantile HGBR P50 · Phase 2 | 25.40 | 32.36 | 27.4% | 7-day holdout, honest |
| Phase 3 · before revision (Jun 1–2) | 27.28 | 35.43 | 23.8% | 2-day holdout, honest |
| **Phase 3 · current (Jun 1–2 2026)** | **25.54** | **30.74** | **22.0%** | **2-day holdout, honest** |

**Phase 3 decomposition diagnostics (Jun 1–2 2026):**

| Metric | Value | Meaning |
|---|---|---|
| Level MAE | £14.73/MWh/day | Error in predicting the day's average price (Stage 1) |
| Shape correlation | 0.326 | Mean Pearson r between predicted and actual intra-day profiles |
| Peak timing error | 1.5 SPs | Mean absolute offset between predicted and actual daily peak (±45 min) |

### Seasonal walk-forward CV (119 days across 4 seasons)

Separate model retrained per fold; each fold evaluated on held-out 30-day window it never saw during training.

| Season | Period | Days | MAE | sMAPE | Level MAE | Shape Corr |
|---|---|---|---|---|---|---|
| Summer | Jul 2025 | 30 | £23.34 | 34.9% | £9.73 | 0.320 |
| Autumn | Oct 2025 | 29 | £25.30 | 46.0% | £14.25 | 0.550 |
| Winter | Dec 2025 | 30 | £19.56 | 30.7% | £7.62 | 0.386 |
| Spring | Apr 2026 | 30 | £32.40 | 52.3% | £17.07 | 0.464 |
| **Aggregate** | | **119** | **£25.15** | **40.9%** | **£12.15** | **0.429** |

Walk-forward aggregate MAE £25.15 matches Phase 2 (£25.40) on a cross-season basis.

> **Why sMAPE varies by season:** Summer and winter price levels are similar (~£75–83/MWh mean) but market drivers differ. Winter is demand-dominated and most predictable (level MAE £7.62) because gas sets marginal cost consistently. Summer is level-predictable but shape-hard (flat profiles, shallow peaks). Autumn has the best shape correlation (0.550) because pronounced demand peaks give the shape model a clear signal. Spring is hardest (level MAE £17.07, sMAPE 52.3%) due to high renewable penetration and erratic dispatch order.

---

## What changed in this revision

| Change | Detail |
|---|---|
| **CPI inflation adjustment** | Monthly UK CPIH index (ONS D7BT) fetched automatically. Training targets deflated to real (current-money) terms via `cpi_deflator = cpi_latest / cpi_month`. Prevents older lower-price periods from biasing level predictions downward. `cpi_index` and `cpi_yoy` added as model features. |
| **3-year training window** | Hard cutoff at `today − 3 years`. Reduces energy-crisis contamination (2022 gas price spike) while retaining sufficient regime diversity. |
| **Relative train/val/test split** | Test = last 2 days (yesterday + day before), Val = 3 days before test, Train = remaining 3-year window. Split rolls forward automatically on each refresh. |
| **Dashboard one-click pipeline** | Refresh button runs: fetch prices → fetch weather → fetch generation mix → fetch CPI → rebuild dataset → rebuild features → retrain models → run forecast. |
| **Seasonal walk-forward CV** | `python src/models/train_phase3.py --walk-forward` evaluates across 4 × 30-day seasonal folds without overwriting production models. |

> **CPI and electricity prices:** General consumer inflation (CPIH) is not the primary driver of electricity prices, which are dominated by gas commodity markets, renewable capacity, and balancing charges. CPI features act as a macro-environment signal to help the level model contextualise whether current prices are elevated or suppressed relative to the general price level — particularly useful across the 2022–2026 period where energy prices diverged sharply from CPI.

---

## Folder structure

```
data/
    raw/
        system_prices.csv          # Elexon BMRS — SSP, NIV, price derivation code
        weather_uk.csv             # Open-Meteo — 30-min UK weather (3 locations, weighted)
        generation_mix.csv         # Carbon Intensity API — wind %, gas % per SP
        cpi_uk.csv                 # ONS CPIH index D7BT (2015=100), monthly
    processed/
        dataset_5yr.csv            # Cleaned + denoised (Tukey outer-fence winsorisation)
        features_5yr.csv           # SP-level feature matrix (117 columns, incl. CPI/wind/gas)

src/
    data/
        fetch_elexon.py            # Smart incremental Elexon ingest (concurrent, day-level)
        fetch_historical.py        # One-shot bulk fetch (ThreadPoolExecutor)
        fetch_weather.py           # Open-Meteo historical archive fetch
        fetch_generation.py        # Carbon Intensity API — wind %, gas % generation mix
        fetch_cpi.py               # ONS CPIH index D7BT — monthly UK inflation
        build_dataset.py           # Cleaning, denoising, derived columns
    features/
        calendar_features.py       # Temporal + cyclic + annual harmonic features
        lag_features.py            # SSP/NIV lags, rolling stats, spike memory, NIV extremes
        weather_features.py        # Weather lags, rolling stats, degree/ramp features
        build_features.py          # Full SP-level feature pipeline (incl. CPI merge)
        level_features.py          # Daily-level aggregation for Stage 1 level model
    models/
        evaluate.py                # MAE, RMSE, sMAPE + decomposition metrics
        train_baseline.py          # Three lag-based baselines
        train_lgbm.py              # Phase 2: quantile HGBR + spike classifier (recursive)
        train_phase3.py            # Phase 3: two-stage training, seasonal walk-forward CV
        forecast.py                # Phase 2: recursive day-ahead inference (P10/P50/P90)
        forecast_phase3.py         # Phase 3: non-recursive two-stage inference
    dashboard/
        streamlit_app.py           # Streamlit analytics + forecast dashboard

model_assets/
    # Phase 3 models
    level_q10.pkl                  # Stage 1 level model — P10
    level_q50.pkl                  # Stage 1 level model — P50
    level_q90.pkl                  # Stage 1 level model — P90
    shape_q50.pkl                  # Stage 2 shape model — P50 deviation forecast
    level_feature_cols.json        # 81 daily-level features for Stage 1
    shape_feature_cols.json        # 67 SP-level lag-48+ features for Stage 2
    phase3_metrics.json            # Phase 3 test-set metrics + decomposition diagnostics
    test_predictions_phase3.csv    # Actuals vs predictions on test window
    walk_forward_predictions.csv   # 119-day seasonal walk-forward evaluation
    next_day_forecast_phase3.csv   # Latest Phase 3 day-ahead forecast (48 SPs)
    forecasts/
        forecast_phase3_YYYY-MM-DD.csv  # Phase 3 archived daily forecasts (verified panel)

demo/
    demo_phase3.py                 # Four-panel demonstration figure
    phase3_demo.png                # Pre-rendered demo output

Project_Brief/
    phase-3-summary.md             # Architecture, innovation, and seasonal analysis
    Phase2_Plan.md
    Phase3_Plan.md

reports/
    annual_modulation_analysis.md  # Statistical analysis of annual price seasonality
```

---

## Setup

```bash
git clone <repo>
cd uk-system-price-forecast
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running the pipeline

### 1 — Fetch raw data

```bash
# Elexon BMRS prices (incremental append)
python src/data/fetch_elexon.py --append

# Open-Meteo weather
python src/data/fetch_weather.py

# Carbon Intensity generation mix (wind %, gas %)
python src/data/fetch_generation.py --append

# UK CPIH inflation index (ONS D7BT)
python src/data/fetch_cpi.py
```

### 2 — Build dataset and features

```bash
python src/data/build_dataset.py \
    --raw data/raw/system_prices.csv \
    --out data/processed/dataset_5yr.csv

python src/features/build_features.py \
    --input  data/processed/dataset_5yr.csv \
    --output data/processed/features_5yr.csv
```

### 3 — Train Phase 3 model

```bash
# Standard holdout (test = last 2 days, 3-year training window)
python src/models/train_phase3.py

# Seasonal walk-forward CV (4 folds × 30 days, does not overwrite models)
python src/models/train_phase3.py --walk-forward
```

Outputs: `level_q10/q50/q90.pkl`, `shape_q50.pkl`, `level/shape_feature_cols.json`, `phase3_metrics.json`, `test_predictions_phase3.csv`.

### 4 — Generate day-ahead forecast

```bash
# Phase 3 — non-recursive, recommended
python src/models/forecast_phase3.py

# Phase 3 — retrodiction for a specific past date
python src/models/forecast_phase3.py --date 2026-06-01
```

Phase 3 fetches live Open-Meteo weather and last-9-days Carbon Intensity data at inference time. Saves `next_day_forecast_phase3.csv` and archives to `model_assets/forecasts/`.

### 5 — Launch the dashboard

```bash
.venv/bin/streamlit run src/dashboard/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). Click **Refresh Data & Run Forecast** in the sidebar to run the full 8-step pipeline automatically.

---

## Dashboard sections

| Section | What it shows |
|---|---|
| Day-Ahead Forecast | 48-period P50 curve with P10/P90 band; predicted daily level from Stage 1 |
| KPI row | Latest SSP, average, min, max, spike count for the selected date range |
| SSP Time Series | Daily average SSP with spike threshold overlay |
| Daily Heatmap | Settlement-period × date heatmap — intra-day and weekly patterns |
| Net Imbalance Volume | Daily average NIV bar chart |
| Settlement Period Profile | Average 30-min price profile across selected date range |
| Price Derivation Code | P vs N code breakdown |
| Model Forecast vs Actual | Phase 3 test-window series, scatter, error histogram; level MAE, shape correlation, peak timing |
| Live Forecast Verification | Archived Phase 3 forecasts vs Elexon actuals — MAE, RMSE, sMAPE, per-SP error |
| Feature Importance | Top-20 Phase 2 features by permutation MAE reduction |
| Raw data | Filterable table with CSV download |

---

## Technical notes

**Level-shape decomposition** — prices on any day decompose into (1) a daily level (how expensive the day is on average) and (2) an intra-day shape (how prices vary across 48 SPs relative to the level). Both are predictable from different information sets: level from daily-aggregated history, shape from fixed lag-48+ SP features. Separating them eliminates recursive error propagation entirely.

**CPI deflation of training targets** — `cpi_deflator = cpi_latest / cpi_month` (>1 for historical rows, ≈1 for recent rows). Both `ssp_raw` (level target) and `ssp_shape_target` (shape target) are multiplied by this ratio before training, so the model learns real-terms relationships. At inference, the deflator is ≈1.0 (tomorrow ≈ today in CPI terms), so no reverse transform is needed on forecast output. The `cpi_deflator` column is excluded from model features.

**3-year training window** — drops data before `today − 1095 days`. Reduces the influence of the 2022 Russia-Ukraine gas price spike (SSP regularly exceeded £500/MWh), which is structurally unlike the current market regime. The CPI deflator already partially corrects for nominal inflation, but the 3-year cutoff handles non-inflation structural breaks.

**Leakage prevention (shape features)** — `shift(1).rolling(w)` features contain within-day actual prices for SPs 2–48 of the forecast day. Phase 3 uses only fixed-point lags (≥ 48 SPs), safe for every SP simultaneously. An earlier bug silently included 32 leaky rolling features; the fix dropped apparent MAE from £22.18 → £25.39 (the ~12% gain was entirely leakage-driven).

**Leakage prevention (level features)** — contemporaneous daily aggregates (`ssp_daily_mean`, `niv_daily_mean`, etc.) are excluded. All level features reference days strictly before D. Same-day weather (temp, wind, solar) is treated as a day-ahead forecast input — available from Open-Meteo before day D starts.

**Level model features (81)** — daily SSP/NIV lags (1/2/7/14/28d) and rolling stats (7/14/28d), spike count lags, calendar harmonics, day-ahead weather for target day, weather lag-1d/7d from history, wind/gas daily lags and 7d rolling mean, `cpi_index` (CPIH level), `cpi_yoy` (12-month change %).

**Shape model features (67)** — `ssp_lag_48/96/336`, `ssp_raw_lag_48/96/336`, `is_spike_lag_48/336`, `is_negative_lag_48/336`, `niv_lag_48/336`, weather lag-48/336, `wind_pct_lag_48/336`, `gas_pct_lag_48/336`, `heating_degree`, `cooling_degree`, SP-position calendar features, and daily-level lag features merged from Stage 1 (`ssp_daily_mean_lag1d`, `ssp_lag48_deviation`, etc.). All SP-level rolling features excluded.

**Uncertainty bands** — three separate quantile HGBR models (P10/P50/P90) at the level stage. The P10/P90 SP-level bands are formed by applying the P50 shape deviation to level P10 and P90 respectively — interval width is constant within a day.

**Live verification loop** — `forecast_phase3.py` archives every forecast to `model_assets/forecasts/forecast_phase3_YYYY-MM-DD.csv`. The dashboard verification panel compares these against Elexon actuals for all dates where both exist (currently May 18 – Jun 2 2026).

**Auto refresh** — the dashboard's Refresh button runs in sequence: `fetch_elexon → fetch_weather → fetch_generation → fetch_cpi → build_dataset → build_features → train_phase3 → forecast_phase3`. Each step shows a spinner; errors surface as sidebar warnings without aborting subsequent steps.

---

## Motivation

UK electricity markets exhibit strong 30-minute periodicity, renewable intermittency, annual demand seasonality, and occasional extreme price spikes — a challenging forecasting environment that benefits from careful feature engineering over model complexity. This project demonstrates a realistic DS workflow: automated ingestion, denoising, systematic feature construction, leakage-aware training and evaluation, two-stage decomposition without recursive error propagation, exogenous generation mix and inflation inputs, and interactive visualisation. The progression from Phase 1 (leaky batch, MAE £15.0) → Phase 2 (honest recursive, £25.4) → Phase 3 (non-recursive decomposition, CPI-adjusted, walk-forward MAE £25.15 across 4 seasons) illustrates both the pitfalls of naive evaluation and the value of architectural choices grounded in the causal structure of the problem.

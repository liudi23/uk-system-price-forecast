# UK Electricity System Price Forecasting — Phase 2 Plan

**Status:** Planning
**Follows:** Phase 1 MVP (shipped May 2026)

---

## Phase 1 recap

Phase 1 delivered an end-to-end forecasting platform:

- 5-year Elexon BMRS data ingestion with smart incremental updates
- 76-feature engineering pipeline (price lags, rolling stats, calendar/annual harmonics, UK weather)
- Production HGBR model: MAE £15.01/MWh · sMAPE 17.9% on the May 11–17 2026 test week
- Day-ahead 48-period recursive forecast with live Open-Meteo weather
- Streamlit dashboard: historical analytics, forecast panel, live comparison, verification, feature importance

A known limitation surfaced at the end of Phase 1: the **live comparison panel** cannot show today's actual prices during the day because Elexon publishes system prices at **Initial Settlement (D+1)** — the next business day. The public BMRS API does not expose intraday finalised prices.

---

## Phase 2 objectives

### 1. Near-real-time price comparison (high priority)

**Problem:** The "Live: Today's Forecast vs Actual SSP" panel currently shows no actuals until the following morning. This limits the dashboard's value as a live monitoring tool.

**Proposed solution:** Integrate Elexon's **indicative settlement prices** from the BMRS near-real-time stream endpoint. These are preliminary imbalance prices published within a few hours of each settlement period ending — before the formal Initial Settlement run.

Key considerations:
- Indicative prices are **not the same as finalised system prices**. They can differ from final SSP/SBP, particularly for P-code (replacement price) periods.
- The dashboard must clearly label them as "indicative" to avoid misleading interpretation.
- Suggested display: show indicative actuals in a **lighter shade** with a tooltip warning; overlay finalised D+1 prices the following day as the authoritative comparison.
- Relevant BMRS endpoint to investigate: `/balancing/settlement/indicative-system-prices/{date}` or the equivalent streaming/polling endpoint.

**Acceptance criteria:**
- Indicative prices appear in the live panel within 2–3 hours of each settlement period
- Chart clearly distinguishes indicative (preliminary) from finalised (D+1) prices
- Running MAE shown separately for indicative vs finalised to track the gap

---

### 2. Probabilistic forecasting

**Problem:** The current model outputs a single point forecast with no uncertainty estimate. Users (e.g. traders, flexibility operators) need to know how confident the model is, especially around spike periods.

**Proposed approaches:**
- **Quantile regression:** Train HGBR with `loss="quantile"` for the 10th, 50th, and 90th percentiles — minimal code change, produces a prediction interval band
- **Conformal prediction:** Wrap the existing model with a split-conformal interval that gives calibrated coverage guarantees on held-out data
- **Ensemble:** Train 10–20 models on bootstrapped training windows; use spread as uncertainty estimate

**Dashboard addition:** Shaded confidence band on the forecast chart (e.g. P10–P90 in light orange), with a toggle to show/hide.

---

### 3. Spike classification

**Problem:** The model's MAE on spike periods is significantly higher than on normal periods. Price spikes (SSP > £200/MWh) are the highest-value periods to predict correctly for trading and risk purposes.

**Proposed approach:**
- Train a separate binary classifier (e.g. gradient boosting) to predict `is_spike` (SSP > threshold) for each settlement period
- Use spike probability as an additional feature in the main regression model
- Add a "spike risk" indicator to the dashboard (colour-coded settlement period heatmap)

---

### 4. Model retraining pipeline

**Problem:** The current model is a static artefact trained once on May 2021–2026 data. As new data accumulates, model accuracy will drift.

**Proposed approach:**
- Automate the full pipeline: `fetch_elexon → fetch_weather → build_dataset → build_features → train_lgbm → forecast`
- Schedule weekly retraining (e.g. via cron or GitHub Actions) on a rolling 5-year window
- Version model artefacts by training date in `model_assets/`
- Track test-set MAE over time in the verification panel to detect performance drift

---

### 5. Extended forecast horizon

**Problem:** The current model forecasts only the next 48 settlement periods (24 hours). Day-ahead markets and flexibility dispatch often require 2–7 day outlooks.

**Proposed approach:**
- Extend recursive inference to 96 periods (48 hours) or 336 periods (7 days)
- Accuracy will degrade beyond 24 hours as SSP autocorrelation decays — quantify this degradation and communicate it in the UI
- Use Open-Meteo's 7-day forecast API (already integrated) for the extended weather horizon

---

### 6. FastAPI inference backend

**Problem:** The current system runs inference as a local script. To share forecasts with colleagues or integrate with other systems, a REST API is needed.

**Proposed endpoints:**
- `GET /forecast/today` — returns today's 48-period forecast as JSON
- `GET /forecast/{date}` — returns archived forecast for a given date
- `GET /actuals/{date}` — returns finalised Elexon prices for a given date
- `GET /metrics/live` — returns running MAE for today's settled periods

**Stack:** FastAPI + uvicorn, Dockerised for easy deployment.

---

## Suggested Phase 2 delivery order

| Priority | Item | Effort | Impact |
|---|---|---|---|
| 1 | Near-real-time indicative prices | Medium | High — fixes the main Phase 1 gap |
| 2 | Quantile/probabilistic forecast | Low–Medium | High — adds decision-making value |
| 3 | Automated retraining pipeline | Medium | High — keeps the model current |
| 4 | Spike classifier | Medium | Medium — targeted improvement |
| 5 | Extended forecast horizon (48h) | Low | Medium — natural extension |
| 6 | FastAPI backend | Medium | Medium — enables sharing/integration |

---

## Open questions

- What is the exact latency and reliability of the BMRS indicative settlement price endpoint? Needs empirical testing.
- For the probabilistic forecast: which coverage level is most useful for the intended use case (trading desk vs flexibility operator vs grid planner)?
- Should the retraining pipeline retain old model versions for backtesting, or always overwrite with the latest?
- Deployment target for Phase 2: local only, cloud VM, or containerised service?

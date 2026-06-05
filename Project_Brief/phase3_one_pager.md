# UK Electricity System Price Forecasting — Phase 3 Deliverable

**Project:** Day-ahead SSP forecasting at 30-minute settlement period resolution  
**Model:** Phase 3 Level-Shape Decomposition — June 2026  
**Forecast horizon:** H+1 (today, 48 SPs) and H+2 (tomorrow, 48 SPs)

---

## Problem

UK electricity System Sell Price (SSP) is highly volatile — driven by renewable intermittency, gas prices, demand seasonality, and market mechanics. Accurate day-ahead forecasts reduce balancing costs, support trading decisions, and flag high-risk periods.

---

## Solution: Two-Stage Level-Shape Decomposition

Rather than predicting each 30-minute price sequentially (Phase 2's recursive approach, where errors compound), Phase 3 separates the forecasting problem into two independent questions:

**Stage 1 — What will today's average price be?** (Level model)  
Quantile HGBR (P10/P50/P90) trained on 85 daily features: price/NIV lags, day-ahead weather, wind and gas generation mix, UK CPIH inflation index, negative-price regime classifier.

**Stage 2 — How will prices vary across the 48 half-hours?** (Shape model)  
HGBR trained on 76 settlement-period features with fixed lags ≥ 48 SPs — leakage-free for every SP in the forecast window. At inference, real day-ahead signals replace lag proxies: BMRS WINDFOR/TSDF (SP-level wind %) and Open-Meteo (SP-level solar irradiance).

**Final forecast per SP:** `Level_P[X] + Shape_deviation[SP]`  
P10/P90 bands propagate level uncertainty uniformly across all 48 SPs.

---

## Data Sources

| Source | Data | Frequency |
|---|---|---|
| Elexon BMRS | Settlement prices (SSP), Net Imbalance Volume | 30-min, D+1 settlement |
| Open-Meteo | Historical weather + day-ahead forecast (3 UK sites) | 30-min |
| Carbon Intensity API | Wind %, gas % in generation mix | 30-min actuals |
| BMRS WINDFOR + TSDF | Day-ahead wind generation and demand forecast | Per SP, D-ahead |
| ONS (series D7BT) | UK CPIH index (2015=100) | Monthly |

Training: 3-year rolling window with CPI deflation of targets (real-money terms).

---

## Performance

### Seasonal walk-forward cross-validation (119 days, 4 folds — honest out-of-sample)

| Season | Period | MAE | Level MAE | Shape Corr |
|---|---|---|---|---|
| Summer | Jul 2025 | £25.4/MWh | £12.5/day | 0.29 |
| Autumn | Oct 2025 | £29.6/MWh | £14.9/day | 0.49 |
| Winter | Dec 2025 | £21.0/MWh | £8.4/day | 0.39 |
| Spring | Apr 2026 | £33.7/MWh | £16.5/day | 0.45 |
| **Aggregate** | | **£27.4/MWh** | **£13.1/day** | **0.43** |

Naive lag-48 baseline: £36.3/MWh · Seasonal naive: £29.4/MWh · Phase 2 recursive: £25.4/MWh

> **7-day holdout (May 29 – Jun 4 2026) shows MAE £31.6** due to the Jun 4 extreme renewable-oversupply event (10 negative-price SPs, −£70 midday). Excluding this event, performance is consistent with walk-forward results.

---

## Key Technical Innovations vs Phase 2

| Feature | Phase 2 | Phase 3 |
|---|---|---|
| Forecast structure | Recursive — SP1 feeds SP2, errors compound | Non-recursive — level + shape, no error propagation |
| Forecast horizon | H+1 only | H+1 (today) + H+2 (tomorrow) |
| Wind signal | None | BMRS WINDFOR day-ahead forecast (SP-level) at inference |
| Solar signal | Daily aggregate lag | Open-Meteo SP-level day-ahead forecast at inference |
| Inflation | None | CPI-deflated training targets (ONS D7BT) |
| Negative-price detection | None | Binary HGBR classifier → `neg_price_risk_prob` as level feature |
| Training window | Full history | 3-year rolling (reduces 2022 energy-crisis bias) |
| Evaluation | 7-day holdout | 4-season walk-forward CV (119 days) |

---

## Model Parameters

All HGBR models share: `learning_rate=0.05`, `max_leaf_nodes=31`, `min_samples_leaf=10`, `l2_regularization=0.1`, `early_stopping=True (patience=50)`, `max_bins=255`. Loss = quantile pinball at target quantile.

---

## Automated Pipeline

One-click **Refresh** in the Streamlit dashboard runs:  
fetch prices → fetch weather → fetch generation → fetch CPI → extend dataset → rebuild features → retrain → H+1 + H+2 forecast

Dashboard panels: H+1 forecast (P10/P50/P90), H+2 forecast, verification vs actuals, price analytics, model accuracy, feature importance.

---

## Known Limitations and Phase 4 Roadmap

The Jun 4 2026 event (renewable oversupply, −£70 prices) illustrates the hard limit of autoregressive models: when the prior day gives no negative-price signal, no lag feature predicts the regime shift. Recommended Phase 4 priorities:

1. **SP-level demand forecast** — BMRS TSDF as a shape feature (day-ahead dispatch signal)
2. **Specialist model for negative-price days** — separate regime for high-wind/low-demand events
3. **H+2 feature enrichment** — add lag-96 NIV, weather, and generation mix to close the gap with H+1
4. **Shape model quantiles** — P10/P90 on shape deviations for SP-level calibrated intervals
5. **Hyperparameter tuning** — separate optimisation for level vs shape models

Full root cause analysis: `reports/phase3_root_cause_analysis.md`

# Phase 3 Root Cause Analysis — Jun 4 2026 Forecast Failure

**Date of incident:** 2026-06-04  
**Observed MAE:** £65.06/MWh (verification panel) / £48.51/MWh (2-day test)  
**Baseline MAE (4-season walk-forward):** £27.39/MWh  
**Shape correlation:** 0.269 (well below 0.40 aggregate)  
**Peak timing error:** 13 SPs (~6.5 hours)  

---

## 1. What happened on Jun 4

Jun 4 2026 was a severe **renewable oversupply event**:

| Metric | Value |
|---|---|
| Daily mean SSP | £36.8/MWh — 4.4th percentile of training distribution |
| Min SSP | −£70.2/MWh (SP29, ~14:00) |
| Negative-price periods | 10 out of 48 SPs |
| Wind % of generation | 52.3% mean, 41–63% range |
| Gas % of generation | 9.7% mean (near-minimum) |
| Model level prediction | £90–100/MWh (£55+ overestimate) |

The prior day (Jun 3) had zero negative prices and averaged £108/MWh. The model had no signal that Jun 4 would be radically different.

**Intraday structure of Jun 4:**
- 00:00–05:00: ~£50 (normal overnight)
- 05:00–10:00: sharp morning ramp to £100 (normal)
- 10:00–15:30: collapse to −£70 (renewable saturation)
- 15:30–21:00: recovery to £100–134 (evening demand)
- 21:00–24:00: back to £50–55

The model predicted a flat curve around £90–100 throughout the day, completely missing the midday crash.

---

## 2. Root cause: not a model bug — a structural forecasting challenge

### 2.1 Feature signals available to the model (day D-1 → D)

| Feature | Value for Jun 4 | Why insufficient |
|---|---|---|
| `ssp_lag_48` (Jun 3 same SP) | £26.9–£160 range | Jun 3 volatile but all positive |
| `is_negative_lag_48` | 0 | Jun 3 had no negative prices |
| `wind_pct_lag_48` | 44.3% (Jun 3 actuals) | High, but model couldn't anticipate 52% next day |
| `wind_pct_lag_336` | prior week's wind | Weaker wind last week |
| `same_sp_mean_7d` | £60–136 | All recent days positive |
| `same_sp_std_7d` | 12–73 | High volatility visible but not directional |
| `is_negative_lag_336` | 0 | Last week also no negatives |

**The fundamental gap:** All available signals pointed to a normal expensive-ish summer day. The renewable oversupply crash is driven by the coincidence of high wind + high solar + low demand — a combination that can only be anticipated with genuine day-ahead generation forecasts, not autoregressive price lags.

### 2.2 Level model failure

The level model predicted £90–103/MWh against an actual daily mean of £36.8/MWh — a level error of £55–65/MWh. This single error propagates uniformly to all 48 SPs, accounting for most of the headline MAE.

The level model's inputs were reasonable given prior-day data: Jun 3 averaged £108, the 7-day rolling mean was £90–110, NIV was near-normal. Nothing in the lag features signalled that tomorrow's mean would be at the 4th percentile.

### 2.3 Shape model failure (midday collapse)

The shape model predicts SP deviations from the predicted daily mean. Even if the level were correct (£37), the shape would need to predict −£107 deviation at SP29 (midday trough). The prior-day shape (Jun 3) showed a normal morning peak — no hint of midday collapse. The `same_sp_mean_7d` for midday SPs was £100+, directly contradicting what would happen.

**Peak timing error of 13 SPs (~6.5 hours):** The model correctly anticipated an afternoon-to-evening demand peak (as is typical), but the actual peak was early-evening SP43. The midday collapse pushed the timing completely off.

### 2.4 Data leakage audit — no leakage found

Full audit of 76 shape features conducted:
- All lags ≥ 48 SPs — no within-day contamination
- No SP-level rolling windows included
- Contemporaneous `wind_pct` and `gas_pct` correctly excluded (removed in prior session)
- CPI deflator correctly excluded from model features

The poor performance is genuine — it is not inflated by leakage.

### 2.5 Data quality — not an issue

- Negative prices well represented in training: 3,094 negative-price SPs in 3-year window (5.9%)
- Training negative price range: −£185 to £0 — Jun 4's −£70 within historical range
- No Tukey fence violations — Jun 4 prices within bounds (fence: lower = −£156.6, upper = £353.7)
- Generation mix data complete for training period

---

## 3. Classification of failure

| Cause | Verdict |
|---|---|
| Data leakage | ✗ Not present |
| Data quality | ✗ Not an issue |
| Hyperparameter tuning | ✗ Not the cause |
| Model architecture | ✗ Correct for the problem |
| **Missing day-ahead exogenous signals** | ✓ **Primary gap** |
| **Autoregressive model limit on regime-change days** | ✓ **Fundamental constraint** |

---

## 4. Guidance for Phase 4

### 4.1 Most impactful additions

**A) SP-level day-ahead solar forecast**  
Solar irradiance (Open-Meteo) is currently a weather-level feature aggregated over the day. What's needed is the hourly/SP-level solar forecast to tell the shape model exactly when solar saturation will occur. Combined with WINDFOR (already integrated), this would give the model the two key drivers of midday price crashes.

Options:
- Open-Meteo already returns hourly `shortwave_radiation` — extract SP-level values rather than daily aggregate
- Map to settlement periods (same UTC→UK conversion as WINDFOR)
- Add `solar_forecast_wm2[sp]` as a direct shape feature (day-ahead, no leakage)

**B) TSDF day-ahead demand forecast at SP level**  
BMRS TSDF boundary='N' already fetched for WINDFOR normalisation. The demand forecast at SP level tells the model whether demand will be low (increasing oversupply risk). Add `demand_forecast_mw[sp]` as a shape feature. Currently available at inference; needs proxy for training (use lagged demand proxies).

**C) Negative-price regime classifier (daily-level)**  
A binary "negative-price risk" flag for the target day, fed into the level model. This could be a simple threshold rule or a trained classifier:
- Inputs: WINDFOR daily mean, solar forecast daily mean, TSDF daily mean, `neg_count_roll_7d`
- Threshold: if `wind_pct_forecast_daily > 45% AND solar_forecast_daily > X AND demand_forecast < Y → flag`
- Supplying this as a level feature would allow the level model to predict near-zero or negative daily means on high-risk days

**D) Separate model for negative-price days**  
Consider a two-stage approach: first classify whether tomorrow is a "negative-price risk day" (as above), then use a specialist model for those days. Negative-price days have fundamentally different price formation — renewable curtailment economics, not gas marginal cost.

### 4.2 Architecture considerations

**Level model — regime detection:**
The current level model uses a single HGBR across all price regimes. Negative-price days (4th percentile) are structurally different from gas-dominated days (normal regime). Consider adding quantile regression targets at P05/P10 that explicitly represent tail downside risk, or a separate "low-price day" specialist.

**Shape model — intraday solar signal:**
`solar_wm2_lag_48` is the top-ranked shape feature (permutation importance 0.11). This confirms solar drives the intraday profile. Moving from lag-48 (yesterday's solar) to a genuine day-ahead solar forecast would be the shape model equivalent of what WINDFOR does for wind.

**Negative-price persistence:**
Add `neg_count_roll_7d` and `is_negative_lag_48` to the level model daily features (currently only in shape). A week with multiple negative-price days raises the probability of another.

### 4.3 Features to add in Phase 4

| Feature | Source | Lag constraint | Target stage |
|---|---|---|---|
| `solar_forecast_wm2[sp]` | Open-Meteo hourly | Day-ahead, no leakage | Shape |
| `demand_forecast_mw[sp]` | BMRS TSDF boundary='N' | Day-ahead, no leakage | Shape + Level |
| `wind_pct_forecast_daily` | BMRS WINDFOR (daily agg) | Day-ahead, no leakage | Level |
| `neg_price_risk_flag` | Rule/classifier | Day-ahead, no leakage | Level |
| `neg_count_roll_7d` | CI actuals | Daily lag-1d | Level |

### 4.4 Evaluation framework for Phase 4

The Jun 4 failure highlights that aggregate MAE is misleading — a handful of extreme events dominate. Phase 4 should track:
- **MAE split by price regime**: normal (£50–200), cheap (<£50), extreme negative (<0)
- **Negative-price day recall**: of all days with ≥5 negative SPs, what fraction does the model classify correctly?
- **Conditional shape correlation**: shape correlation specifically on high-wind, low-demand days
- **Ensemble/quantile coverage**: does the P10 band capture 90% of outcomes?

---

## 5. Summary

Jun 4 2026 was a legitimate hard case — renewable oversupply drove extreme negative prices with no signal in prior-day lag features. The model performed as designed but the problem exceeded what autoregressive features can capture. The path to improvement is **day-ahead renewable generation forecasts at SP resolution** (solar + wind + demand), which transform the shape model from "what did yesterday look like?" to "what will today's renewable dispatch look like?".

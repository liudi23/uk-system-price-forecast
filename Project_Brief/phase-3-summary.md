# Phase 3 Summary: Two-Stage Level-Shape Decomposition

## Overview

Phase 3 introduces a two-stage decomposition architecture for forecasting UK electricity System Sell Price (SSP) at 30-minute settlement period (SP) resolution, 48 SPs per day, one day ahead. The core design principle separates the forecasting problem into two independent questions: what is the expected daily average price (level), and how does price deviate from that average across intra-day settlement periods (shape). This decomposition directly addresses the primary failure mode of Phase 2 — recursive error propagation — while also eliminating a leakage bug that had silently inflated earlier accuracy estimates.

## Architecture

### Stage 1: Level Model

The level model predicts the daily mean SSP for target date D using a quantile Histogram-based Gradient Boosting Regressor (HGBR) producing P10, P50, and P90 estimates. It draws on 79 features organised into six groups:

- **Calendar (13):** day-of-week, month, quarter, and sin/cos cyclical encodings
- **SSP lags and rolling statistics (30):** lag-1d, 2d, 7d, 14d, 28d daily means, plus rolling 7/14/28-day mean, standard deviation, minimum, and maximum
- **Net Imbalance Volume — NIV (6):** lag-1d and lag-7d, rolling 7/14-day mean and standard deviation
- **Spike counts (3):** lag-1d, lag-7d, and rolling 7-day sum
- **Weather (20):** day-ahead Open-Meteo temperature, wind speed, solar irradiance, and precipitation for the target day, plus lag-1d and lag-7d historical values
- **Generation mix (6):** Carbon Intensity API wind percentage and gas percentage — lag-1d, lag-7d, and rolling 7-day mean

Crucially, no features reference data from day D itself. All lags are relative to fully observed prior days. There is no recursion.

### Stage 2: Shape Model

The shape model predicts the intra-day deviation from the daily mean for each of the 48 SPs using an HGBR P50 model trained on 65 features. Every feature uses fixed-point lags of at least 48 SPs, ensuring the forecast is leakage-free across the entire target day. Wind percentage and gas percentage enter as lag-48 and lag-336 per SP.

A key exclusion is SP-level rolling features constructed as `shift(1).rolling(w)`. These contaminate SPs 2 through 48 with within-day actual prices from the target day — they know the answer. A regex bug in an earlier version of the codebase silently retained 32 such features, artificially improving apparent accuracy. Phase 3 uses a precise exclusion pattern to eliminate all of them.

### Combining Stages

The final forecast for each quantile and SP is:

```
ssp_q[X][sp] = level_P[X] + shape_deviation[sp]
```

Prediction intervals at the SP level are formed by propagating level uncertainty (P10 to P90 spread) uniformly across all 48 SPs. The shape deviation itself is currently a point estimate (P50 only), so the interval width is constant across the day.

## Main Innovation over Phase 2

In Phase 2, the recursive HGBR predicted SP1 first and then fed that prediction as a feature for SP2, and so on. An error in SP1 compounded through all subsequent predictions. The architecture also required carefully staggered lag construction at SP resolution, which created ongoing leakage risk.

Phase 3 makes the daily level question explicit and independent. Because the level model is estimated once from daily-aggregated history before the day starts, its error is not propagated forward. Every SP's forecast inherits the same level error independently; errors do not accumulate. The shape model then operates on a residualised target — deviations from a known daily mean — which is a more tractable and stable prediction problem than raw SP prices.

The generation mix features add an important causal signal absent in Phase 2. Low wind penetration forces more gas dispatch, raising the marginal cost of generation and pushing prices higher. The overnight anomaly on 18 May 2026 illustrates this: SSP ranged from £115 to £156/MWh during hours that are typically the cheapest in the day, driven by a mean wind share of just 16.4%. Wind and gas percentage lags help the level model anticipate high-cost days, and the SP-level lags (lag-48 and lag-336) help the shape model identify whether the prior-day pattern was driven by similar generation conditions.

## Performance Results

### Model Comparison

| Model | MAE (£/MWh) | sMAPE | Evaluation |
|---|---|---|---|
| Naive lag-48 | 36.34 | 41.6% | batch |
| Seasonal naive lag-336 | 29.40 | 34.7% | batch |
| Rolling mean 48 SP | 26.78 | 28.5% | batch |
| Phase 2 recursive HGBR (P50) | 25.40 | 27.4% | 7-day holdout, honest |
| **Phase 3 two-stage (P50) — 7-day holdout** | **27.17** | **29.6%** | **May 11–17 2026, honest** |
| **Phase 3 two-stage (P50) — walk-forward** | **25.15** | **40.9%** | **119 days, 4 seasons, honest** |

The 7-day holdout sMAPE of 29.6% is lower than the walk-forward figure of 40.9% because May 11–17 2026 was a low-volatility spring week with limited price dispersion. Autumn and spring transition periods have inherently higher relative error. Walk-forward evaluation across four seasons provides a more reliable estimate of generalisation.

### Seasonal Walk-Forward Results

| Season | Period | Days | MAE | sMAPE | Level MAE | Shape Corr |
|---|---|---|---|---|---|---|
| Summer | Jul 2025 | 30 | £23.34 | 34.9% | £9.73 | 0.320 |
| Autumn | Oct 2025 | 29 | £25.30 | 46.0% | £14.25 | 0.550 |
| Winter | Dec 2025 | 30 | £19.56 | 30.7% | £7.62 | 0.386 |
| Spring | Apr 2026 | 30 | £32.40 | 52.3% | £17.07 | 0.464 |
| **Aggregate** | | **119** | **£25.15** | **40.9%** | **£12.15** | **0.429** |

Mean actual price was similar across seasons (£75–83/MWh), so MAE differences reflect genuine model difficulty rather than price scale effects.

## Seasonal Analysis

**Winter** produces the best level MAE (£7.62) and lowest sMAPE (30.7%). Demand is the dominant driver of winter prices, and demand-driven patterns are regular and predictable. Renewable penetration is lower, meaning gas sets the marginal cost more consistently. NIV and lagged price features are highly informative in this regime, and the model can exploit the stability of morning and evening demand peaks.

**Summer** delivers low absolute MAE (£23.34) and a moderate level MAE (£9.73), but the weakest shape correlation (0.320). Summer prices are low and level is relatively predictable because demand is suppressed. The problem is intra-day shape: summer profiles are flat, and small deviations from a near-flat pattern are difficult to call consistently. Peak timing error is highest in summer (approximately 10 SPs) because summer peaks are shallow and shift easily with cloud cover or demand surprises.

**Autumn** shows the best shape correlation (0.550) but high sMAPE (46.0%). Autumn is a transition season with early heating demand, intermittent wind, and growing renewable penetration. This produces pronounced morning and evening demand peaks, which gives the shape model a clearer signal — hence the highest shape correlation. However, price levels are volatile. Occasional price spikes, combined with rapidly changing wind conditions, drive sMAPE up. The level model's level MAE of £14.25 reflects this difficulty.

**Spring** is the hardest season across all metrics (level MAE £17.07, sMAPE 52.3%). High renewable penetration combined with low demand creates an unpredictable dispatch order. Negative price periods and sudden upward spikes make the daily level exceptionally difficult to forecast. April 2026 in particular was characterised by erratic renewable output with large day-to-day swings in solar and wind availability. In this regime, gas-price lags and NIV are less informative because marginal dispatch is dominated by curtailment decisions rather than thermal plant economics.

## Price Derivation Code (P vs N vs K)

**N — Normal market price (~50% of periods).** Generators bid to supply electricity and consumers offer to reduce demand. ESO accepts the cheapest bids first until supply meets demand — like an auction. The price of the last (most expensive) unit ESO had to accept becomes the system price. N means the market set a genuine price based on real supply and demand.

**P — Backup (Replacement) price (~50% of periods).** Sometimes the auction has too few participants or the result would be unreliable (e.g. only must-run nuclear and wind remain). ESO then uses a formula-based fallback price instead. P is most common during the evening peak (18:00–21:00), when dispatch is complex and the conventional bid stack can break down.

**Why N dominates overnight, P dominates evenings:** In the early hours (01:00–06:00) demand is low and stable — the auction is simple, so N wins (~60–65%). During the evening peak, dispatch involves many generator types simultaneously and the bid-offer stack is more likely to fail the reliability test — P wins (~55–65%).

**Note on the bar chart:** The bar height shows the COUNT of periods per code per day (stacked to 48 total). The N/P split within a day is spread across the full 24 hours — N is not confined to the start of the day; it just appears at the bottom of each bar because of how stacked bars are drawn. On the last Sunday of October (BST→GMT clock change), bars reach 50 because the day has two extra periods (the 01:00–02:00 hour runs twice).

**K — Extremely rare (9 occurrences in 5 years).** Applied when ESO bought or sold nothing at all to balance the grid — supply and demand matched perfectly without any intervention. Too rare to be visible in the chart.

## Uncertainty Quantification

The level model produces three separate HGBR models trained on quantile loss for P10, P50, and P90. These provide calibrated daily prediction intervals. At the SP level, the P10 and P90 bands are formed by applying the P50 shape deviation to the level P10 and P90 respectively, propagating level uncertainty uniformly across all 48 SPs. This means interval width is constant within a day. Extending the shape model to produce its own quantile estimates — capturing intra-day uncertainty beyond level uncertainty — is the primary remaining architecture enhancement.

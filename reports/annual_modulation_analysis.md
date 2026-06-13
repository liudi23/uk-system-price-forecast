# Annual Modulation in UK Electricity System Prices

**Author:** UK System Price Forecast Project  
**Date:** 2026-05-18  
**Data source:** Elexon BMRS (May 2021 – May 2026)

---

## Executive Summary

A statistical investigation confirms that UK electricity system prices (SSP) exhibit a genuine annual seasonal pattern, with winter peaks in December/January and a spring trough in April/May. However, this pattern explains only ~3% of total price variance. Inter-year macroeconomic shocks — most prominently the 2022 Russia-Ukraine energy crisis — dwarf the seasonal signal. Incorporating annual harmonic features into the forecasting model yields an 8% MAE improvement over the 36-day baseline, though the gain is attributable to richer training history rather than the harmonic features themselves. Weather covariates remain the recommended next step for capturing the mechanistic drivers of annual modulation.

---

## 1. Data

- **Period:** May 2021 – May 2026 (5 years)
- **Resolution:** 30-minute Elexon BMRS settlement periods
- **Rows:** 87,686 across 1,827 days
- **Target variable:** System sell price (SSP, £/MWh)

---

## 2. Statistical Test for Annual Modulation

A non-parametric Kruskal-Wallis test was applied across the 12 monthly groups to test whether monthly price distributions are drawn from the same population.

| Test | H-statistic | p-value |
|---|---|---|
| Kruskal-Wallis (12 monthly groups) | 71.87 | 5.38 x 10^-11 |

The null hypothesis (all months identical in distribution) is rejected with very high confidence. Annual modulation in SSP is statistically confirmed.

---

## 3. Monthly Price Patterns

### 3.1 Five-Year Monthly Mean SSP

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| £126 | £106 | £122 | £95 | £86 | £94 | £109 | £131 | £130 | £106 | £116 | £143 |

The aggregate pattern shows a clear W-shape: a winter peak in December/January, a spring trough in April/May, a secondary rise through summer into a late-summer/autumn secondary peak, then a further rise into December.

### 3.2 Year-by-Year Monthly Means (£/MWh)

| Year | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| 2021 | — | — | — | — | 74 | 77 | 95 | 108 | 178 | 159 | 186 | 221 |
| 2022 | 199 | 156 | 234 | 172 | 123 | 163 | 233 | 343 | 252 | 121 | 133 | 265 |
| 2023 | 135 | 136 | 127 | 98 | 75 | 91 | 68 | 79 | 80 | 91 | 90 | 69 |
| 2024 | 74 | 58 | 64 | 49 | 72 | 70 | 72 | 57 | 77 | 82 | 94 | 86 |
| 2025 | 124 | 101 | 88 | 75 | 69 | 69 | 79 | 71 | 65 | 76 | 76 | 75 |
| 2026 | 99 | 80 | 95 | 83 | 102 | — | — | — | — | — | — | — |

The 2022 figures illustrate the severity of the Russia-Ukraine energy crisis: August 2022 averaged £343/MWh, more than four times the five-year August mean. The seasonal pattern is entirely masked in that year.

---

## 4. Fourier Decomposition

A two-harmonic Fourier fit was applied to winsorised daily means (p1–p99 clipping to reduce the influence of extreme settlement periods).

| Component | Period | Amplitude | Peak |
|---|---|---|---|
| Overall mean | — | £113.77/MWh | — |
| 1st harmonic | 12 months | £12.55/MWh | November |
| 2nd harmonic | 6 months | £8.48/MWh | — (captures W-shape) |

- **Overall std (winsorised daily means):** £76.50/MWh
- **Two-harmonic fit R²:** 2.9%

The 2nd harmonic is necessary to represent the W-shaped annual pattern visible in the aggregate monthly means. Despite the statistical significance of the annual cycle, the two harmonics together explain only 2.9% of variance in daily prices, confirming that short-range autocorrelation and macro-level level shifts are the dominant sources of variability.

---

## 5. Forecasting Model Results

| Model | Training window | MAE | sMAPE |
|---|---|---|---|
| Best baseline (rolling mean 24h) | — | £26.78/MWh | 28.5% |
| HGBR (no annual features) | 36 days | £16.24/MWh | 18.8% |
| HGBR (with annual harmonics) | 5 years | £14.93/MWh | 17.7% |

The 5-year HGBR model improves MAE by £1.31/MWh (~8%) over the 36-day model. Feature importance analysis shows `ssp_lag_1` is by far the dominant predictor (importance = 21.7), while the annual harmonic features register near-zero importance. The MAE gain therefore reflects access to a larger and more diverse training set, not the harmonic features per se.

---

## 6. Key Interpretations

- **Seasonal pattern is real but weak.** Winter heating demand drives December/January peaks; the spring trough in April/May coincides with mild temperatures and high renewable generation. The Kruskal-Wallis test confirms this is not sampling noise.
- **2022 is a structural outlier.** The Russia-Ukraine energy crisis produced price levels and seasonal shapes incompatible with other years. Any model trained on 2022 data must account for this regime shift.
- **Annual harmonics explain ~3% of variance.** Short-range price autocorrelation is the dominant signal. Adding harmonics to the feature set is low-cost insurance rather than a high-impact change.
- **The improvement from the 5-year model is training-data driven.** Expanding the training window gives the model exposure to a wider range of price regimes, improving generalisation.
- **Next priority: weather features.** Temperature, wind speed, and solar irradiance are the mechanistic drivers of the annual modulation. These variables should be added in the next modelling iteration to provide causal grounding for the seasonal signal.

---

## 7. Features Added to Model

The following calendar features were added to `src/features/calendar_features.py` as a result of this analysis:

| Feature | Description |
|---|---|
| `sin_annual` | Sine component of 12-month harmonic |
| `cos_annual` | Cosine component of 12-month harmonic |
| `sin_half_annual` | Sine component of 6-month harmonic |
| `cos_half_annual` | Cosine component of 6-month harmonic |
| `day_of_year` | Raw day-of-year index (1–366) |

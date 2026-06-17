# SSP Spike Tail Design: Improving High-Price Coverage

**Branch:** `streamlit-data`  
**Date:** 2026-06-16  
**Status:** Design only — no implementation  
**Constraint:** Must not degrade normal-period PI coverage (~79% LOO) or invalidate the Kalman/PI stack already in production.

---

## 0. Problem Statement

The Phase 4 split-conformal PI calibration achieves 79.1% LOO overall coverage (target 80%).  
For high-price settlement periods the same calibration achieves only **40% coverage** (in-sample, >£150 proxy).  
This is not a calibration failure — it is a structural consequence of class imbalance: spikes are 1.2% of rows, so the p80 conformity score is entirely determined by normal-period behaviour, and the widening it prescribes (δ ≈ £24) is far too small to cover spike excesses that average +£77 above q50.

---

## 1. Data Fix First: Join Price Derivation Code

### Problem with the current proxy

`walk_forward_predictions.csv` has no `price_derivation_code` column.  
The only available proxy is `actual > £150`, which has two fundamental defects:

1. **Outcome-conditioned.** It conditions spike membership on the realised price, which is exactly what we are trying to predict. A model evaluated only on rows where actual > £150 automatically under-covers because those rows are selected for being hard.
2. **Crude.** The UK mechanism distinguishes three derivation codes:
   - **P** (Price): price derived from bid/offer stack during system tightness — the true "spike" mechanism
   - **N** (Normal/NIV): price derived from the NIV weighted average — most normal periods
   - **K** (Default): administered price / floor/cap binding

Code-P SPs can occur at moderate prices if the imbalance stack is active; conversely, a high price can occur via NIV averaging without the stack being tight. The proxy misclassifies in both directions.

### Recommended join

The BMRS API endpoint `/balancing/settlement/system-prices/{date}` already returns `priceDerivationCode` (or `price_derivation_code` in the CSVs). This field is present in the raw Elexon feed.

**Action:** Extend the historical data pipeline to join `price_derivation_code` onto `walk_forward_predictions.csv` from the raw BMRS data for dates 2025-07-01 to 2026-04-30. Store it as a `price_derivation_code` column (values `P`, `N`, `K`).

**Spike definition going forward:**  
`is_spike = (price_derivation_code == 'P') OR (ssp_actual > 354)`  
This matches the definition used in the corrector backtest report and preserves compatibility with the existing spike metrics.

---

## 2. Spike Characterisation (from available data)

All numbers below use the `actual > £150` proxy; expect moderate revisions once Code-P labels are joined.

### 2.1 Frequency and magnitude

| Metric | Value |
|---|---|
| Spike rows (>£150) | 70 / 5,709 = **1.2%** |
| Spike days (≥1 spike SP) | 17 / 119 = **14.3% of days** |
| Spike SPs per spike day | mean 4.1, max 14 (2025-10-13) |
| Mean actual on spike SPs | £194 |
| Max actual | £487 (2025-10-13, SP 37) |
| Mean q50 forecast on spike SPs | £117 |
| Mean residual (actual − q50) | **+£77** |
| Mean q90 forecast on spike SPs | £141 |
| Spike SPs where actual ≤ q90 | 10 / 70 = **14%** |

The base model systematically underestimates spike magnitude by ~£77.  
Importantly, the base model's PI **does not widen on spike days**: mean q90−q10 on spike days is £36.7 vs £34.9 on normal days — statistically identical. The model has some ex-ante information about elevated prices (q50=£117 vs daily mean ~£85) but no information that the uncertainty is fundamentally larger.

### 2.2 Seasonal distribution

| Fold | Spike SPs | Fraction |
|---|---:|---:|
| summer-2025 | 3 | 4.3% |
| autumn-2025 | 37 | **52.9%** |
| winter-2025 | 0 | 0.0% |
| spring-2026 | 30 | **42.9%** |

Winter-2025 has **zero spikes** in this window — a genuine seasonal regime with different supply/demand dynamics.  
Autumn-2025 is dominated by a single cluster: 2025-10-13 produced 14 spike SPs (max £487), with three more days in the £250–£315 range. One event (2025-10-13) accounts for 10/70 spike rows and drives the autumn overhang.

### 2.3 SP / time-of-day clustering

Spike rates by season × SP block:

| Fold | Morning (SP 1–16) | Midday (SP 17–32) | Afternoon (SP 33–40) | Evening (SP 41–48) |
|---|---:|---:|---:|---:|
| autumn-2025 | 0.2% | 1.5% | **12.5%** | 0.0% |
| spring-2026 | 2.1% | 1.9% | **2.5%** | 2.1% |
| summer-2025 | 0.2% | 0.0% | 0.8% | 0.0% |
| winter-2025 | 0.0% | 0.0% | 0.0% | 0.0% |

Two distinct regimes:
- **Autumn afternoon (SP 33–40, ~16:30–20:00 UTC):** dominant spike window — demand ramp, low wind, reserve exhaustion
- **Spring morning/midday (SP 13–19, ~06:30–09:30 UTC) + mixed:** different mechanism, likely solar intermittency / gas-fired ramp constraint

### 2.4 Ex-ante drivers (features available ≥ 48 SP ahead)

The level and shape models already include — and rank highly — several spike-relevant ex-ante signals:

| Feature | Model | Rank | Mechanism |
|---|---|---|---|
| `wind_ms_daily_mean` / `wind_ms_daily_max` | Level | 4–5 | Low wind → higher BM price → spike risk |
| `wind_pct_daily_mean_lag1d` | Level | 22 | Wind penetration proxy |
| `wind_pct_lag_48` | Shape | 8 | SP-level wind forecast 24h ahead |
| `gas_pct_daily_mean_lag1d` | Level | 14 | High gas generation = tighter reserve |
| `gas_pct_lag_48`, `gas_pct_lag_336` | Shape | 2–4 | Gas penetration ex-ante signal |
| `spike_count_lag1d` | Level | 56 | Day-before spike activity (regime signal) |
| `spike_count_roll_7d` | Level | 62 | Rolling spike activity |
| `niv_daily_mean_lag1d` | Level | 40 | Lagged Net Imbalance Volume |
| `niv_same_sp_std_7d` | Shape | 11 | Per-SP NIV volatility |
| `neg_price_risk_prob` | Level | 84 | Existing negative-price risk score |
| `is_spike_lag_48` | Shape | — | Previous day spike at same SP |

These features enter the base quantile models, but the models are not specifically optimised to widen their PI on spike-risk days.

---

## 3. The Right Metric

### Why outcome-conditioned metrics are circular

The current assessment — "coverage = 40% for rows where actual > £150" — is correct as a diagnostic but invalid as a monitoring metric.  
Conditioning on the realised outcome selects rows that are, by construction, hard to cover. It says "we miss spikes when they happen" but cannot say "did we know in advance that a spike was likely?"

### The right conditioning variable: an ex-ante spike-risk signal

Define a **spike-risk flag** constructed from information available at forecast time (D−1):

```
spike_risk_day = (spike_count_lag1d ≥ 1)
              OR (wind_pct_daily_mean_lag1d < 10% AND month ∈ {9,10,11,3,4,5})
```

This is computable at inference time without using realised prices.

**Monitoring metrics going forward:**

| Metric | Target | Meaning |
|---|---|---|
| Overall LOO PI coverage | ≥ 78% | Do not regress on the calibrated majority |
| Coverage on `spike_risk_day = True` | ≥ 65% | Partial goal; represents ex-ante identified risk |
| Coverage on `spike_risk_day = True` AND autumn/spring | ≥ 60% | Hardest-to-cover regime |
| Mean Q90 on spike-risk afternoons (SP 33–40) | Track only | Width diagnostic, not a pass/fail |

The ex-ante coverage is inherently lower than 80% (because the classifier is imperfect) but it does not selectively condition on hard outcomes. Track it as a monotonically improving series as more spike data accumulates.

---

## 4. Option Evaluation

### A. Asymmetric per-SP conformal (separate δ_lo / δ_hi)

**Concept:** Instead of one symmetric δ(sp), fit two separate scores:
- δ_hi(sp) = quantile(actual − q90, 0.80) clipped at 0 → widens q90 up
- δ_lo(sp) = quantile(q10 − actual, 0.80) clipped at 0 → widens q10 down

**Data finding:**

| Segment | δ_hi (p80 upper excess) | δ_lo (p80 lower excess) |
|---|---:|---:|
| Normal (≤£150) | £5.8 | £15.7 |
| Spike (>£150) | £96.9 | £0.0 |

The spike signal is exclusively in the upper tail (spike lower excess p80 = £0 — spikes never violate q10). Normal periods are dominated by **lower** violations (q10 too high, not q90 too low).

**Critical finding:** Global asymmetric conformal is **worse for spikes**, not better.  
At p80 across all 5,709 rows, the upper excess is £6.72 — dominated by the 98.8% normal rows where upper excess is only £5.8. The 1.2% spike rows contribute negligibly. Applying global asymmetric δ:

| | Overall | Spike |
|---|---:|---:|
| Symmetric (current) | 80.0% (in-sample) | 42.9% |
| Asymmetric global | 60.0% | **17.1%** |

The asymmetric approach trades overall coverage (60% vs 80%) for no spike gain. This is actively harmful unless conditioned on regime.

**What asymmetric conformal CAN do:**  
It correctly diagnoses that the base model is biased upward for normal periods — q10 is too high (lower violations dominate). A regime-conditioned version (Section 4D) could use δ_hi only on spike-risk SPs and δ_lo for normals, avoiding the perverse global result.

**Cost:** O(hours) of analysis; does not touch base model.  
**Leakage risk:** None — δ estimated from lagged training fold data.  
**Data sufficiency:** Problematic for spikes (70 rows at p80 gives unstable δ_hi if further segmented).  
**Verdict:** Asymmetric scoring is the right framing but not useful globally. Must combine with regime conditioning (D) or a classifier gate (C).

---

### B. Dedicated high-price quantile head (P95 / P99)

**Concept:** Add a P95 or P99 output to the level and/or shape model, trained with the 95th/99th pinball loss alongside the existing P10/P50/P90 heads.

**Mechanism:** The HGBR multi-output structure would generate Q95(daily level) as an additional column; the shape model would produce Q95(SP deviation) and combine them into `ssp_q95 = q95_level + dev_q95`.

**Quantified challenge with current data:**

- 5 years of SSP data at 48 SP/day ≈ 87,600 rows → ~1,050 rows with actual > £150 (1.2%)
- Current training window: ~2 years → ~35,000 rows → ~420 spike rows
- P99 of 35,000 rows = 350 rows defining the tail → moderate statistical reliability
- P99 by SP position: ~7 spike rows per SP position on average → unreliable at SP level
- Seasonal heterogeneity compounds this: a P99 trained on 1 year of data with 0 winter spikes and 52% autumn spikes will produce a seasonally biased Q99

**What P95/P99 buys:**  
A calibrated tail band that the Kalman corrector can incorporate as additional PI uncertainty (Q95 as an alternative upper bound). Even an imperfect Q95 is more informative than widening Q90 by a fixed δ.

**Cost:** Model retraining; requires touching base model artifacts (not just inference-layer); training time.  
**Leakage risk:** None if trained on same D−1 features.  
**Data sufficiency:** Borderline with 2 years of data; adequate with 3–5 years. The autumn-2025 mega-event distorts P99 significantly.  
**Verdict:** Right long-term approach; premature with current data. Revisit when training window extends to include a full cycle (≥ 3 years, ideally including 2 distinct autumn periods).

---

### C. Spike-probability classifier P(spike | ex-ante features)

**Concept:** Train a binary classifier to output P(spike day) using only D−1 and lagged features. Use the output to conditionally widen q90 only when risk is elevated.

**Design at day level (more tractable than SP-level):**

- Response: `spike_day = 1` if any SP on the day has actual > threshold
- Base rate: 17/119 days = 14.3% → manageable class imbalance
- Ex-ante features (all already in the model, no new data required):
  - `spike_count_lag1d`, `spike_count_roll_7d` — recent spike activity
  - `wind_ms_daily_mean` / `wind_pct_daily_mean_lag1d` — wind forecast
  - `niv_daily_mean_lag1d`, `niv_daily_roll_std_7d` — NIV signal
  - `gas_pct_daily_mean_lag1d` — gas penetration (reserve margin proxy)
  - `month`, `is_business_day`, `is_weekend` — seasonal/calendar
  - D+1 wind forecast from BMRS WINDFOR (already fetched by `fetch_bmrs_forecasts.py`)
- Calibration: `CalibratedClassifierCV` on the training fold; evaluate by Brier score and precision–recall, not accuracy

**SP-level extension (for conditional band widening):**

- Response: `is_spike_sp = 1` per (day, SP) — 1.2% base rate → harder
- Additional features: `sp_block`, `sin_sp`/`cos_sp`, `niv_same_sp_std_7d` (already in shape model)
- Day-level score P(spike day) as an input feature for the SP-level classifier

**Production integration (no base model change needed):**

```
Base HGBR → PI calibration → [classifier: P(spike|features)]
  → if P(spike) > τ: q90 += Δ_spike(sp, season)
  → Kalman corrector → splice actuals
```

The threshold τ and Δ_spike(sp, season) are tunable at inference time.

**Cost:** Lightweight — logistic regression or shallow HGBR on existing features; purely in the inference + O(1) correction layer.  
**Leakage risk:** Must verify every feature has ≥ 48 SP lag (all listed above already satisfy this).  
**Data sufficiency:** 17 positive days for day-level → just sufficient for LOO evaluation; 70 spike SPs for SP-level → needs careful regularisation.  
**Verdict:** Most practical near-term option. Provides an ex-ante signal that C can gate A/D. The classifier does not need to be precise — a recall of 50% at 30% precision still doubles the coverage rate on identified spike days.

---

### D. Regime / season-conditioned δ

**Concept:** Instead of a global δ(sp), compute δ(sp, regime) separately for each season or spike-risk regime, then select at inference time based on the forecast date's regime.

**Season-only conditioning:**

| Season | Spike rate | LOO coverage after global δ | Needed extra δ_hi |
|---|---:|---:|---:|
| Summer (JJA) | 0.2% | 81.4% | None |
| Autumn (SON) | 8.0% | 70.3% (35.1% spike) | Large |
| Winter (DJF) | 0.0% | 89.4% | None — over-calibrated |
| Spring (MAM) | 6.3% | 74.9% (43.3% spike) | Moderate |

Autumn and spring need significantly wider bands, particularly in the afternoon SP block. Winter and summer could use a narrower δ (the current symmetric δ over-widens them).

**SP-block × season conditioning:**

The most targeted version: δ(sp_block, season) — 4 seasons × 4 SP blocks = 16 cells.  
Autumn afternoon (SP 33–40) is the single most under-covered cell (spike rate 12.5%).

**Data sufficiency concern:**  
Each cell has ≈ 119 / 4 folds × 1 season × 1 SP block × 8 SPs = ~240 rows (training). For spike-specific calibration this drops to ~30 spike rows in the highest-spike cell (autumn afternoon). Borderline for a stable p80 estimate — but the DIRECTION is clear even if the magnitude is uncertain.

**Cost:** O(hours) analysis; inference-only change; no base model touch.  
**Leakage risk:** Season/SP block are perfectly known at D−1.  
**Verdict:** Good cheap win; partially addresses the autumn/spring gap. Does not help the within-season forecasting (cannot distinguish a calm autumn day from a spike day). Should be combined with C for a full solution.

---

## 5. Option Interactions

```
A (asymmetric δ) × D (regime conditioning) → regime-conditioned asymmetric δ(sp, season)
  • Correct framing: δ_hi large for autumn/spring afternoon; δ_lo retained for normals
  • Solves the "global asymmetric is anti-spike" problem
  • Data: ~37 spike rows in autumn training folds → δ_hi(autumn, SP 33-40) is noisy but directionally valid

C (classifier) gates A/D → conditional widening
  • Use regime-conditioned asymmetric δ as the "spike-risk band"
  • Apply additional δ_hi only when P(spike) > τ (e.g. 0.15)
  • Normal days get the regular calibrated band; spike-risk days get a wider upper tail
  • This is the recommended near-term architecture

B (P95/P99 head) replaces A+D long-term
  • Once ≥3 years of data are available, a dedicated P95 head is cleaner than the δ patchwork
  • The classifier (C) still serves as a gate even with B
```

---

## 6. Recommended Sequence (cheapest defensible first)

### Step 0: Data fix (prerequisite)

Join `price_derivation_code` from the raw BMRS historical data onto `walk_forward_predictions.csv`.  
Without this, spike counts are proxy-dependent and the evaluation is circular.  
**Blocks:** all subsequent steps.  
**Effort:** ~1–2 hours.

---

### Step 1: Season-conditioned symmetric δ (D, immediate)

Compute δ(sp, season) — one δ per (SP, month group) — using the same split-conformal approach as Phase 4 but stratified by season (e.g., {JJA, SON, DJF, MAM}).  
Select at inference time based on `forecast_date.month`.

This directly addresses the autumn/spring under-coverage without a classifier.  
**Expected gain:** Autumn LOO coverage 70.3% → ~76–78%; winter/summer unchanged or narrowed (no penalty to normal coverage).  
**Expected cost:** None to normal-period coverage if δ is allowed to be season-dependent.  
**Effort:** ~half day.

---

### Step 2: Regime-conditioned asymmetric δ (A+D, after Step 0)

Once Code-P labels are available, compute:
- δ_hi(sp, season) from upper-tail scores on Code-P SPs only
- δ_lo(sp, season) from lower-tail scores on non-Code-P SPs

Apply δ_lo to lower tail (addresses systematic q10-too-high bias in normals); δ_hi for autumn/spring afternoon SPs regardless.  
**Expected gain:** Better normal lower tail (LOO coverage should stay ≥79%); more appropriate upper tail sizing for high-risk SP/season combinations.  
**Effort:** ~1 day.

---

### Step 3: Spike-day classifier (C, after Steps 0–1)

Build a day-level logistic regression or shallow HGBR classifier using:
- `spike_count_lag1d`, `spike_count_roll_7d`
- `wind_ms_daily_mean_lag1d`, `wind_pct_daily_mean_lag1d`
- `niv_daily_mean_lag1d`
- `gas_pct_daily_mean_lag1d`
- `month`, `is_business_day`
- D+1 BMRS WINDFOR forecast (already fetched)

Evaluate LOO by Brier score and conditional coverage at τ ∈ {0.1, 0.2, 0.3}.  
Combine with Step 2: on days where P(spike) > τ, apply a wider `δ_hi_spike(sp, season)` to Q90 before the Kalman update.  
**Expected gain:** ~5–15pp coverage improvement on ex-ante identified spike-risk days; negligible impact on normal days (classifier gates the extra widening).  
**Effort:** ~1–2 days.

---

### Step 4: P95/P99 quantile head (B, long-term)

Extend the HGBR training to include P95 as a fifth quantile output.  
Revisit when the training window covers ≥3 full calendar years (including ≥2 distinct autumn tight-supply periods) so the tail quantile is reliably estimated.  
**Prerequisite:** More historical data; retraining the base model.  
**Effort:** ~2–3 days of modelling + evaluation.

---

## 7. Production Monitoring While Steps 1–4 Are Implemented

Track these metrics as new data accumulates (weekly, not daily — spike events are sparse):

| Metric | How to compute | Threshold |
|---|---|---|
| Overall LOO coverage | Rolling 30-day version of Phase 4 LOO | ≥ 77% |
| Ex-ante spike-risk coverage | Coverage on rows where `spike_count_lag1d ≥ 1` or `wind_pct < 10% AND month ∈ {9,10,3,4}` | Track; no hard gate yet |
| Autumn/spring afternoon coverage | SP 33–40 in SON/MAM months | Track; target ≥ 65% after Step 1 |
| Kalman x̂ on spike days | From `kalman_state.json` logs | Alert if > £50 for 2+ consecutive runs |

**Do not** use `actual > £150` as a production monitoring filter — it conditions on the realised outcome and creates a self-referential evaluation where any model "looks bad" on the rows it failed to cover.

---

## Appendix: Key Numbers Summary

| Quantity | Value |
|---|---|
| Overall LOO PI coverage (current) | 79.1% |
| Spike (>£150 proxy) LOO coverage | 40.0% (in-sample) |
| Spike fraction | 1.2% of rows, 14.3% of days |
| Mean spike residual (actual − q50) | +£77 |
| Base band width: normal vs spike days | £34.9 vs £36.7 (statistically identical) |
| Fraction of spike SPs under base q90 | 14% (10/70) |
| Global p80 conformity score | £24.35 |
| Spike-only p80 upper excess (actual−q90) | £96.9 |
| Autumn afternoon SP 33–40 spike rate | 12.5% |
| Winter spike rate | 0.0% |
| Global asymmetric δ_hi (p80, all rows) | £6.72 (anti-spike: dominated by normals) |
| Global asymmetric spike coverage | 17.1% (worse than symmetric 42.9%) |

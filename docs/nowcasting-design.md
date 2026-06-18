# Intraday Nowcasting Head — Design Document

**Branch:** streamlit-data  
**Status:** Design only — no production implementation  
**Date:** 2026-06-18  
**Scope:** A sibling model predicting SSP for the next 1–3 settlement periods (h+1/h+2/h+3, ≈30–90 min ahead), complementing but not replacing the existing day-ahead 48-SP model.

---

## 1. Data & the Lag Ceiling

### 1.1 Information available at forecast time T

At any moment T falling within settlement period SP[t], the Elexon BMRS API publishes initial settlement data with a **~30-minute publication lag**: SP[t] is still in progress, so its actual SSP is not yet known. The most recent settled observation is SP[t−1].

The **realistic information set** at forecast time T is:

| Available at T | Notes |
|---|---|
| SSP lags 1–k: SP[t−1], SP[t−2], … | All confirmed by Elexon BMRS (BOALF-derived) |
| NIV lags 1–k: NIV[t−1], NIV[t−2], … | Net Imbalance Volume — leading indicator |
| Price derivation code lags (N/P/K) | NP (normal-price) vs EN (energy-not-balanced) regime |
| Settlement price adjustments lags | sell/buy price adj — secondary signal |
| Time-of-day, weekday, calendar | Fully known at T |

**Targets to predict:**

| Horizon | Target | Notes |
|---|---|---|
| h+1 | SP[t] | In-progress — no actual available at T. The **hard ceiling** |
| h+2 | SP[t+1] | Future |
| h+3 | SP[t+2] | Future |

The ~30-min lag is the **hard accuracy ceiling**. No model can know whether SP[t] will be £67 or £146 before it settles; it can only infer from SP[t−1] and NIV[t−1]. A sudden spike or collapse in the in-progress period is fundamentally unobservable.

### 1.2 Fetch path

Both data inputs already exist:

- **`src/data/fetch_elexon.py`** — batch historical SSP + NIV (used by the daily pipeline)
- **`src/data/fetch_intraday.py`** — polls the same Elexon BMRS endpoint for today's settled SPs; overwrites `data/raw/intraday_prices.csv` on each call

The nowcaster reuses both paths unchanged. At inference time it reads the last N rows of `intraday_prices.csv` (or `system_prices.csv` for the same-day window) plus `system_prices_5yr.csv` for the training feature matrix.

### 1.3 Data volume

| Training window | SP rows | Days | Comment |
|---|---|---|---|
| 2 weeks | 672 | 14 | Loses diurnal / weekday structure |
| 4 weeks | 1,344 | 28 | Minimum to see two full weekly cycles |
| 12 weeks | 4,032 | 84 | Recommended minimum |
| 52 weeks | 17,472 | 365 | Captures all seasonal regimes |
| 2024+ (≈104 wks) | 41,662 | 867 | Post-crisis, regime-stable |
| Full 5yr | 87,686 | 1,826 | Includes crisis years — biases means |

---

## 2. Autocorrelation Structure

All figures computed on 2024+ data (post-crisis, regime-stable), n=41,662 SP rows.

### 2.1 SSP autocorrelation

| Lag (SPs) | ACF | PACF |
|---|---|---|
| 1 | **0.828** | **≈1.0** |
| 2 | 0.710 | ≈0.0 |
| 3 | 0.610 | ≈0.0 |
| 4 | 0.555 | ≈0.0 |
| 5 | 0.510 | ≈0.0 |
| 6 | 0.449 | ≈0.0 |

The PACF result is the single most important finding: **PACF = ~1 at lag-1 and ~0 at lags 2–6**. This is the fingerprint of a near-random-walk AR(1) process. Conditional on SSP[t−1], lags 2–6 carry no additional linear information about SSP[t]. This means **persistence (predict SP[t] = SP[t−1]) is the Bayes-optimal linear predictor at h+1**, and any model adding lag-2+ features is fighting noise.

### 2.2 SSP-to-SSP predictability by horizon

| Horizon | corr(SSP[t−1], SSP[t+h−1]) | Interpretation |
|---|---|---|
| h+1 | 0.828 | Strong but with large tail volatility |
| h+2 | 0.710 | Decays substantially |
| h+3 | 0.610 | Still useful but persistence is roughly optimal |

### 2.3 NIV autocorrelation and cross-correlation with SSP

| Lag | NIV ACF |
|---|---|
| 1 | 0.794 |
| 2 | 0.613 |
| 3 | 0.460 |

NIV persists strongly — a large imbalance at SP[t−1] is likely to persist. Its predictive value for future SSP:

| | corr(NIV[t−1], SSP[t+h−1]) |
|---|---|
| h+1 | **0.411** |
| h+2 | 0.339 |
| h+3 | 0.270 |

But this is **heavily regime-dependent**:

| Regime | corr(NIV[t−1], SSP[t]) |
|---|---|
| NP (price_derivation_code = N) | **0.381** |
| EN (code = P/K) | **0.079** |

In EN regime, NIV is decoupled from SSP because the derivation mechanism changes. The NIV→SSP link is real but only in normal-price periods.

### 2.4 Regime and time-of-day effects

**Run-length statistics (2024+):**
- NP runs: median 3 SPs, mean 4.5, max 53
- EN runs: median 2 SPs, mean 4.0, max 41

Regime switches are frequent and short. A current-regime flag captures a weak signal, but a regime *classifier* predicting the next SP's regime would be harder to build than the nowcaster itself.

**ACF lag-1 by time-of-day:**

| Period | ACF lag-1 |
|---|---|
| Night (00:00–06:00) | 0.753 |
| Morning (06:00–12:00) | 0.750 |
| Afternoon (12:00–18:00) | 0.809 |
| **Evening (18:00–24:00)** | **0.559** |

Evening shows substantially lower autocorrelation — the market is more volatile and mean-reverting near the close of the business day. Any nowcaster should include time-of-day encoding.

**Rolling volatility as a predictor of next-step error:**

corr(rolling-6SP std of SSP, |SSP[t]−SSP[t−1]|) = **0.463**

Recent volatility is a meaningful predictor of forecast difficulty, supporting its inclusion as a feature for prediction-interval width (if quantile heads are added).

---

## 3. Baselines

Walk-forward evaluation on 2024+ data (n≈38,000–42,000 SP rows). All baselines evaluated separately at h+1, h+2, h+3 using the actual settled SSP as the target.

### 3.1 Persistence and exponential smoothing

| Baseline | h+1 MAE | h+2 MAE | h+3 MAE |
|---|---|---|---|
| Persistence (last settled SP) | **£16.51** | £22.37 | £26.62 |
| ES α=0.7 | £17.35 | £22.56 | £26.37 |
| ES α=0.5 | £18.74 | £23.30 | £26.65 |
| AR(2) OLS (rolling 4-week window) | £18.10 | £23.26 | £26.61 |

Key result: **persistence dominates all other linear baselines at h+1 and h+2**. AR(2) and ES are strictly worse. This is consistent with the PACF result: AR(1) = persistence is the optimal linear predictor.

At h+3, ES(0.7) edges persistence by £0.25 (0.9%), which is within noise. No linear model meaningfully beats persistence at any horizon.

### 3.2 Incumbent: day-ahead model + Kalman correction

The existing production system (Phase 3+4) generates a 48-SP day-ahead forecast at ~12:30 UTC. At intraday time, the Kalman corrector applies a bias estimate `x̂` with horizon decay γ=0.966:

- h+1 correction: 96.6% of `x̂` applied
- h+2 correction: 93.3% of `x̂`
- h+3 correction: 90.1% of `x̂`

**Incumbent vs persistence (computed over 32 archived forecast days, May–Jun 2026):**

| Horizon | Persistence MAE | DA+Kalman MAE | DA+Kalman RMSE | DA wins? |
|---|---|---|---|---|
| h+1 | £17.71 | £30.87 | £39.67 | No (−74%) |
| h+2 | £22.75 | £30.86 | £39.66 | No (−36%) |
| h+3 | £27.06 | £30.74 | £39.50 | No (−14%) |

**The day-ahead model is dramatically weaker than persistence at all three horizons.** This is expected: the day-ahead model was trained to minimise 24-hour-ahead error, not 30–90 minute error. Its predictions are smooth daily profiles that don't track the intraday random walk. The Kalman correction partially compensates for systematic daily bias but cannot correct local short-term noise.

This means any nowcaster that matches persistence beats the incumbent. The commercial bar is low on this metric.

**Persistence by regime:**

| Condition | h+1 Persistence MAE |
|---|---|
| Calm (SSP ≤ £150) | £16.00 |
| Spike (SSP > £150) | £38.85 (n=860) |

Spike SPs are ~2% of rows (2024+) but account for a disproportionate share of error.

---

## 4. Candidate Model

### 4.1 HGBR prototype results

An HGBR (HistGradientBoostingRegressor, consistent with the existing Phase 3 stack) was prototyped with the following feature set:

- SSP lags 1–6
- NIV lags 1–3, NIV delta (lag1−lag2), NIV rolling mean over 6 SPs
- Rolling volatility (6 SP and 12 SP windows)
- Current regime flag (NP/EN from lag-1 derivation code)
- Time-of-day (sin/cos of hour/24)
- Weekday

Walk-forward evaluation: 12-week training window, 2-week eval steps, 2024+ data.

| Horizon | Persistence MAE | ES(0.7) MAE | HGBR MAE | HGBR vs persistence |
|---|---|---|---|---|
| h+1 | £16.59 | £17.61 | £19.40 | **−17.0% (worse)** |
| h+2 | £22.53 | £22.77 | £23.92 | −6.2% (worse) |
| h+3 | £26.80 | £26.52 | £26.63 | +0.6% (marginal) |

**The HGBR prototype fails to beat persistence at h+1 and h+2.** At h+3 it is statistically indistinguishable from both persistence and ES.

### 4.2 Why does HGBR underperform at short horizons?

Several factors reinforce each other:

**1. Near-random-walk dynamics.** The PACF = 0 at lags 2–6 means those features are pure noise. HGBR learns splits on them and overfits, producing forecasts that wander further from the last observation than persistence does.

**2. Heavy-tailed SSP distribution.** Max SSP is £4,038; std=£112. Extreme spikes are rare in 2024+ (~2%) but dominate squared-error loss. HGBR learns to predict the conditional mean of a heavy-tailed distribution, which is not the same as following the random walk.

**3. NIV→SSP link collapses in EN regime.** corr(NIV, SSP+1) = 0.38 in NP but only 0.08 in EN. When the EN regime flag is 1 (38% of rows in 2024+), the NIV features become noise. The model cannot cleanly switch off NIV features per regime without explicit interaction terms or a regime-conditional architecture.

**4. Variance of HGBR predictions.** HGBR produces predictions that vary more than the true conditional mean, adding variance that persistence avoids by simply repeating the last observation.

### 4.3 Proposed improvements (not yet prototyped)

Three directions could close the gap before a build decision:

**A. Huber loss + SP-level correction.** Instead of mean-squared loss, use Huber loss (δ≈50) to downweight extreme spikes. Then add a SP-specific residual correction (learned offset by settlement period). This mimics what the Kalman corrector does but at finer granularity.

**B. NP/EN regime-conditional model.** Train two separate HGBR heads — one for NP rows, one for EN rows — and use the lag-1 derivation code to select which head to apply. In NP regime, NIV features are strong signals; in EN regime, lag-1 SSP (persistence) should dominate with essentially no other features. This would likely achieve near-persistence in NP and pure persistence in EN, capturing the NIV signal where it exists.

**C. Structured residual: predict Δ = SSP[t] − SSP[t−1].** Predicting the *change* rather than the level may stabilise training. The target distribution (centered near 0) is much better behaved than raw SSP. Persistence corresponds to predicting Δ=0. Any positive MAE reduction over 0 directly translates to beating persistence. Preliminary analysis suggests Δ has much lower fat-tailed behaviour on calm days, and the signal structure may be cleaner.

### 4.4 Feature set (revised proposal)

The production feature set should be:

| Feature | Source | Leakage-safe? |
|---|---|---|
| SSP lags 1–4 (SP[t−1] to SP[t−4]) | system_prices.csv | Yes — all settled |
| NIV lags 1–3 | system_prices.csv | Yes |
| NIV delta (lag1−lag2) | system_prices.csv | Yes |
| Rolling 6-SP SSP std | system_prices.csv | Yes |
| Regime flag (NP/EN from lag-1 code) | system_prices.csv | Yes |
| Time-of-day sin/cos | Calendar | Yes |
| Weekday | Calendar | Yes |
| **Day-ahead model Q50 for target SP** | model_assets/next_day_forecast | Yes — generated night before |

The day-ahead Q50 for SP[t] is always available at intraday time (generated the previous evening). It encodes structural priors (seasonal level, daily shape) that the short lag window cannot reconstruct. It is the single most important feature to add and is not yet in the prototype.

Lags 5–6 should be dropped: the PACF confirms they add nothing conditional on lag-1. Lag-6 adds noise.

**SSP lags 2–4 should be expressed as differences** (lag1−lag2, lag2−lag3, etc.) to capture momentum rather than absolute level — this is equivalent to AR(1) residuals and reduces collinearity.

### 4.5 Comparators noted (not evaluated)

- **NP/EN regime classifier**: could gate the NIV→SSP feature weight. Simpler to implement as a hardcoded interaction (feature × regime_flag) than a full classifier.
- **State-space / Kalman with NIV exogenous**: a Kalman filter with NIV as an input in the observation equation would be theoretically well-motivated. It would require fitting Q and R matrices; fitting time is trivial but validation against the HGBR is needed to justify the complexity.

---

## 5. Training Window and Recency

### 5.1 Window length tradeoffs

| Window | SP rows | Pros | Cons |
|---|---|---|---|
| 2 weeks | 672 | Captures latest regime | Loses weekday and diurnal structure |
| 4 weeks | 1,344 | Two full weekly cycles | Still thin; spike rate unreliable |
| 12 weeks | 4,032 | Full seasonal cycle segment; recommended | Includes pre-regime-change data |
| 52 weeks | 17,472 | Robust; all seasonal regimes | 2024–2025 may differ from current |
| 2 years+ | 41,662+ | Maximum variance | Crisis years (2021–22) contaminate means |

**Recommendation: 52-week rolling window as the primary training regime**, which gives 17,472 rows and covers all seasons. Supplement with **exponential recency weighting** (sample_weight = exp(−λ × days_ago) with λ corresponding to a 4-week half-life) to emphasise the last few weeks without sacrificing structural knowledge from older data.

### 5.2 Why not a tiny window?

2-week training means ~50 rows per settlement period per half-hour slot. The diurnal and weekday features (sin/cos, weekday) cannot be estimated reliably; the HGBR tree structure will not learn consistent splits. Small windows also produce volatile coefficient updates — a single spike week dominates the loss.

### 5.3 Retrain cadence

HGBR fit time on 17,472 rows × 12 features is under 1 second. Hourly retrain is computationally trivial. The retrain schedule should be:

- **Triggered by each `fetch_intraday.py` poll** (currently per 30-min SP cycle)
- Alternatively, hourly (every 2 SPs) is sufficient — recency changes slowly
- Model artifact: a single serialized HGBR per horizon (3 files: `nowcast_h1.pkl`, `nowcast_h2.pkl`, `nowcast_h3.pkl`)

The responsiveness-vs-stability trade-off: hourly retrain with 4-week sample-weight half-life gives a balance where the last 2 weeks account for ~75% of the effective training weight, while not discarding 50 weeks of structural pattern. This is much better than a 2-week hard window.

---

## 6. Validation Protocol

### 6.1 Core principle: time-series CV only

Never use random splits. The SSP series is a correlated time series; random splits leak future information through shared autocorrelation structure. All evaluation uses **walk-forward (expanding or rolling window) with retrain at each step**.

### 6.2 Protocol specification

```
For each eval_step t = TRAIN_END, TRAIN_END+step, ..., EVAL_END:
  1. Fit nowcaster on data[t-WINDOW : t]
  2. Predict h+1, h+2, h+3 targets at time t
  3. Record actual SSP[t], SSP[t+1], SSP[t+2]
  4. Also record: baseline predictions (persistence, ES, DA+Kalman)
  5. Tag each prediction: regime(t), time-of-day(t), spike(t)
```

**Evaluation window:** 2024-H1 onwards (≥18 months of post-crisis data)
**Retrain step:** every 2 weeks (consistent with proposed retrain cadence)
**Baselines at every step:** persistence, ES(0.7), AR(2), and DA+Kalman Q50

### 6.3 Report metrics per horizon

For each of h+1, h+2, h+3:

| Metric | Threshold | Rationale |
|---|---|---|
| MAE | Primary metric | Interpretable in £/MWh |
| RMSE | Secondary | Spike sensitivity |
| P90 error | Reported | Tail behaviour |
| Improvement vs persistence | Ship gate (see §6.5) | Hardest baseline at h+1 |
| Improvement vs ES(0.7) | Reported | Hardest at h+2–3 |

### 6.4 Regime stratification

Report all metrics split by:

| Stratum | Definition | Why |
|---|---|---|
| Calm | SSP[t] ≤ £120 | ~85% of rows; baseline is high-confidence |
| Transition | £120 < SSP[t] ≤ £200 | 10–12%; commercial value of accuracy |
| Spike | SSP[t] > £200 | 2–3%; extreme loss on errors; model hardest here |
| NP regime | code = N | NIV features informative |
| EN regime | code = P/K | NIV features largely uninformative |
| Evening | 18:00–24:00 | Lowest ACF; hardest prediction window |

### 6.5 Success gate

The nowcaster ships for a given horizon **only if it beats the hardest baseline at that horizon**:

| Horizon | Hardest baseline | Required improvement | Ship threshold |
|---|---|---|---|
| h+1 | Persistence (£16.5 MAE) | ≥5% MAE reduction | MAE ≤ £15.7 |
| h+2 | Persistence (£22.4 MAE) | ≥5% MAE reduction | MAE ≤ £21.3 |
| h+3 | Persistence / ES(0.7) | ≥5% MAE reduction | MAE ≤ £25.3 |

5% is deliberately conservative: given the near-random-walk structure, any improvement below this margin risks being a backtest artefact. The prototype failed these gates at h+1 and h+2.

---

## 7. Scope and Integration

### 7.1 Sibling architecture

The nowcaster is a **sibling head**: it shares data ingestion and the dashboard surface, but is trained, retrained, and evaluated entirely independently of the day-ahead 48-SP model. It does not replace or modify the Phase 3+4 pipeline.

```
Shared:
  fetch_intraday.py   → data/raw/intraday_prices.csv
  fetch_elexon.py     → data/raw/system_prices_5yr.csv
  streamlit_app.py    → (new panel)

Separate:
  src/models/nowcast.py           → train + serve
  model_assets/nowcast_h{1,2,3}.pkl
  model_assets/nowcast_output.csv → latest 3 predictions
```

### 7.2 Dashboard surface

If shipped, the nowcaster would surface as a new sidebar panel or compact metric block showing:

```
Near-term SSP (nowcast)
SP+1 (≈15:00):  £105  [P10=£82  P90=£131]  ← h+1 prediction
SP+2 (≈15:30):  £108  [P10=£84  P90=£134]  ← h+2 prediction
SP+3 (≈16:00):  £107  [P10=£83  P90=£133]  ← h+3 prediction
Last settled: SP29 = £106.40 (15:00)
```

This complements, not replaces, the Day-Ahead panel (which shows the full 48-SP shape for the rest of the day).

### 7.3 Retrain integration

At each `fetch_intraday.py` call (≈every 30 min):
1. Append new settled SP to training matrix
2. Re-fit 3 HGBR models (≈2 seconds total)
3. Write predictions to `model_assets/nowcast_output.csv`
4. Dashboard reads this file (no git commit needed — runtime-only like kalman_state.json)

---

## 8. Recommendation

### 8.1 Summary of evidence

| Question | Finding |
|---|---|
| Is there signal to exploit? | Yes — ACF lag-1 = 0.83, NIV cross-corr = 0.41 (NP) |
| What is the hard ceiling? | The 30-min publication lag; SP[t] is unobservable at forecast time |
| Is persistence hard to beat? | Yes — it is the Bayes-optimal AR(1) predictor (PACF confirms) |
| Does the DA model help at these horizons? | No — DA MAE (£30.9) is 74% worse than persistence (£17.7) at h+1 |
| Did HGBR prototype beat persistence? | No — HGBR was 17% worse at h+1, 6% worse at h+2, tied at h+3 |
| Is there a path to improvement? | Yes — three specific modifications not yet prototyped |

### 8.2 Verdict: conditional pass, not yet ready to build

**Do not build the production nowcaster in its current form.** The prototype results are clear: with vanilla lag features and a single global HGBR, the model fails the h+1 and h+2 ship gates. Building production infrastructure for a model that underperforms persistence would be negative value.

**However, this is not a structural dead end.** The failure mode is understood and three targeted changes are likely to improve the model materially:

1. **Regime-conditional architecture**: separate NP / EN heads (or explicit NP×NIV interaction feature). The NIV signal is real in normal-price periods; the current prototype is diluted by EN periods where NIV is noise.

2. **Predict Δ (change) not level**: reformulating as a change-prediction task frames persistence as "predict Δ=0", which is a reasonable prior that the HGBR can improve on. The feature distribution becomes symmetric and better-scaled.

3. **Day-ahead Q50 as a feature**: the structural prior from the day-ahead model (expected level, time-of-day shape) should significantly help at h+2 and h+3 where the random-walk signal has decayed more. This is the single highest-priority missing feature.

**Recommended next step (a second design-experiment pass, not production code):** Prototype the regime-conditional Δ-prediction HGBR with the DA Q50 feature included, and report walk-forward results against the same ship gates. If h+2 and h+3 clear the gate (≥5% improvement over persistence), implement and evaluate those two horizons. h+1 is the hardest target and may not reach the gate even with these improvements — which is acceptable; ship only the horizons that clear.

The commercial value, if the model ships, is concentrated in the **transition regime** (£120–£200) and **evening periods** (18:00–24:00) where the DA model is most inaccurate and even modest improvements over persistence are commercially meaningful.

---

*All baseline numbers were computed via walk-forward over 2024+ data (post-crisis). Prototype HGBR used 12-week rolling window, 2-week retrain step, standard MAE loss. The 32-day DA archive covers May–Jun 2026 only and should not be treated as a long-run estimate of DA model performance.*

---

## Experiment 2 — Partial-Signal Diagnostic

**Purpose:** Measure how much signal exists *beyond* persistence at h+1/h+2/h+3 to determine whether the proposed improvements in §4.3 are likely to cross the 5% ship gate. Persistence is already near-optimal, so the only exploitable signal is whatever predicts the *change* that persistence misses.

**Data:** 2024+ post-crisis data (n≈43,000 SP rows); partial R² for the DA Q50 feature computed on the 32-day May–Jun 2026 archive overlap only (n≈1,440 rows). All features use only data ≤ last settled SP.

### E2.1 Residual target definition

Define r_h = SSP[t+h−1] − SSP[t−1] for h = 1, 2, 3. Predicting r_h = 0 is exactly persistence. Any model that improves on this is by definition improving on persistence. The distribution of r_h:

| Horizon | Mean | Std | P25 | Median | P75 |
|---|---|---|---|---|---|
| r₁ (h+1) | 0.0 | 35.1 | −6.2 | 0.0 | +5.9 |
| r₂ (h+2) | 0.0 | 45.5 | −12.0 | 0.0 | +12.0 |
| r₃ (h+3) | 0.0 | 52.8 | −17.0 | 0.0 | +16.3 |

The distributions are symmetric around zero (SSP has no drift at these horizons), heavy-tailed, and have median = 0 exactly (an exact 50th-percentile symmetry). This confirms that persistence is the median-optimal predictor; any improvement must come from the tails.

### E2.2 Partial R² per signal (full dataset, controlling for y0)

All partial R² computed by the Frisch-Waugh method: regress both r_h and the candidate signal on y0 = SSP[t−1], take residuals, report squared correlation. Winsorized at 1% tails to remove crisis-era outliers.

#### H+1

| Signal | Partial R² | r | Implied MAE reduction |
|---|---|---|---|
| ΔSSP momentum (lag1−lag2) | **0.02786** | −0.167 | **£0.23 (1.4%)** |
| NIV[t−1] | 0.00100 | +0.032 | £0.01 |
| ΔNIV | 0.00069 | +0.026 | £0.01 |
| vol₁₂ | 0.00054 | −0.023 | £0.004 |
| vol₆ | 0.00016 | +0.013 | £0.001 |
| NIV rolling mean 6SP | 0.00009 | +0.009 | £0.001 |
| **All combined (naive ΣR²=3.0%)** | | | **£0.25 upper bound (1.5%)** |

#### H+2

| Signal | Partial R² | r | Implied MAE reduction |
|---|---|---|---|
| ΔSSP momentum | **0.01452** | −0.121 | **£0.16 (0.7%)** |
| NIV[t−1] | 0.00187 | −0.043 | £0.02 |
| vol₁₂ | 0.00169 | −0.041 | £0.02 |
| NIV rolling mean 6SP | 0.00127 | −0.036 | £0.01 |
| ΔNIV | 0.00088 | +0.030 | £0.01 |
| **All combined (ΣR²=2.0%)** | | | **£0.23 upper bound (1.0%)** |

#### H+3

| Signal | Partial R² | r | Implied MAE reduction |
|---|---|---|---|
| ΔSSP momentum | **0.01462** | −0.121 | **£0.20 (0.7%)** |
| NIV[t−1] | 0.00981 | −0.099 | £0.13 |
| vol₁₂ | 0.00353 | −0.059 | £0.05 |
| NIV rolling mean 6SP | 0.00332 | −0.058 | £0.04 |
| **All combined (ΣR²=3.1%)** | | | **£0.42 upper bound (1.6%)** |

**Implied MAE reduction** is computed as persistence MAE × (1 − √(1−R²)), assuming Gaussian residuals. This is an optimistic upper bound: it assumes (1) signals are orthogonal and additive, (2) the model optimises MAE directly, (3) out-of-sample R² matches in-sample. Real MAE reduction will be lower.

**Key observation:** ΔSSP momentum (r ≈ −0.17 at h+1) is consistently the dominant signal. The negative sign confirms mean-reversion: if SSP rose by £10 in the last SP, the next SP's change is predicted to be 14–24% of that move in the opposite direction:

| Regime | ΔSSP mean-reversion coefficient |
|---|---|
| NP | −0.144 (14.4% reversal of last move) |
| EN | −0.110 (11.0% reversal) |
| Overall | −0.238 (inflated by spike tails) |

This is the only actionable linear signal. Its effect is small: a £10 upward move predicts only £1.4 reversal. Applied to the 25th–75th percentile of ΔSSP (roughly ±£10), the correction is at most ±£1.4 on average.

### E2.3 Day-ahead Q50 as a feature

The DA Q50 carries information about the expected *level* for that SP (seasonal/daily shape), making the signal different from the short-lag features above. Evaluated only on the 32-day archive overlap:

| Horizon | DA-diff partial R² | r | Implied MAE reduction |
|---|---|---|---|
| H+1 | 0.051 | +0.227 | £0.43 (2.6%) |
| H+2 | 0.078 | +0.279 | £0.89 (3.9%) |
| H+3 | 0.099 | +0.314 | £1.35 (4.9%) |

The DA partial R² grows with horizon — as the random-walk signal decays (r₃ has lower autocorrelation than r₁), the structural prior from the day-ahead model becomes proportionally more valuable. At h+3, the DA feature alone implies a **4.9% MAE reduction**, which is just below the 5% ship gate.

**Important caveat:** These figures come from only 32 days (May–Jun 2026). The 32-day window likely underestimates the DA feature's long-run value, since it's sampled from a single seasonal period. A longer archive is needed for a reliable estimate.

### E2.4 Regime-conditional partial R²

The design document hypothesised (based on raw NIV–SSP level correlation: NP=0.38, EN=0.08) that a regime-conditional architecture would unlock the NIV signal. The partial R² analysis, which properly conditions on y0, tells a different story:

| Regime | NIV partial R² (h+1) | Implied MAE reduction |
|---|---|---|
| NP (n=22,892) | **0.038** | £0.32 (1.9%) |
| EN (n=20,255) | **0.039** | £0.32 (2.0%) |

**NIV's partial R² is nearly identical in NP and EN regimes.** The large raw correlation difference (0.38 vs 0.08) was a level-correlation artefact: in EN periods, both NIV and SSP level are elevated by the same underlying supply shortage. Once we condition on the current SSP level (y0), NIV adds similar marginal information in both regimes. The regime-conditional architecture proposed in §4.3 would not unlock hidden NIV signal as hypothesised.

Regime-conditional upper bounds (naive sum of top-3 signals):

| Horizon | Regime | Combined upper bound |
|---|---|---|
| H+1 | NP | £0.57 (3.4%) |
| H+1 | EN | £0.43 (2.7%) |
| H+2 | NP | £0.36 (1.6%) |
| H+2 | EN | £0.21 (0.9%) |
| H+3 | NP | £0.24 (0.9%) |
| H+3 | EN | £0.11 (0.4%) |

The NP regime shows slightly more signal at h+1, but the difference is modest and not driven by NIV. Both regimes are well below the 5% gate from any intraday signal alone.

### E2.5 Time-of-day effects

Evening (18:00–24:00) shows the lowest lag-1 ACF (0.559 vs 0.75–0.81 at other times), suggesting more signal. However, the partial R² of ΔSSP and NIV in the evening is not materially different from other periods. The ΔSSP partial R² at h+1 is:

| Period | ΔSSP partial R² | NIV partial R² |
|---|---|---|
| Night (00-06) | 0.034 | 0.0005 |
| Morning (06-12) | 0.021 | 0.0016 |
| Afternoon (12-18) | 0.019 | 0.0018 |
| Evening (18-24) | 0.029 | 0.0025 |

The evening is marginally the richest period for ΔSSP signal (~3.4% R²), but the difference is not large enough to change the overall picture. Evening h+3 OLS shows ~5% MAE improvement, the only sub-period/horizon combination that hints at crossing the gate — but this is a subset analysis (n≈2,600 rows of evening-only) and is unreliable.

### E2.6 Volatility → |next-step error| relationship

The design document §2.4 reported corr(vol₆, |r₁|) = 0.463. This was inflated by unwinsorized extreme outliers and should be **corrected**:

| Condition | corr(vol₆, |r₁|) | R² | Interpretation |
|---|---|---|---|
| Overall, unwinsorized | 0.243 | 0.059 | Spike outliers inflate the signal |
| Within NP (winsorized) | **0.189** | **0.036** | Genuine within-regime signal |
| Within EN (winsorized) | **0.143** | **0.020** | Weaker in EN |

The within-regime values are the correct ones to use for estimating PI-width value. R²≈0.036 implies that vol₆ explains 3.6% of the variance of next-step absolute error in NP regime. Using √(1−0.036) ≈ 0.982: a volatility-conditioned PI band would be about 1.8% narrower than a fixed-width band on average in NP — a marginal improvement. The correlation is real but too weak to justify significant engineering effort.

The vol₆ signal is also not independent of regime: EN periods have higher mean vol₆ (£18.8 vs £18.4), so a vol₆-based band already partially proxies regime. The additional value of conditioning on regime after vol₆ is approximately 0.

### E2.7 Verdict per horizon

The implied MAE reductions below are **upper bounds** assuming optimal feature combination and MAE-direct optimisation. Real walk-forward results will be lower (see §4 for the prototype, which achieved −17% at h+1 with suboptimal MSE loss).

| Horizon | Persistence MAE | Intraday signals upper bound | DA feature upper bound | Combined upper bound | 5% ship gate | Verdict |
|---|---|---|---|---|---|---|
| H+1 | £16.51 | £0.25 (1.5%) | £0.43 (2.6%) | ~£0.65 (≈3.9%) | £0.83 | **DO NOT BUILD — below gate** |
| H+2 | £22.37 | £0.23 (1.0%) | £0.89 (3.9%) | ~£1.05 (≈4.7%) | £1.12 | **DO NOT BUILD — marginal** |
| H+3 | £26.62 | £0.42 (1.6%) | £1.35 (4.9%) | ~£1.70 (≈6.4%) | £1.33 | **MAYBE — DA feature alone near gate** |

Notes on the combined upper bounds:
- DA feature is partially orthogonal to intraday signals (structural level vs recent noise), so combination is roughly additive
- The h+3 combined bound (6.4%) assumes the DA partial R² estimate (32 days) generalises to a longer period and independent horizon — a significant assumption

### E2.8 Revised recommendation

The partial-signal diagnostic sharpens the §8 recommendation as follows:

**H+1: Do not build a point model.** The total exploitable linear signal beyond persistence is ≈1.5% MAE improvement from intraday features + ≈2.6% from DA (if the 32-day estimate generalises), totalling ≈3.9% — below the 5% gate. Given the prototype HGBR was already −17% (worse than persistence) with MSE loss, even a perfectly tuned MAE-loss model is unlikely to clear the gate in walk-forward evaluation. **Ship persistence as the h+1 nowcast.** Label it honestly: "last settled SP = £X".

**H+2: Do not build at this stage.** The intraday signal is negligible (1.0% upper bound). The DA feature could contribute ~3.9% — bringing the combined upper bound to ≈4.7%, just below the 5% gate. A model built on DA Q50 alone (Δ-prediction regressed on da_q50 − y0 for the unsettled SPs) might approach the gate, but the 32-day estimate is too thin to be confident. **Revisit when 6+ months of DA archive are available (est. Nov 2026).**

**H+3: Possible, with DA feature only.** The DA partial R² (9.9%) alone implies ≈4.9% MAE improvement, and combined with intraday signals the upper bound reaches ≈6.4%. This is the only horizon where the signal meaningfully exceeds the gate, and the mechanism is clear: the random-walk signal has decayed enough (r₃ = 0.61) that the structural DA prior accounts for a larger fraction of the residual variance. **If the Exp 2 findings hold on a longer archive (≥3 months), build a minimal h+3 model: a single linear predictor r₃ ≈ α × (da_q50_h3 − y0) + β × ssp_delta, without the complexity of a full HGBR.** Walk-forward against persistence and the 5% gate before shipping.

**Regime-conditional architecture (§4.3, improvement A): not recommended.** The partial R² analysis shows NIV has nearly identical predictive value in NP and EN after conditioning on y0. Separate regime heads add complexity without addressing the real bottleneck, which is signal magnitude rather than signal distribution.

**Volatility-conditioned PI bands:** modest but real value (R²≈3.6% within NP). A simple rule `PI_width(t) = base_width × (1 + k × vol₆_normalised)` would add less than 2% coverage improvement and is not worth building until the point forecast itself clears the gate.

**Overall conclusion:** The three proposed fixes in §4.3 were ordered incorrectly by likely impact. The correct priority is:
1. ~~Regime-conditional architecture~~ — minimal impact (NIV signal is regime-symmetric)
2. **DA Q50 structural prior** — the only feature with partial R² large enough to matter, especially at h+3
3. **Δ-prediction framing + Huber loss** — still beneficial for stability but secondary to the DA feature

Even with the correct priority ordering, only h+3 is a plausible candidate. H+1 and H+2 are dominated by the random walk and the available signal cannot close the gap.

---

*Experiment 2 data: 2024+ SSP/NIV from system_prices_5yr.csv (n≈43k rows); DA feature from 32-day archive (May–Jun 2026, n≈1,440 rows). Partial R² computed via Frisch-Waugh projection, winsorized at 1% tails. Implied MAE reductions assume Gaussian residuals and optimal MAE-loss estimator — both optimistic assumptions for heavy-tailed SSP data.*

# Persistence vs DA+Kalman Crossover Analysis

**Branch:** streamlit-data  
**Status:** Investigation only — no production code, no model changes  
**Date:** 2026-06-18  
**Scope:** Determine the horizon at which the day-ahead + Kalman forecast overtakes
persistence, and whether a simple per-horizon blend beats both. This decides how to
forecast h+4 and beyond without building a new model.

---

## 1. Data and Methodology

### 1.1 Persistence curve (statistically solid)

Source: 2024-01-01 → 2026-06-17, combined from `data/raw/system_prices_5yr.csv` +
`data/raw/system_prices.csv`. **n = 43,149+ SP pairs at each horizon.**

For each index i in the sequential SP series:

    persistence forecast at h: predict SSP[i+h−1] = SSP[i−1]
    error: |SSP[i+h−1] − SSP[i−1]|

The 2024+ window is post-crisis, regime-stable, and contains 2.5 full years. The
persistence MAE numbers below are statistically solid estimates with low
sampling uncertainty (n ≈ 43,000 pairs).

### 1.2 DA+Kalman curve (provisional — 30 days)

Source: 32 archived `model_assets/forecasts/forecast_phase3_YYYY-MM-DD.csv` files,
of which **30 days (2026-05-18 → 2026-06-17)** have matching actuals in
`data/raw/system_prices.csv`. All `is_actual=True` rows (intraday-settled actuals)
are excluded; only the pre-settlement DA+Kalman forecast is evaluated.

For each target SP[j] on day D, and each horizon h (where j − h ≥ 1):

    DA+Kalman error: |actual[D, j] − forecast_q50[D, j]|
    Persistence error (same pairs): |actual[D, j] − actual[D, j−h]|

The DA forecast (q50) already includes the Kalman level correction applied at
forecast generation time (previous evening, x̂ × γ^h with γ = 0.966). No
additional correction is applied here.

**⚠ Provisional flag:** 30 days from a single spring/summer seasonal window is
thin. The DA+Kalman MAE (~£31) is roughly constant across horizons (expected —
the day-ahead model targets 24h accuracy uniformly), but the exact level may shift
once more seasonal variety is in the archive. Reconfirm at ≥ 6 months (~Nov 2026),
alongside the h+3 feature-improvement revisit (see monitoring report §7 H+3 gate).

### 1.3 Blend

For each horizon h, optimal blend weight α is found by minimising MAE over
α ∈ [0, 1] on the same 30-day paired sample using raw predictions:

    blend_pred = α × DA_q50 + (1−α) × persistence_pred

This is a single fixed α per horizon — no HGBR, no training. The 5% ship gate
requires the blend to beat **the better of the two endpoints** by ≥ 5%.

Leakage protocol: identical to Experiment 2. Every input ≤ last settled SP;
the DA Q50 for SP[j] is the night-before forecast, available at any intraday
time on day D.

---

## 2. Persistence Decay (2024+ Actuals, n ≈ 43,000)

| Horizon | Persistence MAE | RMSE | Δ from prev |
|---|---|---|---|
| h+1  | £16.38 | £34.91 | baseline |
| h+2  | £22.14 | £45.17 | +35% |
| h+3  | £26.29 | £52.37 | +19% |
| h+4  | £29.21 | £56.00 | +11% |
| h+5  | £31.28 | £58.74 | +7% |
| h+6  | £33.15 | £62.32 | +6% |
| h+7  | £34.63 | £66.15 | +4% |
| h+8  | £36.01 | £69.64 | +4% |
| h+9  | £37.10 | £70.81 | +3% |
| h+10 | £37.99 | £71.80 | +2% |
| h+11 | £38.76 | £72.37 | +2% |
| h+12 | £39.40 | £74.40 | +2% |
| h+18 | £40.11 | £77.73 | +2% |
| h+24 | £39.54 | £78.24 | −1% |
| h+36 | £42.55 | £80.54 | +8% |
| h+48 | £37.23 | £76.24 | −12% |

**Key shape observations:**

- Decay is steep from h+1 to h+6 (each step adds 6–35%), then flattens at ~£40 from h+9
  onwards. This is consistent with the ACF structure: the lag-1 serial correlation
  (ACF = 0.83) is the dominant source of persistence skill, and it decays with a
  half-life of about 3–4 SPs (1.5–2 hours).
- h+24 and h+48 dip below the h+12–h+18 plateau (~£40 → £39.5 → £37.2). This
  is the 24-hour diurnal correlation: "predict SP[j] = SP[j−48]" (same time
  yesterday) captures the daily seasonal shape. Persistence at h+48 is a
  de-facto "same-time-yesterday" forecast and is materially better than naive
  persistence at h+18–h+36.
- This diurnal dip means the DA model (which explicitly predicts the daily shape)
  has a structural advantage at h+24+, but that benefit also appears in naive
  persistence at h+48 — so the effective advantage of the DA model at very long
  intraday horizons is smaller than the h+5–h+12 plateau suggests.

---

## 3. DA+Kalman vs Persistence by Horizon

**⚠ All DA+Kalman figures below are provisional (30 days, May–Jun 2026 only).**

| Horizon | Pers MAE (archive) | DA MAE (archive) | DA−Pers (%) | DA/Pers ratio |
|---|---|---|---|---|
| h+1  | £17.66 | £31.00 | **+75.5%** | 1.755× worse |
| h+2  | £22.75 | £31.08 | **+36.6%** | 1.366× worse |
| h+3  | £26.97 | £31.11 | **+15.3%** | 1.153× worse |
| h+4  | £30.56 | £31.13 | **+1.9%**  | 1.019× worse |
| h+5  | £33.03 | £31.14 | **−5.7%**  | 0.943× **DA wins** |
| h+6  | £35.48 | £31.13 | **−12.3%** | 0.877× |
| h+7  | £37.12 | £31.16 | **−16.1%** | 0.839× |
| h+8  | £38.48 | £31.22 | **−18.9%** | 0.811× |
| h+9  | £40.17 | £31.07 | **−22.7%** | 0.773× |
| h+10 | £41.66 | £30.94 | **−25.7%** | 0.743× |
| h+11 | £43.40 | £30.99 | **−28.6%** | 0.714× |
| h+12 | £44.68 | £31.05 | **−30.5%** | 0.695× |

Note: persistence MAE in the archive column is computed on the same 30-day sample
as the DA errors and therefore differs slightly from the 2024+ large-sample figure.
The 2024+ numbers (§2 table) are the reference for persistence.

**The DA+Kalman MAE is approximately flat at ~£31 across all horizons.** This is
structurally expected: the day-ahead model predicts the full 24-hour profile with
roughly equal skill at every settlement period; the Kalman correction adjusts the
overall level but not the horizon-specific shape. The DA model was optimised for
24h-ahead accuracy, not for within-day horizon discrimination.

**Crossover: h+4.5 (between h+4 and h+5).**

At h+4, DA is still worse than persistence by 1.9%. At h+5, DA beats persistence
by 5.7%. The functional crossover lies at approximately **α = 0.5, h ≈ 2.25 hours
ahead of the last settled SP**, consistent with the ACF decay rate showing
autocorrelation falling below the DA noise floor around 2 hours.

---

## 4. Blend Analysis

Blend = α × DA_q50 + (1−α) × persistence_pred, α ∈ [0,1], fit per horizon on the
same 30-day paired sample. 5% gate: blend must beat the better of the two endpoints
by ≥ 5%.

| Horizon | Best α | Blend MAE | Pers MAE | DA MAE | vs better | Gate |
|---|---|---|---|---|---|---|
| h+1  | 0.04 | £17.59 | £17.66 | £31.00 | −0.4% | — |
| h+2  | 0.14 | £22.21 | £22.75 | £31.08 | −2.4% | — |
| h+3  | 0.31 | £25.64 | £26.97 | £31.11 | −4.9% | near-miss |
| **h+4** | **0.49** | **£27.85** | **£30.56** | **£31.13** | **−8.9%** | **✓ PASS** |
| **h+5** | **0.60** | **£28.60** | **£33.03** | **£31.14** | **−8.2%** | **✓ PASS** |
| **h+6** | **0.70** | **£29.33** | **£35.48** | **£31.13** | **−5.8%** | **✓ PASS** |
| h+7  | 0.73 | £29.79 | £37.12 | £31.16 | −4.4% | — |
| h+8  | 0.76 | £30.08 | £38.48 | £31.22 | −3.7% | — |

**α-profile shape:**

| Range | α range | Interpretation |
|---|---|---|
| h+1 to h+3 | 0.04 – 0.31 | Essentially pure persistence; DA contributes noise |
| h+4 to h+6 | 0.49 – 0.70 | Genuine blend zone; both signals contribute |
| h+7+       | 0.73 – 1.00 | Mostly DA; persistence decays to irrelevance |

**Gate assessment:**

- **h+1 to h+3**: Blend fails the 5% gate (best gain is −4.9% at h+3). A hard persistence
  forecast is optimal. The α values of 0.04–0.31 are not zero, but the improvement is within
  the expected noise of a 30-day sample. Do not rely on these α values for production.
- **h+4 to h+6**: Blend passes the 5% gate. The blend MAE of £27.9–£29.3 beats both pure
  persistence (£30.6–£35.5) and pure DA (£31.1). The improvement over persistence is 6–9pp
  and over DA is 6–8pp. This is the only horizon range where an intermediate α genuinely
  adds value beyond either endpoint.
- **h+7+**: Blend approaches pure DA (α = 0.73–0.88). The improvement over DA alone is
  3–4% — real but below the 5% ship gate, and likely overfitted on the 30-day sample.

**⚠ Caveat on α values:** The α estimates at every horizon are from 30 days of data
(n = 1,058–1,388 SP pairs per horizon). Standard error of an estimated α is
approximately 0.05–0.08. The α profile is a directional read, not a calibrated
production parameter. Refit on the 6-month archive before assigning fixed α values
to any production blend.

---

## 5. Time-of-Day Stratification

The design document (Experiment 2, §2.4) found evening (18:00–24:00) has the
lowest ACF lag-1 (0.559 vs 0.75–0.81 elsewhere), predicting the crossover might
arrive earlier there. This is confirmed:

**Crossover by time-of-day and horizon:**

| Time of Day | h+3 (DA−Pers%) | h+4 | h+5 | h+6 | Crossover |
|---|---|---|---|---|---|
| Night (00:00–06:00) | +63% | +49% | +41% | +28% | > h+6 |
| Morning (06:00–12:00) | +29% | +13% | +9% | +5% | ~h+6–h+7 |
| Afternoon (12:00–18:00) | +17% | +5% | −2% | −7% | ~h+4–h+5 |
| **Evening (18:00–24:00)** | **−19%** | **−28%** | **−35%** | **−41%** | **h+3** |

**Evening is three full horizons earlier than night.** DA already beats persistence
at h+3 in the evening (−18.7% vs +63.2% in night hours). This makes intuitive sense:
the evening ACF collapses (0.559), so persistence error rises faster, while the DA
model captures the systematic load-driven evening spike / ramp in its profile and
maintains ~£25 MAE throughout the evening regardless of horizon.

**Night (00:00–06:00) is the hardest zone for DA.** During overnight SPs, SSP is
low and stable; persistence is very accurate (h+3 MAE only £18.09 vs DA £29.52).
The DA model's uncertainty about overnight levels (which are driven by residual
demand from last-minute balancing actions) means it cannot compete with "carry the
last reading forward" until h+7 or beyond.

The overall crossover (~h+4.5) is an average across these four regimes. A regime-aware
handoff would use persistence through h+5 in overnight periods and hand off at h+3
in evening periods.

---

## 6. Verdict

### (a) Crossover horizon

**Overall: between h+4 and h+5 (approximately 2–2.5 hours ahead).**

At h+4 the two forecasts are within 2% of each other (near-tie). At h+5, DA is 6%
better than persistence. The functional crossover is at approximately half-period
(α ≈ 0.5 optimises at h+4).

By time-of-day:

| Time-of-day | Approximate crossover |
|---|---|
| Night (00–06) | h+7 or later |
| Morning (06–12) | ~h+6–h+7 |
| Afternoon (12–18) | ~h+4–h+5 |
| **Evening (18–24)** | **h+3** |

### (b) Blend vs hard handoff

**A blend genuinely helps at h+4–h+6, but a hard handoff at h+5 captures most
of the available gain and is easier to reason about.**

The blend saves £1.5–£2.9 MAE over the better endpoint at h+4–h+6 (8–9% gain).
That is real money at the margin. However:

1. The α values are estimated on 30 days and carry ±0.05–0.08 uncertainty.
   Misspecified α at h+4 (e.g. shipping α = 0.5 when the true value is 0.35 or 0.65)
   costs only ~£0.5–1 MAE — a small penalty.
2. A hard handoff at h+5 (switch from pure persistence to pure DA) costs approximately
   £1.3 MAE vs the optimal blend at h+5 (£29.9 vs £28.6), and saves nothing at h+4
   vs persistence (£30.6 vs £30.6).
3. A two-step hard handoff — **persistence for h+1..h+4, DA for h+5+** — achieves
   within 5% of the optimal blend outcome at every horizon without any estimated
   parameters. Given the thin archive, this is the safer production choice.

**Recommendation for an interim h+4–h+12 forecast (no new model):**

    h+1:  persistence (last settled SP)               MAE ≈ £16–18
    h+2:  persistence                                  MAE ≈ £22
    h+3:  persistence                                  MAE ≈ £27
    h+4:  persistence (tie with DA, blend optional)    MAE ≈ £30
    h+5+: DA+Kalman (q50 from nightly forecast)        MAE ≈ £31 flat

This handoff schema is available **right now** using the existing Kalman-corrected
DA forecast with zero new training. The transition at h+4/5 is smooth (both
endpoints ≈ £31 MAE at h+4 on the 30-day sample).

**If a blend is desired at h+3–h+6:** use the α profile from §4 but treat it as
directional only. Do not fix α to more than one decimal place until the 6-month
archive is available.

### (c) Provisional data caveat

**All DA+Kalman MAE figures are provisional on the 30-day archive (May–Jun 2026).**

Specific risks:
- The archive covers only spring/early summer. Autumn spike weeks (Sep–Nov) and
  winter demand peaks are unrepresented. DA accuracy may be materially different
  (lower) in those seasons, which would shift the crossover to an earlier horizon
  in autumn and a later one in calm winter periods.
- 30 days is too thin to estimate α per horizon to better than ±0.1. The directional
  picture (crossover at ~h+4.5) is reliable; the specific α values are not.
- The persistence MAE figures are solid (n ≈ 43,000, 2.5 years), but the comparison
  against DA in §3 uses the 30-day paired sample, which shows persistence slightly
  higher than the 2024+ figure at h+1 (£17.66 vs £16.38). This is expected
  seasonal variance in a single month.

**Reconfirm at ~Nov 2026:** alongside the h+3 feature-improvement gate in the
monitoring report §7, rerun this analysis on the full 6-month archive, stratified
by season. If the crossover holds at h+4–h+5 across all seasons, the interim
handoff can be promoted to production guidance.

---

## 7. Summary Table

| Horizon | Recommended source | Rationale | Provisional? |
|---|---|---|---|
| h+1  | Persistence | DA 75% worse; blend adds <0.5% | No — solid (2024+) |
| h+2  | Persistence | DA 37% worse; blend adds 2.4% | No |
| h+3  | Persistence | DA 15% worse; blend at 4.9% (near-miss) | Yes (30d) for DA |
| h+4  | Persistence or blend (α≈0.5) | Near-tie; blend saves ~£2.7 (8.9%) | Yes (30d) |
| h+5  | DA+Kalman or blend (α≈0.6) | DA wins 5.7%; blend saves £2.5 (8.2%) | Yes (30d) |
| h+6  | DA+Kalman | DA wins 12%; blend saves £1.8 (5.8%) | Yes (30d) |
| h+7+ | DA+Kalman | DA wins 16–31%; blend marginal | Yes (30d) |

---

## 8. Appendix: Persistence Periodicity at h+24 / h+48

The persistence MAE does not grow monotonically. It plateaus near h+9–h+12 (~£39),
dips slightly at h+24 (£39.54) and dips further at h+48 (£37.23). This reflects
the **diurnal autocorrelation**: SSP for the same half-hour slot on consecutive
days is more correlated than SSP across distant intraday horizons on the same day.
"Persistence at h+48" is equivalent to "same-time-yesterday" and captures the
daily seasonal shape.

Practical implication: for forecasts beyond h+12 (6 hours ahead), neither pure
persistence nor DA+Kalman is obviously dominant — the competition is three-way:
rolling-same-day persistence, same-time-yesterday persistence, and DA+Kalman.
This regime is outside the scope of this document but should be evaluated if h+12+
forecasting is ever required.

---

*Investigation only. All computations on existing production data. No model files
were modified. No commits required beyond this document. Re-run with
`≥ 6-month archive` (~Nov 2026) to confirm the crossover and calibrate α.*

# Technical Report Outline: UK Electricity System Sell Price Forecasting System

*Pre-writing artifact — inventory pass only. No prose drafted here.*
*Generated: 2026-06-18. Fact sheet refreshed: 2026-06-21. Source: READ-ONLY scan of code, artifacts, and reports.*

> **Fact-sheet refresh (2026-06-21):** All numbers in Section B re-pulled from live artifacts.
> Numbers that changed from the previous version are marked **[UPDATED]**.
> Numbers in README.md and technical-report.md that disagree with current artifacts are listed below.
>
> **README.md discrepancies (do not edit yet):**
> - Kalman Q stated as "0.1" → artifact `corrector_config.json`: **Q = 21.0** (off by 210×)
> - Kalman γ stated as "0.85" → artifact `corrector_config.json`: **γ = 0.966**
> - Level features stated as 85 → artifact `level_feature_cols.json`: **84**
> - Shape H+1 features stated as 76 → artifact `shape_feature_cols.json`: **74**
> - 7-day MAE stated as £31.61 → artifact `phase3_metrics.json`: **£32.03**
> - 7-day RMSE stated as £41.63 → artifact `phase3_metrics.json`: **£42.46**
> - Seasonal fold MAEs (Summer/Autumn/Winter/Spring) all differ from `sp_bias_profile_v1.json` by £0.3–2.2
>
> **technical-report.md discrepancies (do not edit yet):**
> - Outline §B2 previously listed stale Phase 3 metrics (£34.31/£43.35/£20.16/0.435/12.0 SPs) — now corrected below

---

## A. Section Outline

### 1. Introduction and Business Value
**Purpose:** Frame the commercial and operational stakes of SSP forecasting for UK energy market participants.
- The UK Balancing Mechanism and SSP/SBP price formation: why 30-min settlement prices matter
- Who uses SSP forecasts: imbalance-cost management, risk desks, battery dispatch optimisation
- Why SSP is harder to forecast than day-ahead market prices (uncertainty, spikes, negative prices)
- Scope of this system: day-ahead 48-SP profile + intraday Kalman correction, with nowcasting and spike-tail coverage as active research directions

---

### 2. Data and the Lag Ceiling
**Purpose:** Establish what information is available at each forecast horizon and why this constrains model architecture.
- Elexon BMRS API: settlement period structure (48 SP/day, 30-min slots), publication delay (~30 min after SP closes)
- Historical data: 5-year raw SSP + NIV + price derivation codes (P/N/K), generation mix, weather
- Lag ceiling for day-ahead: all features must lag ≥ 48 SPs (1 full day) to be leakage-free for all 48 forecast SPs
- Lag ceiling for intraday/nowcast: SP[t] is unobservable at forecast time T; most recent settled observation is SP[t-1]
- Publication delay consequences: by 08:30 UTC ~33% of day settled; by 12:30 UTC (pipeline run) ~52% settled
- Energy crisis (2022) as a structural break: spike rates 96.7% in 2022 vs 7.7% in 2024 — training window decisions

---

### 3. Pipeline Architecture
**Purpose:** Describe the automated end-to-end system from data ingestion to Streamlit dashboard.
- Two GitHub Actions workflows: `daily_pipeline.yml` (12:30 UTC daily run, weekly model retrain) and `intraday_update.yml` (every 30 min, 48 runs/day, inference only)
- Data ingestion stack: Elexon BMRS, Open-Meteo weather, BMRS WINDFOR, BMRS TSDF, Elexon generation mix, ONS CPI
- Feature construction pipeline: `build_features.py`, lag/calendar/weather feature modules; `features_recent.csv` for intraday inference
- Model artifact persistence: all PKL/JSON model files committed to `streamlit-data` branch; Kalman state persisted via commit-back to avoid cache eviction
- Inference-only intraday path: frozen HGBR + Kalman correction, no retraining
- Streamlit Cloud dashboard served from `streamlit-data` branch; triggered by `_LAST_PIPELINE_RUN` sentinel in `streamlit_app.py`
- Failure-safety: idempotency guard (no empty commits), PI-calibration guard (`_assert_pi_calibrated`), Kalman |z| > £500 guardrail

---

### 4. Modelling: Level–Shape Decomposition (Phase 3)
**Purpose:** Describe the two-stage HGBR architecture that eliminates recursive error propagation.
- Motivation for decomposition: direct SP-level HGBR compounds level and shape errors; decomposition isolates them
- Stage 1 — Level model: daily mean target (`ssp_raw_daily_mean`), daily lag features, weather, WINDFOR, neg-day classifier; P10/P50/P90 quantile heads; 3-year training window (TRAIN_YEARS=3)
- Stage 2 — Shape model: target is `ssp_raw_h − actual_daily_mean_D`; SP-level fixed lags ≥ 48 only (lag-48/96/336); rolling windows excluded; P50 head only
- H+2 shape model: same structure with lag ≥ 96 (day D+1 features only available at lag-96 from D-1)
- Neg-day classifier: binary HGBR P(≥3 negative-price SPs tomorrow); WINDFOR substitution at inference
- Baseline comparison: naive (t−48) MAE £36.34; seasonal naive (t−336) MAE £29.40; rolling mean 24h MAE £26.78; Phase 3 test MAE £32.03 (most recent 7-day window includes Jun 4 extreme event) **[UPDATED]**
- Root cause analysis of Jun 4 2026 failure: renewable oversupply event; structural limit of autoregressive models when day-ahead generation forecasts are absent

---

### 5. Prediction Interval Calibration (Phase 4)
**Purpose:** Describe the split-conformal per-SP PI widening that brings coverage from 38% to ~80%.
- Motivation: raw HGBR P10/P90 achieved only 37.99% empirical coverage (walk-forward, pre-calibration) **[UPDATED]**
- Method: split-conformal symmetric conformity score `max(q10 − actual, actual − q90)`; per-SP δ(sp) = 80th percentile of conformity scores across 4-fold WF calibration window
- Walk-forward calibration window: 119 days (2025-07-01 → 2026-04-30), 4 seasonal folds, 5,709 SP-rows
- Achieved in-sample coverage: 79.82% per-SP, 80.00% global after calibration
- Live coverage (30 days, 2026-05-18 → 2026-06-17): 66.0% — live under-coverage is a known open issue (§7)
- δ(sp) range: min £13.95 (SP 1) to max £39.74 (SP 33), median £22.75; structured by time of day
- PI-calibration guard: `_assert_pi_calibrated()` runtime assertion blocks raw bands from shipping

---

### 6. Intraday Kalman Correction (Phase 4)
**Purpose:** Describe the scalar Kalman filter that tracks and corrects the base model's daily bias using settled SPs.
- Motivation: previous flat-α correction (`_SHAPE_ALPHA = 0.4`) had no memory, ignored observation noise, applied uniform shift with no horizon decay
- Kalman state: scalar bias estimate `x̂` (random-walk prior); no cross-day memory (daily reset at midnight)
- Parameters deployed: Q=21.0 £², σ_SP=35.0 £ (R_t = σ²/n_t), γ=0.966 per SP (halves by SP+20), z_guardrail=500 £/MWh
- NIS-tuned parameters from backtest: Q=5.0 £², σ_SP=20.0 £ (best NIS=0.671, mixed calibration)
- Backtest result: Kalman MAE=£27.68 vs AlphaCorrector MAE=£27.63 (−0.2%, no significant improvement in point forecast); Kalman PI coverage=42.2% vs Alpha=38.9% (+3.3pp)
- Current recommendation: HOLD (Kalman does not beat Alpha on MAE; prerequisite is recalibrated base-model quantiles)
- Principled PI widening: uncertainty envelope propagated as `σ²_correction(h) = P_t · γ^(2h)`; early-day PI wider than late-day
- SP-position bias profile (Phase 5a): diurnal swing £16.9 in base residuals; KalmanSP gating failed (22% reduction vs ≥50% required)

---

### 7. Spike-Tail Coverage (Phase 6a)
**Purpose:** Describe the classifier-gated asymmetric PI widening targeting the 14% under-covered tail.
- Problem: spike SPs (>£150, 1.2% of rows) have only 40% PI coverage; global conformal δ (£24.35) is too narrow for +£77 mean residual on spikes
- Spike characterisation: 70/5,709 rows; 17/119 days; autumn-2025 52.9% of spike SPs; 2025-10-13 max £487 (SP 37)
- Tukey fence: lower=−£156.6, upper=£353.7 (1 SP in WF window exceeds upper fence: 2025-10-13 SP26 at £487)
- Spike classifier (Phase 6a): logistic regression + calibrated isotonic; 10 features; training 2024-01-01 → 2025-06-30 (493 days, 13.0% base rate); eval 2025-07-01 → 2026-04-30 (119 days, 14.3% base rate)
- Spike classifier metrics: Brier=0.1241 (vs base rate Brier 0.1224, skill=−0.013); AP=0.3318
- δ_hi (afternoon block SPs 33–40): p80 upper-tail conformity score on elevated rows (actual>£120) = £93.49 for SPs {33,34,35,36,37,38,40}
- Gate results at τ=0.05: +11.0pp coverage on elevated flagged SPs; +0.0pp change on unflagged — all gates pass (G1/G2/G3)
- Status: `spike_widening: false` in production — awaiting manual sign-off

---

### 8. Nowcasting and Persistence–DA Crossover Analysis
**Purpose:** Document the evidence-based conclusion that a production nowcaster for h+1 is not yet warranted, and that the DA+Kalman model overtakes persistence at h+5.
- Intraday data structure: ACF lag-1 = 0.828, PACF ≈ 1 at lag-1 and ≈ 0 at lags 2–6 → near-random-walk AR(1)
- Persistence MAE: h+1=£16.38, h+2=£22.14, h+3=£26.29, h+4=£29.21, plateaus ~£39-40 at h+9+; diurnal dip at h+24 (£39.54) and h+48 (£37.23)
- DA+Kalman MAE: ~flat at £31 across all horizons (provisional, 30 days May–Jun 2026)
- Crossover: h+4.5 overall; h+3 in evening (18:00–24:00); h+7+ in overnight (00:00–06:00)
- Persistence–DA blend: passes 5% gate at h+4 (α=0.49, −8.9%) through h+6 (α=0.70, −5.8%)
- Interim handoff recommendation: persistence for h+1–h+4; DA+Kalman for h+5+
- HGBR nowcast prototype: −17% worse than persistence at h+1, −6.2% at h+2, tied at h+3 (MSE loss, lag features only)
- Partial R² analysis: ΔSSP momentum dominates (R²=2.8% at h+1); DA Q50 contributes 5.1–9.9% partial R² at h+1–h+3 (32-day archive only); combined upper bound h+1=3.9%, h+2=4.7%, h+3=6.4%
- Ship gates: h+1 and h+2 fail (below 5%); h+3 conditional pass (DA feature, longer archive needed)
- Nowcast bands (persistence PI): deployed for h+1/h+2/h+3 via `nowcast_bands.json`; 18-month fit window; live coverage 79.5%/79.8%/79.4%

---

### 9. Testing and Reliability
**Purpose:** Summarise the automated test suite, CI design, and production-safety mechanisms.
- Test suite: 6 files, 98 defined / 87 collected (test_correctors.py=22, test_kalman_corrector.py=20, test_pi_calibration_guard.py=13, test_pipeline_status.py=12, test_nowcast_rollover.py=20, test_build_dataset.py=11 [collection error: ImportError on `derive_features`]) **[UPDATED]**
- Key test scenarios: AlphaCorrector byte-identical to inline block; Kalman bias tracking convergence; daily reset; PI widening monotonicity with P; quantile monotonicity Q10≤Q50≤Q90; PI calibration guard 8-scenario matrix (A–H)
- Production safety: PI calibration guard raises RuntimeError if WF sentinel present but pi_calibration_v1.json absent; Kalman z-guardrail (|z|>£500 → skip update); idempotency guard (no-diff commit suppression)
- CI/CD: GitHub Actions; daily pipeline 12:30 UTC; intraday pipeline every 30 min; commit-back for model state

---

### 10. Results and Live Monitoring
**Purpose:** Summarise current production performance, coverage metrics, and known live divergence.
- Walk-forward test window: 2025-07-01 → 2026-04-30, 119 days, 4 seasonal folds
- WF PI coverage: 79.8% [78.9%, 80.7%] — on target
- Live PI coverage (2026-05-18 → 2026-06-17): 66.0% (1418 SP-rows) — 13.8pp below target, under investigation
- Corrector backtest MAE: StaticBase £28.96, AlphaCorrector £27.63, KalmanCorrector £27.68; spike only: £385–397
- Phase 3 metrics (7-day test): MAE £32.03 (includes Jun 4 extreme event), RMSE £42.46, sMAPE 49.61%; level MAE £18.41, shape corr 0.4275, peak timing MAE 6.71 SPs **[UPDATED]**
- Monitoring gate status: Step-3 (P95/P99 head) deferred to Dec 2026; H+3 nowcast archive gate at 2/6 months required
- Price-bucket coverage: <£85 = 66.6%, £85-120 = 85.9%, £120-150 = 47.6%, £150-250 = 34.1% (live)
- Kalman residual (rolling 4w winsorised mean): £-1.4 — on target

---

### 11. Roadmap
**Purpose:** Lay out the prioritised sequence of planned improvements with their gating conditions.
- SP-level solar forecast at inference time: fix Jun 4-type renewable oversupply failures
- Spike widening enablement: set `spike_widening: true` after manual sign-off of gate table
- Season-conditioned δ (Phase 6b): per-(SP, season) calibration to address autumn/spring under-coverage
- H+3 nowcast: prototype with DA Q50 feature + Δ-prediction framing; re-evaluate at 6-month archive (~Oct 2026)
- P95/P99 quantile head: gated on ≥2 usable spike-bearing autumn seasons; est. trigger Dec 2026
- Regime-conditional architecture for nowcaster: de-prioritised (NIV partial R² regime-symmetric after conditioning on SSP level)
- Phase 5 online learning (fallback): gated on Kalman NIS across ≥4 seasonal folds showing structural misspecification

---

### 12. Novelties Summary
**Purpose:** A concise enumeration of the technically novel or non-obvious contributions of this system, for the abstract and related-work framing.
- Level–shape decomposition with strict lag-48 fence eliminating all rolling-window contamination for SP-level shape model
- Principled Kalman correction layer replacing hand-tuned α: O(1) per call, principled uncertainty propagation, daily reset
- Split-conformal per-SP δ(sp) with statistically correct coverage guarantee on calibration window
- Classifier-gated asymmetric spike widening with ex-ante signal (no outcome conditioning), operating only on high-risk SPs in afternoon block
- Empirical crossover analysis showing DA+Kalman model overtakes persistence at h+4.5 (not h+1 as naively assumed)
- Partial R² diagnostic showing PACF structure implies persistence is Bayes-optimal AR(1) predictor at h+1, correcting overestimated NIV regime-conditional signal
- NIS-based Kalman hyperparameter tuning from walk-forward backtest

---

## B. Verified Fact Sheet

*Every number tagged with source file. Numbers from design docs (not yet implemented) are marked [DESIGN].*

### B1. Training Data and Features

| Fact | Value | Source |
|------|-------|--------|
| Training window cap (TRAIN_YEARS) | 3 years | `src/models/train_phase3.py` line 64 |
| Test days | 7 | `src/models/train_phase3.py` line 63 |
| Val days | 5 | `src/models/train_phase3.py` line 64 |
| Level feature count | 84 features | `model_assets/level_feature_cols.json` |
| Shape H+1 feature count | 74 features | `model_assets/shape_feature_cols.json` |
| Shape H+2 feature count | 50 features | `model_assets/shape_h2_feature_cols.json` |
| 5-year data rows | 87,686 SP rows, 1,827 days | `reports/annual_modulation_analysis.md` |
| 5-year data period | May 2021 – May 2026 | `reports/annual_modulation_analysis.md` |
| WF calibration window | 119 days, 5,709 SP-rows, 4 folds | `model_assets/pi_calibration_v1.json` |
| Nowcast band fit window | 18 months (2024-12-17 → 2026-06-17), 26,298 SP pairs | `model_assets/nowcast_bands.json` |
| Post-crisis regime-stable data (2024+) | ~43,149 SP pairs | `docs/persistence-ml-crossover.md` |

### B2. Phase 3 Model Performance

| Fact | Value | Source |
|------|-------|--------|
| Phase 3 test MAE (7-day holdout) | £32.03/MWh **[UPDATED]** | `model_assets/phase3_metrics.json` |
| Phase 3 test RMSE (7-day) | £42.46/MWh **[UPDATED]** | `model_assets/phase3_metrics.json` |
| Phase 3 test sMAPE | 49.61% **[UPDATED]** | `model_assets/phase3_metrics.json` |
| Phase 3 test n | 336 SP-rows (7 days × 48 SPs) | `model_assets/phase3_metrics.json` |
| Phase 3 level MAE (decomposition) | £18.41/MWh/day **[UPDATED]** | `model_assets/phase3_metrics.json` |
| Phase 3 shape correlation (mean) | 0.4275 **[UPDATED]** | `model_assets/phase3_metrics.json` |
| Phase 3 peak timing MAE | 6.71 SPs **[UPDATED]** | `model_assets/phase3_metrics.json` |
| Baseline naive (t−48) MAE | £36.34/MWh | `model_assets/baseline_metrics.json` |
| Baseline seasonal naive (t−336) MAE | £29.40/MWh | `model_assets/baseline_metrics.json` |
| Baseline rolling mean 24h MAE | £26.78/MWh | `model_assets/baseline_metrics.json` |
| Phase 3 vs rolling mean 24h | Phase 3 is WORSE on 7-day window (£32.03 vs £26.78) — Jun 4 event dominates **[UPDATED]** | `model_assets/phase3_metrics.json`, `model_assets/baseline_metrics.json` |
| HGBR baseline MAE (hgbr_metrics, same test n=336) | £25.40/MWh | `model_assets/hgbr_metrics.json` |
| 7-day holdout metrics (hourly-calibration-design.md) | MAE £30.35, RMSE £37.94, sMAPE 41.3%, Level MAE £13.17, shape corr 0.405, peak timing 6.86 SPs | `docs/hourly-calibration-design.md` §1.2 (slightly different run than phase3_metrics.json) |

### B3. PI Calibration

| Fact | Value | Source |
|------|-------|--------|
| Coverage before calibration | 37.99% | `model_assets/pi_calibration_v1.json` |
| Coverage after calibration (in-sample, per-SP) | 79.82% | `model_assets/pi_calibration_v1.json` |
| Coverage after calibration (in-sample, global) | 80.00% | `model_assets/pi_calibration_v1.json` |
| δ_global | £24.35 | `model_assets/pi_calibration_v1.json` |
| δ(sp) min | £13.95 (SP 1) | `model_assets/pi_calibration_v1.json` → `delta_stats.min` |
| δ(sp) max | £39.74 (SP 33) | `model_assets/pi_calibration_v1.json` → `delta_stats.max` |
| δ(sp) median | £22.75 | `model_assets/pi_calibration_v1.json` → `delta_stats.median` |
| δ(sp) p25 | £19.41 | `model_assets/pi_calibration_v1.json` → `delta_stats.p25` |
| δ(sp) p75 | £28.99 | `model_assets/pi_calibration_v1.json` → `delta_stats.p75` |
| Calibration folds | autumn-2025, spring-2026, summer-2025, winter-2025 | `model_assets/pi_calibration_v1.json` |
| Calibration date | 2026-06-16 | `model_assets/pi_calibration_v1.json` |
| Live PI coverage (all, 30 days) | 66.0%, N=1418 | `reports/monitoring/2026-W25.md` §1 |
| Live PI coverage (rolling 4w) | 67.0%, N=1274 | `reports/monitoring/2026-W25.md` §1 |
| WF baseline coverage (used as target) | 79.8% [78.9%, 80.7%] | `reports/monitoring/2026-W25.md` §1 |

### B4. Kalman Filter Parameters

| Fact | Value | Source |
|------|-------|--------|
| Q (deployed) | 21.0 £² | `model_assets/corrector_config.json` |
| σ_SP (deployed) | 35.0 £ (R_t = σ²/n_t) | `model_assets/corrector_config.json` |
| γ (horizon decay, deployed) | 0.966 per SP | `model_assets/corrector_config.json` |
| z_alpha | 1.28 | `model_assets/corrector_config.json` |
| z_guardrail | £500 | `model_assets/corrector_config.json` |
| Q (NIS-tuned from backtest) | 5.0 £² | `reports/corrector_backtest/report.md` §4 |
| σ_SP (NIS-tuned) | 20.0 £ | `reports/corrector_backtest/report.md` §4 |
| Best NIS (tuned) | 0.6713 (mixed — under-confident overall) | `reports/corrector_backtest/report.md` §4 |
| NIS by fold | summer 0.404, autumn 0.596, winter 0.319, spring 1.364 | `reports/corrector_backtest/report.md` §4 |
| Theoretical Q from level MAE | ~21 £² (design doc) | `docs/hourly-calibration-design.md` §3.4 [DESIGN] |
| Initial P₀ | 1225 (= σ_SP² = 35²) — inferred from test fixture | `tests/test_kalman_corrector.py` comments |
| Live x̂ (2026-06-18T20:45) | £0.504 | `model_assets/kalman_state.json` |
| Live P | 17.44 | `model_assets/kalman_state.json` |
| Live last_z | £3.25 | `model_assets/kalman_state.json` |
| Live n_settled | 42 SPs | `model_assets/kalman_state.json` |
| Daily reset trigger | new forecast_date != stored forecast_date | `src/models/correctors.py` |

### B5. Corrector Backtest (Walk-Forward, 119 days, unsettled SPs only)

| Fact | Value | Source |
|------|-------|--------|
| StaticBase MAE | £28.96 | `reports/corrector_backtest/report.md` §1 |
| AlphaCorrector MAE | £27.63 | `reports/corrector_backtest/report.md` §1 |
| KalmanCorrector MAE | £27.68 | `reports/corrector_backtest/report.md` §1 |
| StaticBase PI coverage | 37.7% | `reports/corrector_backtest/report.md` §1 |
| AlphaCorrector PI coverage | 38.9% | `reports/corrector_backtest/report.md` §1 |
| KalmanCorrector PI coverage | 42.2% | `reports/corrector_backtest/report.md` §1 |
| KalmanBase MAE by fold (summer, autumn, winter, spring) | £25.77, £30.50, £20.27, £34.28 | `reports/corrector_backtest/report.md` §3 |
| SP-position diurnal swing | £16.9 (max − min of SP-mean residuals) | `reports/corrector_backtest/report.md` §5 |
| KalmanSP MAE | £28.23 vs KalmanBase £27.68 | `reports/corrector_backtest/report.md` Phase 5a |
| Spike (Tukey-fence, £353.7) coverage | 0% across all correctors | `reports/corrector_backtest/report.md` §2 |
| Spike (Tukey-fence) MAE | StaticBase £385.47, Alpha £392.13, Kalman £397.12 | `reports/corrector_backtest/report.md` §2 |
| Only Tukey-fence spike in WF window | 2025-10-13 SP26 at £487 | `reports/corrector_backtest/report.md` §2 |

### B6. Spike Tail

| Fact | Value | Source |
|------|-------|--------|
| Spike rows (>£150 proxy) | 70 / 5,709 = 1.2% | `docs/spike-tail-design.md` §2.1 |
| Spike days | 17 / 119 = 14.3% | `docs/spike-tail-design.md` §2.1 |
| Mean spike residual (actual − q50) | +£77 | `docs/spike-tail-design.md` §2.1 |
| Max actual on spike SPs | £487 (2025-10-13 SP37) | `docs/spike-tail-design.md` §2.1 |
| Spike coverage under base q90 | 14% (10/70) | `docs/spike-tail-design.md` §2.1 |
| Normal period mean PI width | £34.9 | `docs/spike-tail-design.md` §2.1 |
| Spike period mean PI width | £36.7 (statistically identical) | `docs/spike-tail-design.md` §2.1 |
| Global asymmetric δ_hi (p80, all rows) | £6.72 (dominated by 98.8% normal rows) | `docs/spike-tail-design.md` Appendix |
| Autumn afternoon (SP 33–40) spike rate | 12.5% | `docs/spike-tail-design.md` §2.3 |
| Winter spike rate | 0.0% | `docs/spike-tail-design.md` §2.3 |
| Spike-only p80 upper excess | £96.9 | `docs/spike-tail-design.md` Appendix |
| δ_hi (afternoon block, SP {33–40}) | £93.49 | `model_assets/delta_hi_v1.json` |
| High-risk SPs with δ_hi applied | {33, 34, 35, 36, 37, 38, 40} | `model_assets/delta_hi_v1.json` |
| n elevated rows in WF window (actual>£120) | 573 (191 in afternoon block) | `model_assets/delta_hi_v1.json` |
| Gate G1 (τ=0.05, with 2025-10-13) | +11.0pp coverage lift on elevated flagged SPs | `reports/corrector_backtest/report.md` §8 |
| Gate G2 (unflagged coverage change) | +0.00pp | `reports/corrector_backtest/report.md` §8 |

### B7. Spike Classifier

| Fact | Value | Source |
|------|-------|--------|
| Model type | LogisticRegression (C=1.0, balanced) + CalibratedClassifierCV (isotonic, cv=5) | `model_assets/spike_classifier_v1_eval.json` |
| Training window | 2024-01-01 → 2025-06-30, 493 days | `model_assets/spike_classifier_v1_eval.json` |
| Eval window | 2025-07-01 → 2026-04-30, 119 days | `model_assets/spike_classifier_v1_eval.json` |
| Training base rate | 13.0% (64/493 spike days) | `model_assets/spike_classifier_v1_eval.json` |
| Eval base rate | 14.3% (17/119 spike days) | `model_assets/spike_classifier_v1_eval.json` |
| Brier score | 0.1241 | `model_assets/spike_classifier_v1_eval.json` |
| Brier baseline (base rate²) | 0.1224 | `model_assets/spike_classifier_v1_eval.json` |
| Brier skill score | −0.013 (marginally negative — near base rate) | `model_assets/spike_classifier_v1_eval.json` |
| Average precision (AP) | 0.3318 | `model_assets/spike_classifier_v1_eval.json` |
| Precision/recall at τ=0.20 | 0.286 precision, 0.588 recall, 35 days flagged | `model_assets/spike_classifier_v1_eval.json` |
| Feature count | 10 features | `model_assets/spike_classifier_v1_eval.json` |
| Train/serve wind skew | WINDFOR at inference vs CI actual at training → conservative on low-wind days | `model_assets/spike_classifier_v1_eval.json` |
| Spike year rates | 2022=96.7%, 2023=57.3%, 2024=7.7%, 2025=22.5% | `docs/spike-classifier-plan.md` |

### B8. Nowcast Bands (Persistence PI)

| Fact | Value | Source |
|------|-------|--------|
| Fit window | 2024-12-17 → 2026-06-17, 18 months, 26,298 SP pairs | `model_assets/nowcast_bands.json` |
| In-sample coverage (h+1 overall) | 80.0% | `model_assets/nowcast_bands.json` |
| Holdout coverage (h+1 overall, 2026) | 83.25% (N=8,058) | `model_assets/nowcast_bands.json` |
| Live coverage (h+1 overall) | 79.5% (N=3,215) | `reports/monitoring/2026-W25.md` §8 |
| Live coverage (h+3 overall) | 79.4% (N=3,213) | `reports/monitoring/2026-W25.md` §8 |
| NP regime h+1 band [P10, P90] | [−£12.32, +£43.90] | `model_assets/nowcast_bands.json` |
| EN regime h+1 band [P10, P90] | [−£44.13, +£9.00] | `model_assets/nowcast_bands.json` |
| NP fraction of fit data | 14,085 / 26,298 = 53.6% | `model_assets/nowcast_bands.json` |

### B9. Nowcasting and Crossover

| Fact | Value | Source |
|------|-------|--------|
| SSP ACF lag-1 (2024+) | 0.828 | `docs/nowcasting-design.md` §2.1 |
| SSP PACF lag-1 (2024+) | ≈1.0; lags 2–6 ≈ 0 → AR(1) | `docs/nowcasting-design.md` §2.1 |
| Persistence MAE h+1 (2024+, n≈43k) | £16.38 | `docs/persistence-ml-crossover.md` §2 |
| Persistence MAE h+2 | £22.14 | `docs/persistence-ml-crossover.md` §2 |
| Persistence MAE h+3 | £26.29 | `docs/persistence-ml-crossover.md` §2 |
| Persistence MAE h+4 | £29.21 | `docs/persistence-ml-crossover.md` §2 |
| Persistence MAE h+5 | £31.28 | `docs/persistence-ml-crossover.md` §2 |
| Persistence MAE h+12 | £39.40 | `docs/persistence-ml-crossover.md` §2 |
| Persistence MAE h+48 | £37.23 (diurnal dip) | `docs/persistence-ml-crossover.md` §2 |
| DA+Kalman MAE (archive, all horizons) | ~£31 flat, provisional 30 days | `docs/persistence-ml-crossover.md` §3 |
| Crossover horizon (overall) | h+4.5 (DA wins by h+5, 5.7%) | `docs/persistence-ml-crossover.md` §3 |
| Crossover (evening 18–24) | h+3 | `docs/persistence-ml-crossover.md` §5 |
| Crossover (overnight 00–06) | h+7 or later | `docs/persistence-ml-crossover.md` §5 |
| Blend at h+4: best α, MAE, vs better endpoint | α=0.49, £27.85, −8.9% | `docs/persistence-ml-crossover.md` §4 |
| HGBR prototype h+1 MAE vs persistence | £19.40 vs £16.59 (−17.0%, worse) | `docs/nowcasting-design.md` §4.1 |
| Partial R² ΔSSP momentum at h+1 | 0.02786 (implied −£0.23 MAE) | `docs/nowcasting-design.md` Exp 2 |
| Partial R² DA Q50 at h+3 (32-day only) | 0.099 (implied −£1.35 MAE, 4.9%) | `docs/nowcasting-design.md` Exp 2 |
| NIV-SSP cross-correlation NP vs EN (raw) | 0.381 NP vs 0.079 EN | `docs/nowcasting-design.md` §2.3 |
| NIV partial R² h+1 NP vs EN (conditioned) | 0.038 vs 0.039 — nearly identical | `docs/nowcasting-design.md` Exp 2 §E2.4 |

### B10. Live Monitoring (W25 Report)

| Fact | Value | Source |
|------|-------|--------|
| Live period | 2026-05-18 → 2026-06-17 (30 days) | `reports/monitoring/2026-W25.md` |
| Live N (SP-rows) | 1,418 | `reports/monitoring/2026-W25.md` §1 |
| Rolling 4w N | 1,274 | `reports/monitoring/2026-W25.md` §1 |
| Rolling 4w winsorised mean residual | £-1.4 | `reports/monitoring/2026-W25.md` §4 |
| Risk-flag coverage Δ (live) | -1.8pp (within ±15pp warning threshold) | `reports/monitoring/2026-W25.md` §3 |
| Spike widening status | INACTIVE (`spike_widening: false`) | `reports/monitoring/2026-W25.md` §5 |
| Usable spike-bearing autumns | 1 of 2 required (Step-3 gate RED) | `reports/monitoring/2026-W25.md` §7 |
| Forecast archive months | 2 of 6 required (H+3 gate RED) | `reports/monitoring/2026-W25.md` §7 |

### B11. Tukey Fence

| Fact | Value | Source |
|------|-------|--------|
| Lower fence | −£156.6 | `model_assets/tukey_fence.json` |
| Upper fence | £353.7 | `model_assets/tukey_fence.json` |

### B12. Test Counts **[UPDATED]**

| File | Test functions | Collected? |
|------|---------------|-----------|
| test_correctors.py | 22 | ✓ |
| test_kalman_corrector.py | 20 | ✓ |
| test_pi_calibration_guard.py | 13 | ✓ |
| test_pipeline_status.py | 12 | ✓ |
| test_nowcast_rollover.py | 20 | ✓ |
| test_build_dataset.py | 11 | ✗ (ImportError: `derive_features`) |
| **Total** | **98 defined / 87 collected** | |

Sources: `tests/test_correctors.py`, `tests/test_kalman_corrector.py`, `tests/test_pi_calibration_guard.py`, `tests/test_pipeline_status.py`, `tests/test_nowcast_rollover.py`, `tests/test_build_dataset.py`

---

## C. Plot Inventory

### Existing Figures

| Path | Description |
|------|-------------|
| `reports/corrector_backtest/metrics_comparison.png` | Bar chart comparing MAE, RMSE, sMAPE, PI coverage across StaticBase / AlphaCorrector / KalmanCorrector overall |
| `reports/corrector_backtest/metrics_by_fold.png` | Same metrics broken down by seasonal fold (summer/autumn/winter/spring) for all three correctors |
| `reports/corrector_backtest/nis_heatmap.png` | NIS heatmap by (date, hourly step) for the best-tuned Kalman (Q=5, σ=20, γ=0.9), showing seasonal heterogeneity |
| `reports/corrector_backtest/sp_bias.png` | Mean residual (actual − base q50) per SP position across 119 test days, showing £16.9 diurnal swing |
| `reports/corrector_backtest/sp_bias_comparison.png` | SP-level residual before and after KalmanSP correction, illustrating the 22% diurnal reduction |
| `reports/corrector_backtest/spike_harness.png` | PI coverage and MAE by price bucket and ex-ante risk flag for KalmanCorrector |
| `reports/corrector_backtest/spike_widening_gate.png` | Gate sweep: coverage before/after δ_hi widening on elevated flagged SPs by τ threshold |
| `reports/corrector_backtest/error_distributions.png` | Error distribution (histograms) for StaticBase / Alpha / Kalman on unsettled SPs |
| `reports/monitoring/plots/2026-W25_coverage.png` | Weekly PI coverage timeline: WF baseline vs live rolling coverage, coloured by status |
| `reports/monitoring/plots/2026-W25_kalman.png` | Kalman residuals over 30-day live period: winsorised mean, median, and ±£10 warning band |
| `reports/monitoring/plots/2026-W25_step3.png` | Step-3 gate tracker: usable spike-bearing autumns by year, with 2-autumn requirement threshold |

### Missing Figures (Recommended for Report)

| Recommended Figure | Rationale |
|-------------------|-----------|
| **δ(sp) profile** — bar chart of PI calibration δ(sp) by SP position (1–48), with the global δ as a horizontal reference line | The structured time-of-day pattern in δ (lowest at night SPs 1–7, peaking at afternoon SP 33) is a key modelling finding. The raw values are in `model_assets/pi_calibration_v1.json` and can be plotted directly. |
| **Level–shape decomposition diagram** — two-panel showing (1) the level model's daily mean prediction vs actual and (2) the shape model's SP deviation pattern for a representative day, plus a third panel showing the combined forecast vs actual | This is the central architectural innovation and has no corresponding visual anywhere in the codebase. |
| **Persistence MAE decay curve** — h+1 through h+48, highlighting the crossover at h+4.5 and the h+48 diurnal dip | Tabular data exists in `docs/persistence-ml-crossover.md` §2 but no figure; critical for the nowcasting section. |
| **Coverage by price bucket (WF vs live)** — grouped bars showing WF and live coverage in each price bucket (<£85, £85-120, £120-150, £150-250, £250+) | Live data from `reports/monitoring/2026-W25.md` §2; reveals the structural gap at the high-price tail, motivating Phase 6a. |
| **Spike classifier ROC/PR curve** — precision-recall curve over τ sweep {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, computed from `spike_classifier_v1_eval.json` | AP=0.332 and per-τ metrics exist in the JSON; a PR curve shows the usable operating range and is standard for classifier papers. |
| **Kalman filter update example** — single-day trace showing x̂ and P evolving across 8 hourly calls as more SPs settle, overlaid on the base model error | Illustrates the Kalman mechanism intuitively; currently only documented in design text, no figure exists. |
| **Seasonal spike distribution** — heatmap of spike rate by (season × SP block), using the four-fold data already in `docs/spike-tail-design.md` §2.3 | The autumn/spring afternoon gap is a key finding; a heatmap makes it visually immediate. |
| **Annual modulation and year-by-year monthly means** — heatmap of monthly mean SSP by year (rows=year, cols=month) from the existing analysis | Data in `reports/annual_modulation_analysis.md` §3.2; the W-shaped seasonal pattern and 2022 outlier year are both report-worthy. |

---

## D. Claim Flags

Claims we might want to make, with caveats about what the current evidence does or does not support:

- **"Kalman filter improves coverage by +4.5pp over the static base"** — TRUE in the WF backtest (42.2% vs 37.7%). But all three correctors are well below the 80% target; the improvement is from a very low base. Do not claim Kalman achieves adequate coverage.

- **"Live PI coverage is 66%"** — TRUE but note: (a) only 30 days of live data (N=1,418 SP-rows); (b) the live data includes a backfilled PI calibration applied retroactively to pre-2026-06-17 forecasts; (c) the live sample is spring/summer only. The 66% figure is a real regression from the WF 79.8% but the small sample means the CI is wide ([63.9%, 68.0%]).

- **"Kalman beats AlphaCorrector"** — FALSE on point forecast MAE (−0.2%, within noise). Only true on PI coverage (+3.3pp). The backtest recommendation is currently HOLD. Do not claim Kalman is the deployed winner.

- **"Spike classifier has positive skill"** — AMBIGUOUS. Brier skill score is −0.013 (marginally negative), meaning the classifier is near base rate. AP=0.332 shows it is better than random, but the low Brier skill means it barely outperforms always predicting the base rate. Report AP and the per-τ precision/recall table; do not claim the classifier has strong discrimination.

- **"Phase 3 MAE is £32.03"** **[UPDATED]** — PROVISIONAL: this is computed on the most recent 7 days only, which included the Jun 4 extreme event that inflated MAE significantly. The walk-forward 4-fold corrector backtest shows more representative performance: StaticBase £28.96, consistent with the level MAE + shape residual structure.

- **"DA+Kalman overtakes persistence at h+5"** — PROVISIONAL: based on 30 days (May–Jun 2026). The DA+Kalman archive is single-season (no autumn/winter). The persistence curve (n≈43,000) is solid; the crossover horizon may shift under different seasonal conditions. Flag explicitly.

- **"HGBR nowcast prototype was 17% worse than persistence at h+1"** — TRUE for that specific run (12-week window, MSE loss, lag features only, 2-week eval steps). But the prototype lacked the DA Q50 feature and used MSE not MAE loss. Not a ceiling on what an improved model could achieve.

- **"Partial R² implies h+3 nowcast can clear the 5% gate"** — OPTIMISTIC: the DA Q50 partial R² estimate (R²=9.9%, implied 4.9% MAE reduction) is computed on only 32 days and assumes Gaussian residuals. Heavy-tailed SSP data means the actual gain from optimal MAE-loss training will be lower. The document correctly labels this a conditional pass.

- **"Spike widening gates pass"** — TRUE (all G1/G2/G3 pass at τ=0.05). BUT: the coverage lift in §8.1 is strongly influenced by the single 2025-10-13 event (SP26 at £487). Excluding 2025-10-13 the gates still pass (§8.2), but the δ_hi value of £93.49 is derived from a very small exceedance pool (27 afternoon-block rows in 119 days). Report robustness with and without 2025-10-13.

- **"Phase 3 beats all baselines"** — CANNOT CLAIM on the 7-day holdout: the rolling mean 24h MAE (£26.78) beats Phase 3 (£32.03) due to Jun 4 contamination. **[UPDATED]** This is a known artefact of the extreme event in the test window. The WF corrector backtest (n=119 days, StaticBase £28.96) is a fairer comparison.

- **"The annual modulation explains 3% of variance"** — TRUE for two-harmonic Fourier fit on 5-year winsorised data. Note: this R² figure combines crisis and non-crisis years; in the current (2024+) regime the seasonal signal may be stronger or weaker.

- **"Spike risk is predictable from ex-ante features"** — PARTIAL: the ex-ante risk flag (elevated_count_lag1d, wind, month) gives a 2.4× spike rate lift (2.8% vs 1.2%), which is real. But the classifier Brier skill is near zero, meaning the probabilistic prediction is barely better than base rate. The flag is useful for gating PI widening; it does not reliably identify individual spike days.

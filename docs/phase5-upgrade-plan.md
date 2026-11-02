# Phase 5 Upgrade Plan — UK SSP Forecasting Platform

*Drafted 2026-11-02 with Claude, grounded in the repo's current walk-forward artifacts.*

## Diagnosis (why this ordering)

Current HGBR feature importances: `ssp_lag_1` = 22.0, second place (`niv_lag_1`) = 1.3, everything else ≤0.34. The model is persistence plus small corrections, and **every one of the 91 features is backward-looking** (lags, rolls, calendar, lagged weather outturn). The information that actually sets tomorrow's SSP — day-ahead auction price, forecast wind/demand/margin, outages, gas — is known before delivery and absent from the feature set. That is the headroom. Recent degradation (level MAE £38/day on the Oct test vs £11.7 in Aug) also shows regime sensitivity worth attacking via window/weighting and target transform.

## Assessment of the five candidate ideas

1. **Training-window optimisation — yes, cheap, do first.** Sweep {12, 18, 24, 36, 60} months × sample weighting {uniform, exp half-life 6/12 mo} on the existing 119-day walk-forward harness. Expect single-digit % MAE gains. Prefer exponential weighting over hard truncation: short windows starve the P10/P90 tails of rare spike examples.
2. **Log-returns r = log(p_t/p_{t−30min}) — not as stated.** GB SSP is frequently negative or zero, so log is undefined; and 30-min returns discard the level information MAE is scored on. Salvageable core: replace the hard winsorisation (cap ~£353) with an **asinh target transform** (log-like for large |p|, linear through zero, defined for negatives) for the quantile models. Test in the same sweep as (1).
3. **Alternative data — the most promising lever by far.** Ranked by gain/effort:
   - **WINDFOR + TSDF as first-class training features** (already fetched in `fetch_bmrs_forecasts.py`, currently only used to patch `wind_pct` at inference). Add forecast wind MW, demand MW, margin = TSDF − WINDFOR, and their ramps; train on historic day-ahead vintages to stay leakage-free.
   - **GB day-ahead auction price** (N2EX hourly; or Elexon MID from the same BMRS API as a market-price proxy). The strongest single SSP-level predictor in the EPF literature: SSP ≈ DA price ± imbalance premium.
   - **LOLP / derated margin** (BMRS `LOLPDRM`) — scarcity signal, the canonical spike predictor.
   - **REMIT outage data** (Elexon IRIS) — plant unavailability.
   - Later: day-ahead gas (NBP/TTF proxy — monthly CPI is far too slow to carry fuel costs), interconnector schedules / FR-NL-BE day-ahead prices (ENTSO-E free token).
   - **Skip traffic data** (weak demand proxy already covered by calendar + TSDF). More weather variables only as *forecast* vintages, not more outturn lags.
4. **Neural networks — not the centrepiece; shadow-test later.** On tabular features at this scale, tuned gradient boosting matches or beats DNNs in most EPF benchmarks, and CI free-tier minutes are a real constraint. After the data upgrade, compare offline (TFT/N-HiTS class models); promote via the existing shadow-model panel only if walk-forward MAE improves >5%.
5. **CWoLa — no.** CWoLa solves classification when per-event labels are unavailable (mixed samples with different signal fractions). Here every settlement period gets a ground-truth label the next day; weak supervision would discard information. The underlying goal (regime/spike separation) is already better served by the supervised Phase 6a spike-classifier plan in `docs/spike-classifier-plan.md` and the NP/EN derivation-code regimes.

## Statistical baseline (adopted, extended)

Add a **LEAR-style model** (Lasso-Estimated AutoRegressive — the standard interpretable EPF baseline): one lasso per settlement period on SSP lags {1d, 2d, 7d}, DA price, forecast demand/wind/margin, calendar dummies, asinh transform, rolling re-estimation. Fast enough for daily CI; coefficients are directly interpretable (publish a per-SP coefficient panel on the dashboard). It is also an honest yardstick — if HGBR can't beat LEAR on the same information set, the complexity isn't paying — and a known-good ensemble partner (mean of LEAR + HGBR often beats both).

## Phased execution

- **5a (1–2 wks, offline):** error autopsy on the October degradation (level vs shape, NP vs EN, spike days) + window/weighting sweep + asinh-vs-winsorise sweep on the walk-forward harness.
- **5b (2–4 wks):** data wave 1 — WINDFOR/TSDF features with historic vintages, LOLPDRM, MID/N2EX DA price; rebuild features; re-run walk-forward. Expected largest MAE gain. Zero-leakage rule: only D−1-available vintages.
- **5c (1–2 wks):** LEAR baseline + dashboard model-comparison panel (LEAR vs HGBR vs persistence; rolling 30-day MAE + pinball loss) + coefficient panel; ship the simple ensemble if it wins.
- **5d (research, optional):** offline NN comparison; Phase 6a spike classifier if the autopsy shows spike days dominate the error budget.

## Adoption gate

A change ships only if it passes **one** of the two clauses below, both of which also require **no PI-coverage regression** (overall walk-forward coverage must not drop by more than 0.02):

1. **General clause:** overall walk-forward MAE improves ≥3%, or CRPS/pinball improves ≥5%.
2. **Regime-conditional clause** (for a change targeting a *pre-diagnosed* failure mode): the diagnosis must be written down *before* the validation run (which fold(s) the change should fix, and why the others are expected to tie). The change then passes if (a) MAE improves ≥5% in the targeted fold(s), (b) no non-targeted fold degrades by more than 2%, and (c) coverage holds both overall and in the targeted fold(s). The diluted overall average is reported but not binding.

Rationale: a regime-targeted fix (e.g. the Phase 5a window change) shows up almost entirely in the folds that resemble the diagnosed regime; averaging it across calm folds that are mathematically near-identical under both configs only dilutes the signal. The pre-registration requirement is what keeps this clause from becoming an excuse — the targeted folds are named before the numbers are seen, never after.

**Precedent — Phase 5a window change (2026-11-02):** overall −2.0% (below clause 1) but autumn-2026 (the pre-diagnosed regime fold) −6.8% MAE / −22% level MAE / bias halved, no other fold worse than +1.3%, coverage improved 0.37→0.40. Passed under clause 2 reasoning (adopted before this clause was formalised; this section exists so future cases don't need a judgment call).

**Post-deployment arbiter:** every adoption is provisional until the rolling level-bias / MAE on the monitoring panel confirms it over 1–2 weeks of live operation; if live behaviour contradicts the backtest, revert regardless of the gate outcome.

Add CRPS + per-quantile pinball to the harness so probabilistic skill is tracked alongside MAE.

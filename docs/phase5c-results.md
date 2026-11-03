# Phase 5c Results — LEAR Interpretable Baseline

*2026-11-03. Branch `phase-5c`, merged to `streamlit-data`. Scope per `docs/phase5-upgrade-plan.md`: LEAR baseline, model-comparison panel, coefficient panel, ensemble decision.*

## Model

LEAR (Lasso-Estimated AutoRegressive), the standard interpretable EPF benchmark, adapted to GB half-hourly settlement: **one lasso per settlement period** on asinh-robust-scaled raw SSP. Regressors (~163, all known by 12:00 UTC D−1 — the identical information set and vintage cutoff as the production HGBR): full D−1/D−2/D−7 price profiles, daily aggregates, MID market-index lags, Phase-5b day-ahead vintages (WINDFOR/TSDF/margin/penetration/surplus), day-of-week and holiday. α per SP via 5-fold LassoCV; P10/P90 from training-residual quantiles (pre-conformal). Full 48-model fit: ~6 seconds.

## Walk-forward verdict (same 5 folds, same evaluation rows as the HGBR)

Window sweep {365d, 1092d, full}: 1092d best; full history barely helps a linear model — the 5a full-history effect is tree-specific.

| Fold | LEAR MAE | HGBR MAE | LEAR/HGBR |
|---|---|---|---|
| autumn-2025 | 26.28 | 27.49 | **0.956** |
| autumn-2026 | 58.42 | 47.69 | 1.225 |
| spring-2026 | 32.72 | 31.30 | 1.045 |
| summer-2025 | 23.58 | 24.63 | **0.957** |
| winter-2025 | 18.36 | 19.08 | **0.962** |
| **ALL** | **31.87** | **29.93** | **1.065** |

**Headline:** the production model beats the standard interpretable benchmark by **6.5% overall and by 22% in the current (autumn-2026) regime**. LEAR actually wins the three calm folds — a classic EPF result — but as a linear map it badly under-responds to regime-level jumps (autumn bias −£33/day vs HGBR's −£12). HGBR's complexity is earned exactly where it matters.

## Ensemble decision: HELD

50/50 LEAR+HGBR beats HGBR in 4/5 folds and overall (−1.3%) but is **+5% worse in autumn-2026** — it trades live-regime accuracy for backtest-average vanity, so it fails the regime-conditional reading of the gate. Not shipped. (A regime-gated ensemble is a possible future experiment; complexity would need stronger evidence.)

## Interpretability payoff

Per-SP standardised coefficients are physically meaningful and shipped to the dashboard coefficient panel (`model_assets/lear_coefficients.json`): overnight SPs load on the previous evening's close and **negative wind penetration**; the morning ramp on the demand forecast; the evening peak on forecast margin. The Phase-5b vintage features explain themselves in linear form.

## What shipped

- `src/models/train_lear.py` — `--backtest` (5-fold WF harness, window sweep), `--fit` (production fit + coefficient export), `--predict` (daily H+1 on production HGBR timing, appends `lear_forecast_history.csv`), `--seed-history`.
- Dashboard **Model Comparison panel**: rolling 30-day daily MAE *and* pinball (P10/P90) toggle — LEAR vs HGBR vs persistence (persistence MAE-only) — with the actual daily-mean price on a secondary axis for context.
- **Seed/live separation**: the seeded walk-forward period is shaded "not like-for-like" with a live-boundary marker; headline metrics switch to live-only once ≥7 live days settle. (The seeded window pairs backtest-LEAR with the *pre-upgrade* as-published HGBR and must not be read as a live verdict.)
- **Coefficient panel**: per-SP top-10 standardised coefficients, α, sparsity.
- `daily_pipeline.yml`: weekly LEAR refit on the retrain gate + daily prediction, both `continue-on-error` — the baseline can never block production.

## Reading the live panel

The fair live comparison starts 2026-11-04 (first full live day for both models post-upgrade). Differences under ~10% over a month are noise (daily-MAE σ ≈ £20); the walk-forward remains the decision instrument, the panel the public sanity check. Monitoring verdicts for 5a/5b/5c land ~mid-November.

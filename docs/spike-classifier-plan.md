# Phase 6a Plan: Spike Classifier + Gated Asymmetric PI Widening

## Context and constraints

- Inference order: base HGBR → PI calibration (δ per SP) → **[Phase 6a HERE]** → Kalman → splice
- `is_spike_P` fires on 1 row in the WF window (2025-10-13 SP26, £487). Target for classifier is `spike_day = any SP with actual > £150` (17 days in the 119-day backtest window, 14%)
- δ_hi from **elevated-price conformity scores** (`actual - q90_calibrated > 0` where `actual > £120`), NOT Code-P (46% of SPs at normal prices)
- Config flag `"spike_widening": false` — off by default; only enable after all gates pass

---

## Data findings

**Spike-day rate by year** (ssp > £150):

| Year | Days | Spike days | Rate |
|------|------|------------|------|
| 2022 | 365 | 353 | 96.7% — energy crisis, exclude |
| 2023 | 365 | 209 | 57.3% — post-crisis elevated, use with caution |
| 2024 | 366 | 28 | **7.7%** |
| 2025 | 365 | 82 | **22.5%** |
| WF window (2025-07 – 2026-04, 119 days) | — | **~17** | **~14%** |

**Training window decision:** 2024-01-01 to 2025-06-30 (548 days; ~14% positive rate matching WF regime). If this gives a Brier score > 0.12, extend back to 2023-01-01.

**Tukey-fence threshold (> £353.7):** only 9 positive training days — far too sparse for a classifier. Use £150 threshold.

---

## Features (all ≥ 48-SP lagged — no leakage)

All are constructed from D-1 information:

| Feature | Source | Lag guarantee |
|---|---|---|
| `elevated_count_lag1d` | #SPs on D-1 with `ssp_raw_lag_48 > 120` | same-SP lag-48 = D-1 |
| `spike_count_roll_7d` | sum of `(elevated_count_lag1d > 0)` over rolling 7 days | all D-1..D-7 |
| `wind_pct_daily_mean_lag1d` | mean(`wind_pct_lag_48`) per day | D-1 CI wind; **substituted with WINDFOR at inference** via existing `lf` dict mechanism |
| `wind_ms_daily_mean_lag1d` | mean(`wind_ms_lag_48`) per day | D-1 |
| `gas_pct_daily_mean_lag1d` | mean(`gas_pct_lag_48`) per day | D-1 |
| `niv_daily_mean_lag1d` | mean(`net_imbalance_volume_lag_48`) per day | D-1 |
| `sin_month`, `cos_month` | from `month` (cyclical) | known at D-1 |
| `is_business_day` | calendar lookup for D+1 | known at D-1 |

`neg_day_classifier` precedent: use the same `lf` dict at inference time. The spike classifier reads these features from `lf` — WINDFOR substitution of `wind_pct_daily_mean_lag1d` is already wired at line 621 of `forecast_phase3.py`.

**Leakage audit:** `ssp_raw_lag_48` at row (date D, SP h) = ssp_raw at (date D-1, SP h). Mean across all 48 SPs of day D = mean ssp_raw on day D-1. All features are strictly lagged ≥48 SPs. ✓

---

## Deliverables

### 1. `src/models/train_spike_classifier.py` (NEW)

```
Inputs  : data/processed/features_5yr.csv
Output  : model_assets/spike_classifier_v1.pkl         (sklearn Pipeline)
          model_assets/spike_classifier_v1_feats.json
          model_assets/spike_classifier_v1_eval.json
```

Steps:

1. Load features, aggregate to day level (mean per day for continuous features; sum for counts)
2. Compute `spike_day = any(ssp_raw > 150)` per day as target
3. Compute `spike_count_roll_7d` = 7-day rolling sum of `(elevated_count_lag1d > 0)` indicator
4. Train on 2024-01-01 – 2025-06-30; evaluate on 2025-07-01 – 2026-04-30 (WF window, 119 days)
5. Model: `LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)` inside `Pipeline([StandardScaler, LR])`; wrapped in `CalibratedClassifierCV(cv=5, method='isotonic')`
6. Evaluate: Brier score, precision/recall curve, conditional coverage at τ ∈ {0.1, 0.2, 0.3}
7. Save pkl + feats JSON; write eval JSON with per-day P(spike) predictions for the WF window

**Evaluation definition — conditional coverage:** for each τ, compute coverage (% of actual within [q10, q90]) on flagged days (P(spike) > τ) in the WF simulation, before and after applying δ_hi widening. Gate is: coverage lift ≥ 5pp on flagged days.

---

### 2. `src/models/compute_delta_hi.py` (NEW)

```
Inputs  : model_assets/walk_forward_predictions.csv
          model_assets/pi_calibration_v1.json
Output  : model_assets/delta_hi_v1.json
```

Algorithm:

1. Load WF predictions (raw q90) and pi_calibration_v1.json (δ(sp) per SP)
2. Compute `q90_cal = q90_raw + delta_sp(sp)` for each row
3. Filter to elevated rows: `actual > 120`
4. Compute upper-tail conformity score: `score_hi = max(actual - q90_cal, 0)`
5. Per-SP quantile: `delta_hi(sp) = p80(score_hi[sp])` — pooled across seasons (per-season too sparse: ~360 raw elevated rows / 48 SPs ≈ 7.5 per SP; season-split gives ~2/group)
6. Fallback: if SP has < 5 elevated rows, use global p80 of score_hi
7. Save artifact with per-SP δ_hi and global fallback

**Note on season split:** with only 119 WF days, per-(SP, season) groups average ~2 rows — cannot compute a reliable p80. Season will become viable after one more year of walk-forward data. Artifact schema reserves `delta_hi_by_sp_season` as an empty dict until then.

---

### 3. `src/models/correctors.py` (ADD)

New function `apply_spike_widening(df, p_spike, tau, delta_hi_path, season)`:

- If `p_spike <= tau`: return `df` unchanged
- Load δ_hi(sp) from artifact
- `out["ssp_q90"] += delta_hi_arr` (asymmetric — Q10 and Q50 unchanged)
- Return widened df

---

### 4. `src/models/forecast_phase3.py` (ADD)

After PI calibration block (~line 723), before Kalman block (~line 734):

```python
# Phase 6a: spike-gated asymmetric PI widening
_spike_clf_path   = ASSETS_DIR / "spike_classifier_v1.pkl"
_delta_hi_path    = ASSETS_DIR / "delta_hi_v1.json"
_spike_feats_path = ASSETS_DIR / "spike_classifier_v1_feats.json"
if (cfg.get("spike_widening", False)
        and _spike_clf_path.exists() and _delta_hi_path.exists()):
    try:
        spike_clf   = joblib.load(_spike_clf_path)
        spike_feats = json.load(open(_spike_feats_path))
        _x_spike    = np.array([[lf.get(f, 0.0) for f in spike_feats]])
        p_spike     = float(spike_clf.predict_proba(_x_spike)[0, 1])
        tau         = cfg.get("spike_tau", 0.2)
        _season     = _date_to_season(target_date)
        result      = apply_spike_widening(result, p_spike, tau, _delta_hi_path, _season)
        log.info("Spike widening: P(spike)=%.3f tau=%.2f season=%s applied=%s",
                 p_spike, tau, _season, p_spike > tau)
    except Exception as _e:
        log.warning("Spike widening skipped: %s", _e)
```

Add to `model_assets/corrector_config.json`:

```json
"spike_widening": false,
"spike_tau": 0.2
```

---

### 5. `src/models/backtest_correctors.py` (ADD §8)

New function `compute_spike_widening_gate(all_sim, eval_json_path, delta_hi_path, taus)`:

For each τ:

1. Load per-day P(spike) from `spike_classifier_v1_eval.json`
2. Identify flagged days (P(spike) > τ) and unflagged days
3. Compute post-hoc widened Q90 for flagged-day rows: `q90_wide = pred_q90 + delta_hi(sp)`
4. Coverage before/after on flagged days; coverage on unflagged days (expect unchanged)
5. MAE change on unflagged days (expect < 0.5%)

Report with/without 2025-10-13 at every τ.

---

## Gates (all must pass at chosen τ before enabling `spike_widening: true`)

| # | Gate | Condition |
|---|---|---|
| G1 | Coverage lift on flagged days | after − before ≥ +5pp absolute |
| G2 | Coverage stability on unflagged days | \|before − after\| ≤ 0.5pp |
| G3 | MAE on unflagged days | change < 0.5% |
| G4 | Brier score on WF window | < 0.14 (base rate² ≈ 0.12) |

---

## Implementation order

1. `compute_delta_hi.py` — no sklearn, quick to validate
2. `train_spike_classifier.py` — training + WF evaluation
3. `correctors.py: apply_spike_widening()`
4. `forecast_phase3.py` — wire in (config off)
5. `backtest_correctors.py §8` — gate evaluation

---

## Open questions

1. **Training window:** 2024-01-01 – 2025-06-30 gives ~14% base rate matching WF window. Extend to 2023-01-01 (adding 293 more spike days at 57% rate) if Brier > 0.12?

2. **δ_hi source:** Computed against calibrated q90 (`q90_raw + delta_sp` from pi_calibration_v1.json), not raw q90. Confirm this is the intended baseline.

3. **τ default in config:** 0.2 chosen as moderate precision/recall tradeoff. Prefer 0.1 (recall-focused) or 0.3 (precision-focused)?

4. **Classifier storage format:** Following neg_day_classifier precedent — pkl + feats JSON. Confirm (vs JSON-serialised like PI calibration artifact).

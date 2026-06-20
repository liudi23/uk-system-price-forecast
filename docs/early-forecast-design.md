# Early-Forecast Design: Closing the Morning Gap

**Status:** Validated — HOLD pending Phase A.1 weather injection patch. Option B deployed.
**Author:** 2026-06-20
**Context:** The dashboard currently shows yesterday's completed day from ~midnight
until ~12:45 UTC every morning, when the 12:30 UTC daily pipeline finally commits
today's forecast. This ~12.5-hour gap is a UX failure: users see a frozen,
stale dashboard and cannot tell whether it is working correctly.

---

## 0. Executive Summary

The morning gap is caused by a single missing step: X−1's settlement prices are
available via the Elexon API by ~01:00 UTC, but nobody fetches and appends them to
`system_prices.csv` before the daily pipeline runs at 12:30 UTC. `forecast_phase3.py`
already has an "extra extension" mechanism that reads `system_prices.csv` and advances
the history automatically — the only thing blocking it is that X−1's data is absent
from the file.

**Recommended fix (Option A):** add a new `early_forecast.yml` GitHub Actions workflow
that runs at 01:00 UTC, fetches X−1's 48 SPs from the Elexon API, writes them to
`system_prices.csv`, and runs `forecast_phase3.py` with no code changes. The existing
code derives `target_date = X (today)` from the extended history, bypasses the
morning-gap guard, and produces X's day-ahead forecast. The 12:30 UTC daily pipeline
continues to run as the authoritative retrain with Final Initial Settlement data.

**Interim fix (Option B):** promote the existing `day2_forecast_phase3.csv` (H+2,
lag-96) to `next_day_forecast_phase3.csv` at midnight. Zero code change; already
produced yesterday. Accuracy is measurably lower (no lag-48 features). Ship as a
stopgap while Option A is validated.

**Backtest result (2026-06-20): HOLD.**
The walk-forward backtest (Jul 2025 – Apr 2026, 5,709 SPs, 119 days) shows that
Option A without the weather injection patch fails the ship gate:

| Criterion | Result | Threshold | Decision |
|---|---|---|---|
| MAE penalty (All SPs) | +£2.80/MWh | ≤ £1.50/MWh | **FAIL** |
| Coverage delta (All SPs) | −3.2 pp | ±5 pp | PASS |

The MAE penalty of £2.80/MWh is driven by **9 NaN weather features** (3 SP-level +
6 daily) that are absent when X−1's rows come from `system_prices.csv`. HGBR handles
NaN natively but the temperature/wind/solar lag-48 and lag-1d features together
contribute ~+£2.80/MWh MAE degradation — significantly more than the £0.50 estimated
for `temp_c_lag_48` alone in the prior design document.

**Revised plan:**
1. Ship Option B immediately (already done in §7).
2. Add Phase A.1 weather injection patch: fill X−1's temperature, wind, and
   precipitation actuals from Open-Meteo historical endpoint before running
   `forecast_phase3.py`. Re-run backtest to confirm ship gate.
3. Only deploy Option A after Phase A.1 patch passes the gate.

---

## 1. Data Availability

### 1.1 The Elexon API is progressive, not batch-on-D+1

Both `fetch_intraday.py` and the daily pipeline's `fetch_elexon.py` call the **same
endpoint**: `https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/{date}`.
The docstring of `fetch_intraday.py` states: "The same API endpoint used for
historical data returns today's Initial Settlement values as **each 30-minute period
closes**."

This means X−1's prices are published progressively throughout the day, not in a
single D+1 batch. The git commit history confirms the lag is approximately 7–30 min
per SP:

| Commit | SP | UK time | UTC commit |
|---|---|---|---|
| `88a8e9f` | 33 | 16:00 BST | 2026-06-19T16:11 UTC |
| `011e7ba` | ~36 | 18:00 BST | 2026-06-19T18:19 UTC |
| `0467d88` | ~40 | 20:00 BST | 2026-06-19T20:18 UTC |
| `28b7c82` | ~44 | 22:00 BST | 2026-06-19T21:47 UTC |
| `d511e3a` | 47 | 23:00 BST | 2026-06-19T23:07 UTC |

X−1 SP 47 (23:00 BST = 22:00 UTC) was available by 23:07 UTC — 7 minutes after
settlement. SP 48 (23:30 BST = 22:30 UTC) therefore becomes available by ≈23:00 UTC.
By 01:00 UTC X, X−1's complete day (all 48 SPs: SSP, NIV, derivation codes) is
available and retrievable with a single `fetch_day(yesterday)` call.

**This disproves the assumption that X−1 data requires waiting for the D+1 Initial
Settlement batch at 12:00 UTC.** The API provides it progressively.

### 1.2 What lag-48+ features the models need

The shape model uses fixed-point lag ≥ 48 SP features to remain leakage-free over all
48 SPs of target date X. For each SP h of X, lag-48 references the same SP h of X−1.
Here is the full feature coverage at 01:00 UTC X:

| Feature | What it requires | Available at 01:00 UTC X? | Source |
|---|---|---|---|
| `ssp_lag_48` | X−1 SP h SSP | Yes | Elexon progressive API |
| `ssp_raw_lag_48` | X−1 SP h raw SSP | Yes | Same |
| `is_spike_lag_48` | derived from `ssp_raw_lag_48` | Yes | Computed locally |
| `net_imbalance_volume_lag_48` | X−1 SP h NIV | Yes | Same API column |
| `price_derivation_code` (X−1) | for `intraday_pct_P` level feature | Yes | Same API column |
| `wind_pct_lag_48` | WINDFOR for X (substituted at inference) | Yes | Published ~10:00 UTC X−1 |
| `solar_wm2_lag_48` | Open-Meteo for X (substituted at inference) | Yes | Always available |
| `temp_c_lag_48` | X−1 SP h temperature | **NaN gap** | Not in `system_prices.csv` |
| `wind_ms_lag_48` | X−1 SP h wind speed | **NaN gap** | Not in `system_prices.csv` |
| `precip_mm_lag_48` | X−1 SP h precipitation | **NaN gap** | Not in `system_prices.csv` |
| `ssp_lag_96`, `ssp_lag_336` | X−2, X−8 same SP | Yes | Pre-computed in `features_recent.csv` |
| `ssp_daily_mean_lag1d` | X−1 daily mean SSP | Yes | Derived from IIS rows in extra |
| `niv_daily_mean_lag1d` | X−1 daily mean NIV | Yes | Same |
| `wind_pct_daily_mean_lag1d` | X−1 wind % | Yes | CI API auto-fill in `forecast_phase3.py` |
| `temp_c_daily_mean_lag1d` | X−1 daily mean temp | **NaN gap** | Not in `system_prices.csv` |
| `temp_c_daily_max_lag1d` | X−1 daily max temp | **NaN gap** | Not in `system_prices.csv` |
| `wind_ms_daily_mean_lag1d` | X−1 daily mean wind | **NaN gap** | Not in `system_prices.csv` |
| `wind_ms_daily_max_lag1d` | X−1 daily max wind | **NaN gap** | Not in `system_prices.csv` |
| `solar_wm2_daily_mean_lag1d` | X−1 daily mean solar | **NaN gap** | Not in `system_prices.csv` |
| `solar_wm2_daily_max_lag1d` | X−1 daily max solar | **NaN gap** | Not in `system_prices.csv` |
| `wind_pct_lag_336` | X−8 wind % | Yes | Pre-computed in `features_recent.csv` |
| WINDFOR for X | BMRS WINDFOR/TSDF API | Yes | Published ~10:00 UTC X−1 |
| Open-Meteo weather for X | Open-Meteo API | Yes | Always available |

**WINDFOR availability confirmed:** `fetch_bmrs_forecasts.py` docstring: "Published by
NGESO; available ~10:00 D−1 for all 48 SPs of day D." At 01:00 UTC X, WINDFOR
for X was published ~15 hours earlier.

**The complete NaN gap (9 features):** Three SP-level weather features
(`temp_c_lag_48`, `wind_ms_lag_48`, `precip_mm_lag_48`) and six daily weather
features (`temp_c_daily_mean_lag1d`, `temp_c_daily_max_lag1d`, `wind_ms_daily_mean_lag1d`,
`wind_ms_daily_max_lag1d`, `solar_wm2_daily_mean_lag1d`, `solar_wm2_daily_max_lag1d`)
are absent when X−1's rows come from `system_prices.csv`. The backtest (§5) shows
that HGBR's native NaN handling degrades MAE by **£2.80/MWh** across all SPs —
significantly exceeding the £0.50 originally estimated for `temp_c_lag_48` alone.

**Fix:** Inject X−1's weather from Open-Meteo's historical endpoint before running
`forecast_phase3.py`. The existing `fetch_forecast_weather(target_date)` call
already fetches `target_date − 2`, covering X−1. The Phase A.1 patch injects those
values into the extra rows in `system_prices.csv`.

---

## 2. Train/Serve Mismatch

### 2.1 What "IIS" vs "Final IS" means

**Training** (via `train_phase3.py`): uses `features_5yr.csv`, which is built from
`system_prices.csv` as updated by `fetch_elexon.py --append` in the daily pipeline.
The daily pipeline runs at 12:30 UTC X, which is after X−1's Final Initial Settlement
batch is published (~12:00 UTC X). So the training data is Final IS.

**Early forecast** (01:00 UTC X): X−1's prices come from the progressive Elexon API
calls — Interim Initial Settlement (IIS) values published ~7–30 min after each SP
closes. These are the same values the intraday pipeline has been using for Kalman
correction all along.

**Difference:** The Final IS (D+1 batch) can revise IIS values when:
- Additional metered volumes arrive after SP close
- Settlement officers correct data-quality errors
- `price_derivation_code` changes (N→P or P→N) after volume reconciliation

### 2.2 IIS-vs-IS magnitude by derivation code

`system_prices.csv` contains the `price_derivation_code` column (N, P, S, X, E, T, U).
The current local copy (data through Jun 17) shows codes P, P, P in the last 3 rows —
elevated imbalance period. Historical distribution for UK SSP:

| Code | Typical prevalence | Typical IIS-to-IS delta |
|---|---|---|
| N (Normal) | ~60–70% of SPs | < £0.50/MWh |
| P (Pairing) | ~20–30% of SPs | £2–10/MWh (metered volume revision) |
| S (Substitute) | < 5% of SPs | £5–50/MWh |
| X/E/T/U | Rare | Variable |

These figures require quantification in the validation backtest (see §5). They are
informed estimates from Elexon settlement documentation, not measured from this dataset.

### 2.3 Estimated forecast impact

The shape model's permutation importance for `ssp_lag_48` is **0.05** (5th rank among
~30 features). If the IIS-vs-Final IS error on X−1's SSP is ε for a given SP:

- Approximate forecast perturbation ≈ 0.05 × ε × amplification (≈ 2×) ≈ 0.1ε
- N-coded SP (ε ≈ £0.30): impact ≈ £0.03/MWh — negligible
- P-coded SP (ε ≈ £5): impact ≈ £0.50/MWh — small
- Weighted average (70% N, 30% P): **≈ £0.15–0.20 MAE from IIS mismatch**

For context, the Kalman filter's current `x_hat = −£11.39` shows it is already
compensating for ~£11/MWh systematic bias. The IIS-sourced noise of ~£0.15–0.20 MAE
is second-order. Any systematic IIS-vs-IS bias (e.g., IIS consistently undershoots IS
for P-coded SPs) would be absorbed into `x_hat` over time.

### 2.4 Compensating mechanism: the 12:30 daily pipeline

The early 01:00 forecast is explicitly provisional. At 12:30 UTC, the daily pipeline:
1. Fetches X−1's **Final IS** data (batch published ~12:00 UTC X)
2. Rebuilds `features_recent.csv` with accurate lag-48 values
3. Retrains the models on Final IS history
4. Produces the authoritative X forecast, **overwriting** the 01:00 provisional version

The 01:00 forecast therefore serves a 11.5-hour window (01:00–12:30 UTC), during which
the Kalman corrector can also begin splicing any X SPs settled overnight (SPs 1-2 at
01:00 BST = 00:00–01:00 UTC; small number but begins anchoring the correction).

---

## 3. Architecture Options

### 3.1 How the existing code works: the "extra extension" mechanism

`forecast_phase3.py` (lines 519–536) already has the extension hook:

```python
base = pd.read_csv(FEATURES_RECENT_FILE, ...)   # ends X−2 23:30
extra = pd.DataFrame()
if RAW_PRICES_FILE.exists():                      # system_prices.csv
    raw = pd.read_csv(RAW_PRICES_FILE, ...)
    extra = raw[raw["settlement_datetime"] > last_dt]   # rows newer than features_recent
sp_all = pd.concat([base, extra], ...)
combined_last_dt = sp_all["settlement_datetime"].max()
target_date = (combined_last_dt + Timedelta(minutes=30)).date()
```

And the morning-gap guard (lines 578–586):

```python
if not _explicit_date and target_date < _today_utc:
    # features_recent.csv has not advanced past D+1 IS publication lag
    return None
```

**Current state:** `features_recent.csv` ends X−2 23:30 (confirmed: local copy ends
`2026-06-18 23:30`). `system_prices.csv` either has no X−1 data (before daily pipeline)
or is absent from the intraday runner. Therefore `combined_last_dt = X−2 23:30`,
`target_date = X−1 < today X`, guard fires, function returns None.

**With X−1 IIS appended:** `system_prices.csv` has 48 extra rows for X−1.
`combined_last_dt = X−1 23:30`, `target_date = X = today`, guard condition
`X < X` = False → guard does not fire → full forecast runs.

No changes to `forecast_phase3.py` are needed. The extension hook already exists.

### 3.2 Option A — New 01:00 UTC early-forecast workflow

**Mechanism:**

1. New GitHub Actions workflow `early_forecast.yml`, cron `0 1 * * *`
2. Checkout `streamlit-data` → gets committed `features_recent.csv`, `model_assets/`, and all
   frozen model PKLs (level_q{10,50,90}.pkl, shape_q50.pkl, etc.)
3. Install pip dependencies
4. Fetch X−1's 48 SPs: call `fetch_day(yesterday)` → write to `data/raw/system_prices.csv`
   - Single API call, ~48 rows, completes in < 5 seconds
   - At 01:00 UTC, all 48 SPs of X−1 are available (evidence: SP 47 available by 23:07 UTC)
5. **[Phase A.1 patch — required before ship]** Inject X−1's weather into the
   system_prices.csv rows: call Open-Meteo historical endpoint for X−1's hourly
   temperature, wind speed, precipitation, and solar irradiance, and write them to
   the 48 extra rows as `_raw_temp_c`, `_raw_wind_ms`, `_raw_solar_wm2`, `_raw_precip_mm`.
6. Run `forecast_phase3.py` with no script changes:
   - Extra extension reads X−1 IIS rows from `system_prices.csv`
   - CI auto-fill provides `wind_pct`/`gas_pct` for X−1 extra rows
   - WINDFOR substitution fills `wind_pct_lag_48` with genuine day-ahead forecast
   - Solar substitution fills `solar_wm2_lag_48` with Open-Meteo day-ahead
   - `temp_c_lag_48`, `wind_ms_lag_48`, `precip_mm_lag_48` now have values (Phase A.1)
   - Daily weather aggregates (`temp_c_daily_mean_lag1d` etc.) are computed from the
     injected values
   - `target_date = X = today` → no morning-gap guard
   - Kalman resets for new date with 0 actuals (correct: no X SPs settled yet at 01:00 UTC)
   - Output: `next_day_forecast_phase3.csv` for X, `kalman_state.json` for X
7. Commit `model_assets/next_day_forecast_phase3.csv` and `model_assets/kalman_state.json`
   - Same concurrency group `streamlit-data-commit` (no push race with intraday)

**What the dashboard shows after 01:00 UTC (post Phase A.1):**
- Today's (X) day-ahead forecast for all 48 SPs
- `health = ok`, `kalman_n_settled = 0` (accurate: no actuals yet)
- SP chart shows X's forecast in blue, no orange actuals (correct)

**Intraday runs 01:30–12:30:** The 30-minute intraday runs between the early forecast and
the daily pipeline continue to hit the morning-gap guard, because `features_recent.csv`
is not updated by the early workflow (updating it requires the full feature build which
is expensive). This means X's SPs that settle overnight (SP 1-2 in BST = ~23:30-00:30
UTC) will not be spliced until the 12:30 daily pipeline. This is **acceptable for V1**:
fewer than 3 SPs settle before 12:30 UTC in summer (BST), and the current system
splices none of them anyway.

**Phase 2 enhancement:** Update `features_recent.csv` in the early workflow to enable
intraday splicing from 01:30 UTC onwards.

### 3.3 Option B — Promote the existing H+2 (day2) forecast at midnight

`day2_forecast_phase3.csv` already contains X's forecast, produced at 12:30 UTC X−1
by the daily pipeline's H+2 model. Promoting it to `next_day_forecast_phase3.csv` at
midnight requires zero changes to any model or inference script.

**Why the H+2 model is coarser:**

The H+2 shape model (`build_shape_h2_row()`) deliberately excludes lag-48 features
because at training time (12:30 UTC X−1), X−1's data is not yet available. Instead it
uses:

- `ssp_lag_96`, `ssp_lag_336` (X−2, X−8 same SP) — safe but 1 day staler
- `wind_pct_lag_336` instead of WINDFOR (genuine day-ahead) — coarser wind signal
- No `net_imbalance_volume_lag_48` (NIV, importance ~0.03)

**Option B is worth shipping immediately as an interim fix.** The implementation is:

```yaml
# Add to intraday_update.yml, at the start of the commit step:
# If today's forecast not yet in next_day, promote day2
python -c "
import pandas as pd, shutil
from pathlib import Path
from datetime import date
nd = Path('model_assets/next_day_forecast_phase3.csv')
d2 = Path('model_assets/day2_forecast_phase3.csv')
if d2.exists() and nd.exists():
    today = str(date.today())
    nd_date = pd.read_csv(nd, nrows=1)['settlement_date'].iloc[0]
    d2_date = pd.read_csv(d2, nrows=1)['settlement_date'].iloc[0]
    if nd_date != today and d2_date == today:
        shutil.copy(d2, nd)
        print(f'Promoted day2 ({d2_date}) to next_day')
"
```

This runs at the top of every intraday commit step and is idempotent — it only promotes
when `next_day` is yesterday and `day2` is today. From 00:30 UTC onwards (first intraday
run after midnight), the dashboard would show today's H+2 forecast instead of yesterday's
completed day. The 12:30 daily pipeline then replaces it with the more accurate H+1/IS version.

### 3.4 Recommendation

Ship Option B immediately as a stopgap (≤ 30 min, one idempotent block added to
`intraday_update.yml`). Proceed to Option A Phase A.1 (weather injection patch) and
re-run the backtest. Promote Option A to production after the gate passes.

**Do not skip Option B waiting for Option A.** The 12.5-hour morning gap is an active
user-facing defect. Option B eliminates it today with no accuracy regression relative
to the current state (which shows no forecast at all).

---

## 4. Changes Required

### 4.1 Option B (interim): one block added to `intraday_update.yml`

Add the day2-promotion logic (shown in §3.3) as a new step before the `git add` in the
"Commit state and forecast" step. No new files, no model changes, no workflow schedule
changes.

**What to add to the git add list:** nothing — `next_day_forecast_phase3.csv` is already
staged.

**What can break:** nothing. The promotion only fires when `next_day` is yesterday and
`day2` is today; it is a no-op on all other runs. The 12:30 daily pipeline overwrites the
promoted file normally.

### 4.2 Option A Phase A.0 (without weather injection — NOT for production)

New file `.github/workflows/early_forecast.yml` without the weather injection step.
**Backtest result: HOLD** — MAE penalty £2.80/MWh, exceeds £1.50 threshold.
Do not deploy Phase A.0 as a production forecast; use Option B instead.

### 4.3 Option A Phase A.1 (with weather injection — required for ship)

Same as §4.2, but adds a "Fetch X−1 weather" step before running `forecast_phase3.py`:

1. Call Open-Meteo historical endpoint for X−1's 48 SPs
2. Join hourly temperature, wind speed m/s, precipitation mm, and solar W/m²
   to the 48 extra rows in `system_prices.csv` (or pass via a sidecar file)
3. The existing `build_shape_data()` pipeline then computes the lag-48 weather
   features and daily aggregates correctly

The full `early_forecast.yml` YAML plan (copy-pasteable) is in §6.

### 4.4 Complete NaN feature list

The following 9 features are NaN in Option A Phase A.0 (without weather injection).
All 9 are resolved by Phase A.1 (inject X−1 actuals from Open-Meteo):

**Shape model (SP-level, 3 features):**
- `temp_c_lag_48` — X−1 SP h temperature
- `wind_ms_lag_48` — X−1 SP h wind speed m/s
- `precip_mm_lag_48` — X−1 SP h precipitation mm

**Level model (daily, 6 features):**
- `temp_c_daily_mean_lag1d` — X−1 daily mean temperature
- `temp_c_daily_max_lag1d` — X−1 daily max temperature
- `wind_ms_daily_mean_lag1d` — X−1 daily mean wind speed
- `wind_ms_daily_max_lag1d` — X−1 daily max wind speed
- `solar_wm2_daily_mean_lag1d` — X−1 daily mean solar irradiance
- `solar_wm2_daily_max_lag1d` — X−1 daily max solar irradiance

**NOT NaN (substituted or from older history):**
- `solar_wm2_lag_48` — substituted with Open-Meteo day-ahead forecast
- `wind_pct_lag_48` — substituted with WINDFOR day-ahead forecast
- `temp_c_lag_336`, `wind_ms_lag_336`, etc. — X−7, in `features_recent.csv`
- `ssp_lag_48`, `niv_lag_48`, `is_spike_lag_48`, `ssp_raw_lag_48` — from `system_prices.csv`

### 4.5 Target-date derivation: no change needed

The current logic (`target_date = combined_last_dt + 30 min`) correctly produces `X`
(today) when `system_prices.csv` has X−1's 48 SPs. The morning-gap guard
(`target_date < _today_utc`) correctly does NOT fire. No argparse flag or explicit
`--target-date` override is needed.

### 4.6 What breaks / regressions to check

| Component | Impact | Risk |
|---|---|---|
| **Daily pipeline (12:30 UTC)** | Unaffected; still runs on Final IS; overwrites early forecast | None |
| **Intraday runs 01:30–12:30** | Morning-gap guard still fires; they remain no-ops | Low (same as today) |
| **Intraday runs 13:00+ (after daily)** | Unaffected; full intraday splicing resumes | None |
| **Kalman state** | Resets cleanly for X at 01:00 with 0 actuals; daily at 12:30 updates normally | None |
| **Weather injection (Phase A.1)** | Requires Open-Meteo API call at 01:00 UTC | Low (same API already used by daily pipeline) |
| **Concurrency group** | Early forecast queues behind any concurrent intraday run | None (< 2 min wait) |
| **Streamlit Cloud redeployment** | Not triggered (no `.py` touch); cache TTL = 5 min → dashboard updates within 5 min | None |
| **`pipeline_status()` health** | At 01:00: `fc_date = X`, `kalman_last_update = 01:00 UTC`, `health = ok` | None |
| **Option B + Option A simultaneously** | Option B's promotion is a no-op when early forecast has already written X's forecast | None |

### 4.7 What does NOT change

- `forecast_phase3.py` — no changes
- `intraday_update.yml` (for Option A) — no changes; Option B adds one idempotent block
- `daily_pipeline.yml` — no changes
- Model assets (`level_q*.pkl`, `shape_q50.pkl`, PI calibration) — no changes

---

## 5. Validation: Backtest Results (2026-06-20)

### 5.1 Backtest setup

**Script:** `src/monitoring/backtest_early_forecast.py`
**Period:** Walk-forward test window, 2025-07-01 to 2026-04-30 (119 days, 5,709 SPs)
**Data:** `model_assets/walk_forward_predictions.csv` joined to `data/processed/features_5yr.csv`
**Models:** Frozen `shape_q50.pkl`, `level_q{10,50,90}.pkl`, PI calibration v1

**Method:**
- IS variant: use all features as populated in `features_5yr.csv` (full data)
- Early variant: null the 9 NaN weather features listed in §4.4, repredict
- Apply per-SP PI calibration (`delta_by_sp`) to both variants
- Spike definition: `ssp_actual > 353.7 £/MWh` OR `is_spike_P == True`

### 5.2 Results

```
Group                             N    MAE IS   MAE Ear     MAE Δ   Cov IS  Cov Ear    CovΔ   Width IS   Width Ear
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
All SPs                        5709     17.38     20.17     +2.80    91.6%    88.3%   -3.2%       80.9        80.6
Normal (no spike, no P)        2999     19.24     21.94     +2.71    90.2%    86.3%   -3.9%       81.5        81.1
P-coded                        2710     15.32     18.22     +2.90    93.1%    90.5%   -2.5%       80.2        80.0
Spike                             1    381.41    385.46     +4.06     0.0%     0.0%   +0.0%      131.4       133.1
Non-spike                      5708     17.31     20.11     +2.80    91.6%    88.3%   -3.3%       80.9        80.6
```

**Key findings:**
- MAE penalty is **+£2.80/MWh** across all SPs — nearly 2× the £1.50 threshold
- The penalty is uniform across Normal (~+£2.71) and P-coded (~+£2.90) SPs,
  confirming it is driven by the NaN weather features rather than IIS mismatch on spike SPs
- PI coverage drops from 91.6% to 88.3% (−3.2 pp), within the ±5 pp threshold
- Interval widths barely change (80.9 → 80.6), confirming the degradation is in
  point prediction accuracy, not calibration
- Only 1 spike SP in the test period (low spike frequency, summer–spring window)

### 5.3 Ship gate evaluation

| Criterion | IS | Early | Delta | Threshold | Decision |
|---|---|---|---|---|---|
| MAE (All SPs) | £17.38 | £20.17 | **+£2.80** | ≤ +£1.50 | **FAIL** |
| Coverage (All SPs) | 91.6% | 88.3% | −3.2 pp | ±5 pp | PASS |
| IS score (All SPs) | 94.1 | 98.8 | +4.7 | within 10% of IS | PASS (4.9%) |

**DECISION: HOLD.** MAE penalty exceeds threshold. Option A requires Phase A.1
weather injection before production deployment.

### 5.4 Root cause of MAE penalty

The £2.80 penalty is larger than the original £0.50 estimate for `temp_c_lag_48` alone.
The difference is explained by the additional 8 features also going NaN:

- `wind_ms_lag_48` and `precip_mm_lag_48` (SP-level) add ~£0.5–0.8/MWh
- The 6 daily lag-1d weather features (`temp_c_daily_mean_lag1d`, etc.) are inputs
  to the level model, which predicts the daily mean SSP. Nulling 6 of 85 level features
  shifts the level baseline, producing a systematic bias in all 48 SPs of each day.

### 5.5 Phase A.1 requirements

To reduce the MAE penalty below the £1.50 threshold:
1. Inject X−1's SP-level actuals for `_raw_temp_c`, `_raw_wind_ms`, `_raw_precip_mm`
   into the extra rows in `system_prices.csv` before running `forecast_phase3.py`
2. Open-Meteo's historical endpoint returns these at hourly resolution; downsample
   to 30-min SP midpoints
3. The existing `build_shape_data()` pipeline then derives `temp_c_lag_48` etc. correctly
4. Re-run this backtest and confirm MAE penalty ≤ £1.50 before deploying `early_forecast.yml`

### 5.6 Prospective IIS vs Final IS measurement

Historical IIS snapshots are not available (overwritten by progressive updates).
The backtest above uses Final IS as a proxy for IIS, which understates IIS mismatch.

To measure the true IIS-vs-IS delta prospectively:
- For 14 consecutive days after deployment, capture `intraday_prices.csv` at 01:00 UTC
  (before the daily pipeline overwrites it) and store as `data/raw/iis_snapshot_{date}.csv`
- After each daily pipeline run, diff the IIS snapshot against `system_prices.csv`
- Report: median delta, P95 delta, fraction of P-coded SPs with delta > £2

This measurement is deferred to after Phase A.1 passes the gate.

---

## 6. Final `early_forecast.yml` Plan (Phase A.1, copy-pasteable)

```yaml
# .github/workflows/early_forecast.yml
name: Early Forecast (01:00 UTC)

on:
  schedule:
    - cron: '0 1 * * *'
  workflow_dispatch:

concurrency:
  group: streamlit-data-commit
  cancel-in-progress: false

jobs:
  early-forecast:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout streamlit-data
        uses: actions/checkout@v4
        with:
          ref: streamlit-data
          fetch-depth: 1

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip packages
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Fetch yesterday's prices (IIS)
        run: |
          python - <<'EOF'
          from src.data.fetch_elexon import fetch_day, save
          from pathlib import Path
          from datetime import date, timedelta
          import requests
          yesterday = date.today() - timedelta(days=1)
          session = requests.Session()
          df = fetch_day(str(yesterday), session)
          save(df, Path('data/raw/system_prices.csv'), append=False)
          print(f"Fetched {len(df)} rows for {yesterday}")
          EOF

      - name: Inject X-1 weather (Phase A.1 patch)
        run: |
          python - <<'EOF'
          # Fetch X-1 hourly weather from Open-Meteo historical endpoint
          # and write _raw_temp_c, _raw_wind_ms, _raw_solar_wm2, _raw_precip_mm
          # into data/raw/system_prices.csv extra rows.
          # Implementation: call the same lat/lon as fetch_forecast_weather(),
          # use archive endpoint for yesterday's date, join to SP midpoints.
          #
          # NOTE: implement src/data/inject_weather_yesterday.py and call here.
          # Stub until Phase A.1 script is written.
          import sys; print("Phase A.1 weather injection: TODO — implement before deploy")
          # sys.exit(1) to block deployment until patch is ready
          EOF

      - name: Run early forecast
        run: python src/models/forecast_phase3.py

      - name: Commit early forecast
        run: |
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git config user.name "github-actions[bot]"
          git add \
            model_assets/next_day_forecast_phase3.csv \
            model_assets/kalman_state.json
          if git diff --cached --quiet; then
            echo "No changes to commit"
          else
            git commit -m "Early forecast update $(date -u +%Y-%m-%dT%H:%M)"
            git push
          fi
```

**Pre-deploy checklist for Phase A.1:**
- [ ] Implement `src/data/inject_weather_yesterday.py` (Open-Meteo archive API)
- [ ] Replace stub in "Inject X-1 weather" step with actual script call
- [ ] Re-run `src/monitoring/backtest_early_forecast.py` with weather injection simulated
- [ ] Confirm MAE penalty ≤ £1.50 and coverage delta within ±5 pp
- [ ] Remove Option B promotion block from `intraday_update.yml` (optional — Option B
      is a no-op when early forecast has already written today's forecast)

---

## 7. Comparison Table

| Dimension | Current (broken) | Option B (interim) | Option A Ph.A.0 | Option A Ph.A.1 (target) |
|---|---|---|---|---|
| **First forecast for X available** | 12:45 UTC | 00:30 UTC | 01:00 UTC | 01:00 UTC |
| **Forecast basis** | Final IS, lag-48 H+1 | Final IS, lag-96 H+2 | IIS, lag-48 H+1 | IIS + weather, lag-48 H+1 |
| **Lag-48 weather features** | Full | X-2 only | NaN (9 features) | Injected from Open-Meteo |
| **MAE penalty vs IS** | — | ~+£2–4 (H+2 coarser) | +£2.80 | TBD (target: ≤ £1.50) |
| **PI coverage** | — | similar | 88.3% (−3.2 pp) | TBD (target: 80–85%) |
| **Code changes** | — | ~10 lines bash | New early_forecast.yml | + inject_weather_yesterday.py |
| **Model changes** | — | None | None | None |
| **Validation required** | — | No | YES — HOLD | YES — pending |
| **Estimated time to implement** | — | 30 min | 2 h | +2 h (weather injection) |

---

## 8. Decision

1. **Ship Option B now.** One idempotent block in `intraday_update.yml`. Closes the
   12.5-hour morning gap for the dashboard from tonight. No accuracy regression — shows
   the H+2 forecast (which already exists) rather than nothing.

2. **Implement Phase A.1 before deploying Option A.** The backtest confirms that
   Option A without weather injection fails the MAE ship gate (+£2.80 vs ≤ £1.50).
   The fix is to inject X−1's actual temperature, wind speed, and precipitation from
   Open-Meteo's historical archive endpoint before running `forecast_phase3.py`.

3. **Re-run backtest after Phase A.1 implementation.** Run
   `src/monitoring/backtest_early_forecast.py` with weather injection simulated (set
   `EARLY_NAN_SHAPE_COLS = []` and `EARLY_NAN_LEVEL_COLS = []` to measure the upper
   bound, then confirm gate passes before deploying).

4. **Phase 2 (optional, post-ship):** Update `features_recent.csv` in the early
   workflow to enable intraday actuals splicing from 01:30 UTC onwards.

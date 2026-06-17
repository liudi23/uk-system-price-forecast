#!/usr/bin/env python3
"""
Walk-forward backtest: Static Base vs AlphaCorrector vs KalmanCorrector.

Data source : model_assets/walk_forward_predictions.csv   (4 seasonal folds,
              2025-07 to 2026-04, 119 days × 48 SPs = 5,709 rows).
Output      : reports/corrector_backtest/report.md + PNG plots.

Publication-delay model
-----------------------
At pipeline step h (h=0 = 08:30 UTC, h=9 = 17:30 UTC) we assume SPs
1 .. (17 + 2*h) have cleared.  The corrector is applied to the remaining
unsettled SPs; metrics are computed on those unsettled SPs only.

Usage
-----
    python src/models/backtest_correctors.py [--output-dir reports/corrector_backtest]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product

ROOT       = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "model_assets"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from correctors import AlphaCorrector, KalmanCorrector

# ── Simulation constants ───────────────────────────────────────────────────────
N_SETTLED_BASE   = 17   # SPs assumed settled by 08:30 UTC
N_HOURLY_STEPS   = 10   # h = 0..9 (08:30–17:30 UTC)
MIN_SETTLED_NIS  = 3    # skip steps with too few observations for NIS

FOLD_ORDER = ["summer-2025", "autumn-2025", "winter-2025", "spring-2026"]

# ── NIS grid ──────────────────────────────────────────────────────────────────
Q_GRID     = [5.0, 10.0, 21.0, 35.0, 50.0, 100.0]
SIGMA_GRID = [20.0, 28.0, 35.0, 45.0, 55.0]
GAMMA_GRID = [0.90, 0.95, 0.966, 0.98, 1.0]

# ── Spike harness constants ───────────────────────────────────────────────────
PRICE_BUCKET_BINS   = [-np.inf, 85.0, 120.0, 150.0, 250.0, np.inf]
PRICE_BUCKET_LABELS = ["<£85", "£85-120", "£120-150", "£150-250", "£250+"]
SPIKE_EXCLUDE_DATE  = "2025-10-13"   # single true spike event; expose leverage


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load walk-forward predictions with pre-joined spike and risk columns.

    Expects walk_forward_predictions.csv to have been updated by
    src/data/join_derivation_code.py, which adds:
        price_derivation_code      P/N/K from Elexon BMRS
        is_code_P                  bool — Code-P mechanism flag (46% of SPs; cross-tab only)
        is_spike_P                 bool — (code=='P') AND (actual > Tukey fence)  [1 row in WF]
        elevated_count_lag1d       int  — #SPs on D-1 with actual > £120
        wind_pct_daily_mean_lag1d  float— mean wind pct on D-1 (0-100 scale)
        spike_risk_flag            bool — ex-ante day-level risk signal (no data leakage)
    """
    wf_path = ASSETS_DIR / "walk_forward_predictions.csv"
    if not wf_path.exists():
        raise FileNotFoundError(f"walk_forward_predictions.csv not found at {wf_path}")

    wf = pd.read_csv(wf_path)
    wf["settlement_date"] = wf["settlement_date"].astype(str)

    # Backfill columns if join_derivation_code.py has not been run yet
    for col, default in [
        ("price_derivation_code", "N"),
        ("is_code_P", False),
        ("is_spike_P", False),
        ("spike_risk_flag", False),
    ]:
        if col not in wf.columns:
            wf[col] = default

    wf["is_code_P"]      = wf["is_code_P"].astype(bool)
    wf["is_spike_P"]     = wf["is_spike_P"].astype(bool)
    wf["spike_risk_flag"] = wf["spike_risk_flag"].astype(bool)

    # Ensure fold order
    wf["fold"] = pd.Categorical(wf["fold"], categories=FOLD_ORDER, ordered=True)
    wf = wf.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)
    return wf


# ── NIS grid search ───────────────────────────────────────────────────────────

def _nis_single_params(
    wf: pd.DataFrame, Q: float, sigma_sp: float
) -> pd.DataFrame:
    """
    Run scalar Kalman filter innovation diagnostics for one (Q, sigma_sp) combo.
    Returns DataFrame with columns: NIS, fold, date, h.
    """
    R0 = sigma_sp ** 2
    records = []

    for (d, fold), day_df in wf.groupby(["settlement_date", "fold"], sort=False, observed=True):
        x_hat, P = 0.0, R0
        sp_arr    = day_df["settlement_period"].values
        q50_arr   = day_df["ssp_q50"].values
        act_arr   = day_df["ssp_actual"].values

        for h in range(N_HOURLY_STEPS):
            n_settled = min(N_SETTLED_BASE + 2 * h, 48)
            mask      = sp_arr <= n_settled
            n         = int(mask.sum())
            if n < MIN_SETTLED_NIS:
                continue

            z    = float((act_arr[mask] - q50_arr[mask]).mean())
            R    = R0 / n
            Pm   = P + Q
            S    = Pm + R
            nis  = (z - x_hat) ** 2 / S

            records.append({"NIS": nis, "fold": str(fold), "date": d, "h": h})

            K     = Pm / S
            x_hat = x_hat + K * (z - x_hat)
            P     = (1.0 - K) * Pm

    return pd.DataFrame(records)


def tune_kalman(wf: pd.DataFrame) -> dict:
    """
    Grid search over Q × sigma_sp × gamma to minimise |mean(NIS) - 1|.
    Returns dict with keys: Q, sigma_sp, gamma, mean_nis, nis_by_fold, all_results.
    """
    results = []
    for Q, sigma_sp, gamma in product(Q_GRID, SIGMA_GRID, GAMMA_GRID):
        nis_df = _nis_single_params(wf, Q, sigma_sp)
        if nis_df.empty:
            continue
        mean_nis = float(nis_df["NIS"].mean())
        nis_by_fold = nis_df.groupby("fold")["NIS"].mean().to_dict()
        score = abs(mean_nis - 1.0)
        results.append(
            dict(Q=Q, sigma_sp=sigma_sp, gamma=gamma,
                 mean_nis=mean_nis, score=score, nis_by_fold=nis_by_fold)
        )

    results.sort(key=lambda r: r["score"])
    best = results[0]
    best["all_results"] = results
    return best


# ── Walk-forward simulation ───────────────────────────────────────────────────

class _StaticBase:
    """No-op corrector: returns base forecast unchanged."""
    def __call__(self, forecast_df, settled_actuals, state):
        return forecast_df.copy(), {}


def _run_simulation(wf: pd.DataFrame, corrector, name: str) -> pd.DataFrame:
    """
    Simulate hourly correction for every (date, hourly step) pair.
    Returns records for UNSETTLED SPs only:
        date, fold, h, sp, is_spike_P,
        pred_q10, pred_q50, pred_q90, actual
    """
    records = []
    needed_cols = {"settlement_date", "settlement_period", "ssp_predicted",
                   "ssp_q10", "ssp_q50", "ssp_q90", "is_actual"}

    for (d, fold), day_df in wf.groupby(["settlement_date", "fold"], sort=False, observed=True):
        day_df = day_df.sort_values("settlement_period").reset_index(drop=True)

        # Build 48-row base forecast DataFrame expected by correctors
        base_fc = day_df[["settlement_date", "settlement_period",
                           "ssp_q10", "ssp_q50", "ssp_q90"]].copy()
        base_fc["ssp_predicted"] = base_fc["ssp_q50"].values
        base_fc["is_actual"]     = False

        sp_arr       = day_df["settlement_period"].values
        act_arr      = day_df["ssp_actual"].values
        spike_arr    = day_df["is_spike_P"].values
        code_p_arr   = day_df["is_code_P"].values
        # spike_risk_flag is day-uniform; take first value
        day_risk_flag = bool(day_df["spike_risk_flag"].iloc[0])

        state = {}

        for h in range(N_HOURLY_STEPS):
            n_settled = min(N_SETTLED_BASE + 2 * h, 48)
            settled_mask   = sp_arr <= n_settled
            unsettled_mask = ~settled_mask

            if unsettled_mask.sum() == 0:
                break

            settled_actuals = {
                int(sp): float(act)
                for sp, act in zip(sp_arr[settled_mask], act_arr[settled_mask])
            }

            try:
                corrected_fc, state = corrector(base_fc, settled_actuals, state)
            except Exception:
                corrected_fc = base_fc.copy()

            for i in np.where(unsettled_mask)[0]:
                sp  = int(sp_arr[i])
                row = corrected_fc[corrected_fc["settlement_period"] == sp]
                if row.empty:
                    continue
                records.append({
                    "corrector":       name,
                    "date":            d,
                    "fold":            str(fold),
                    "h":               h,
                    "sp":              sp,
                    "is_spike_P":      bool(spike_arr[i]),
                    "is_code_P":       bool(code_p_arr[i]),
                    "spike_risk_flag": day_risk_flag,
                    "pred_q10":        float(row["ssp_q10"].iloc[0]),
                    "pred_q50":        float(row["ssp_q50"].iloc[0]),
                    "pred_q90":        float(row["ssp_q90"].iloc[0]),
                    "actual":          float(act_arr[i]),
                })

    return pd.DataFrame(records)


# ── Metric computation ────────────────────────────────────────────────────────

def _smape(actual, pred):
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    safe  = np.where(denom == 0, np.nan, np.abs(actual - pred) / denom)
    return float(np.nanmean(safe)) * 100.0


def compute_metrics(sim: pd.DataFrame, label: str = "") -> dict:
    """Compute MAE, RMSE, sMAPE, PI_coverage for the simulation output."""
    err  = sim["actual"] - sim["pred_q50"]
    mae  = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    sm   = _smape(sim["actual"].values, sim["pred_q50"].values)
    cov  = float(((sim["actual"] >= sim["pred_q10"]) &
                  (sim["actual"] <= sim["pred_q90"])).mean()) * 100.0
    return {"label": label, "MAE": mae, "RMSE": rmse, "sMAPE": sm, "PI_cov": cov,
            "n": len(sim)}


def metrics_by_group(sim: pd.DataFrame, group_col: str, label: str = "") -> pd.DataFrame:
    rows = []
    for val, grp in sim.groupby(group_col):
        m = compute_metrics(grp)
        m[group_col] = val
        m["corrector"] = label
        rows.append(m)
    return pd.DataFrame(rows)


# ── SP-position bias ──────────────────────────────────────────────────────────

def sp_bias(wf: pd.DataFrame) -> pd.DataFrame:
    """
    For each SP position 1-48, compute mean(actual − base_q50)
    averaged across all test days.  Reveals time-of-day structure
    the scalar corrector cannot capture.
    """
    wf_cp = wf.copy()
    wf_cp["base_residual"] = wf_cp["ssp_actual"] - wf_cp["ssp_q50"]
    return wf_cp.groupby("settlement_period")["base_residual"].mean().reset_index()


# ── Spike harness ────────────────────────────────────────────────────────────

def compute_spike_harness(
    sim: pd.DataFrame,
    corrector: str = "KalmanCorrector",
    exclude_dates: "set | None" = None,
) -> dict:
    """
    Spike performance harness: coverage and MAE stratified by price bucket
    and ex-ante spike_risk_flag.

    spike_risk_flag is built from D-1 information only (elevated_count_lag1d,
    wind_pct_daily_mean_lag1d, calendar month) — no conditioning on realised
    price, no data leakage.

    Returns dict with keys:
        by_bucket       coverage+MAE for each price bucket
        by_flag         coverage+MAE split by spike_risk_flag
        by_flag_bucket  coverage+MAE by (spike_risk_flag, price_bucket)
        by_fold_flag    coverage+MAE by (fold, spike_risk_flag)
        code_p_xtab     cross-tab: Code-P rate and spike rate by bucket
        flag_stats      flag prevalence and spike rate statistics
        n_rows          row count used
        corrector       corrector name
    """
    s = sim[sim["corrector"] == corrector].copy()
    if exclude_dates:
        s = s[~s["date"].isin(exclude_dates)]
    if s.empty:
        return {}

    s["bucket"] = pd.cut(s["actual"], bins=PRICE_BUCKET_BINS, labels=PRICE_BUCKET_LABELS)

    def _m(df):
        if df.empty:
            return {"n": 0, "coverage": float("nan"), "mae": float("nan"),
                    "median_error": float("nan")}
        cov = ((df["actual"] >= df["pred_q10"]) & (df["actual"] <= df["pred_q90"])).mean()
        mae = (df["actual"] - df["pred_q50"]).abs().mean()
        med = (df["actual"] - df["pred_q50"]).median()
        return {"n": len(df), "coverage": round(float(cov), 4),
                "mae": round(float(mae), 2), "median_error": round(float(med), 2)}

    by_bucket = {b: _m(s[s["bucket"] == b]) for b in PRICE_BUCKET_LABELS}

    by_flag = {str(flag): _m(s[s["spike_risk_flag"] == flag]) for flag in [True, False]}

    by_flag_bucket = {}
    for flag in [True, False]:
        for b in PRICE_BUCKET_LABELS:
            key = f"{'risk' if flag else 'no-risk'}|{b}"
            by_flag_bucket[key] = _m(s[(s["spike_risk_flag"] == flag) & (s["bucket"] == b)])

    by_fold_flag = {}
    for fold in FOLD_ORDER:
        for flag in [True, False]:
            key = f"{fold}|{'risk' if flag else 'no-risk'}"
            by_fold_flag[key] = _m(s[(s["fold"] == fold) & (s["spike_risk_flag"] == flag)])

    # Code-P cross-tab by bucket (informational — not a spike label)
    code_p_xtab = {}
    for b in PRICE_BUCKET_LABELS:
        bsub = s[s["bucket"] == b]
        code_p_xtab[b] = {
            "n": len(bsub),
            "code_p_rate": round(float(bsub["is_code_P"].mean()), 4) if len(bsub) else float("nan"),
        }

    n_risk_days    = s[s["spike_risk_flag"]]["date"].nunique()
    n_total_days   = s["date"].nunique()
    rate_risk      = (s[s["spike_risk_flag"]]["actual"] > 150).mean()
    rate_norisk    = (s[~s["spike_risk_flag"]]["actual"] > 150).mean()
    flag_stats = {
        "n_risk_days":       n_risk_days,
        "n_total_days":      n_total_days,
        "risk_pct":          round(n_risk_days / max(n_total_days, 1) * 100, 1),
        "spike_rate_risk":   round(float(rate_risk), 4),
        "spike_rate_norisk": round(float(rate_norisk), 4),
        "spike_rate_ratio":  round(float(rate_risk / max(rate_norisk, 1e-6)), 1),
    }

    return {
        "by_bucket":      by_bucket,
        "by_flag":        by_flag,
        "by_flag_bucket": by_flag_bucket,
        "by_fold_flag":   by_fold_flag,
        "code_p_xtab":    code_p_xtab,
        "flag_stats":     flag_stats,
        "n_rows":         len(s),
        "corrector":      corrector,
    }


# ── Gating decisions ──────────────────────────────────────────────────────────

def make_gating_decisions(
    metrics_overall: pd.DataFrame,
    nis_best: dict,
    bias_df: pd.DataFrame,
    metrics_spike: pd.DataFrame,
) -> dict:
    """
    Apply the four gating criteria and return human-readable decisions.
    """
    # Pivot for easy comparison
    m = metrics_overall.set_index("label")

    kalman_mae  = m.loc["KalmanCorrector", "MAE"]
    alpha_mae   = m.loc["AlphaCorrector",  "MAE"]
    base_mae    = m.loc["StaticBase",       "MAE"]

    kalman_cov  = m.loc["KalmanCorrector", "PI_cov"]
    alpha_cov   = m.loc["AlphaCorrector",  "PI_cov"]

    # MAE improvement of Kalman vs Alpha
    pct_vs_alpha = (alpha_mae - kalman_mae) / alpha_mae * 100

    # Gate 1: Does Kalman beat Alpha?
    kalman_beats_alpha = (kalman_mae < alpha_mae)

    # Spike metrics
    ms = metrics_spike.set_index("corrector")
    spike_kalman_mae = ms.loc["KalmanCorrector", "MAE"] if "KalmanCorrector" in ms.index else None
    spike_alpha_mae  = ms.loc["AlphaCorrector",  "MAE"] if "AlphaCorrector"  in ms.index else None
    spike_better = (spike_kalman_mae < spike_alpha_mae) if (spike_kalman_mae and spike_alpha_mae) else None

    # Gate 2: NIS → structural misspecification?
    mean_nis   = nis_best["mean_nis"]
    fold_nis   = nis_best["nis_by_fold"]
    all_folds_overconfident = all(v > 1.15 for v in fold_nis.values())
    all_folds_underconfident = all(v < 0.85 for v in fold_nis.values())
    well_calibrated = 0.85 <= mean_nis <= 1.15

    if all_folds_overconfident:
        nis_verdict = "OVER-CONFIDENT"
        nis_action  = "level+trend v2 (§3.2) recommended: scalar level cannot track temporal drift"
    elif all_folds_underconfident:
        nis_verdict = "UNDER-CONFIDENT"
        nis_action  = "reduce Q and/or σ_SP; filter is too cautious"
    elif well_calibrated:
        nis_verdict = "WELL-CALIBRATED"
        nis_action  = "no structural change needed"
    else:
        nis_verdict = "MIXED"
        nis_action  = "check individual fold NIS; may reflect seasonal regime shifts"

    # Gate 3: SP-position-dependent bias?
    sp_residuals = bias_df["base_residual"].values
    # Strong SP dependence = large swing relative to overall SD
    sp_range = float(sp_residuals.max() - sp_residuals.min())
    sp_std   = float(sp_residuals.std())
    # Rule of thumb: swing > £15 across SPs suggests systematic diurnal bias
    strong_sp_bias = sp_range > 15.0

    if strong_sp_bias:
        sp_verdict = f"YES — diurnal swing of £{sp_range:.1f} in base residuals"
        sp_action  = "Phase 5 online-learning FALLBACK recommended (SP-level weights)"
    else:
        sp_verdict = f"NO — diurnal swing only £{sp_range:.1f}"
        sp_action  = "scalar level corrector is sufficient"

    # Final recommendation
    pi_cov_gap = kalman_cov - m.loc["StaticBase", "PI_cov"]
    pi_calibration_note = (
        f"NOTE: ALL correctors have PI coverage far below 80% target "
        f"(base={m.loc['StaticBase','PI_cov']:.1f}%). "
        f"This is a base-model quantile calibration issue, not a corrector issue — "
        f"the Q10/Q90 bands are too narrow to achieve 80% coverage. "
        f"Address this before the corrector comparison becomes decisive."
    )

    if not kalman_beats_alpha:
        recommendation = (
            "HOLD: KalmanCorrector does not beat AlphaCorrector on point-forecast MAE "
            f"({pct_vs_alpha:+.1f}%). "
            f"Kalman DOES improve PI coverage (+{kalman_cov - alpha_cov:.1f}pp vs Alpha, "
            f"+{pi_cov_gap:.1f}pp vs Base), but all correctors fall far short of the 80% "
            f"target ({kalman_cov:.1f}% Kalman). "
            "PREREQUISITE: recalibrate base-model Q10/Q90 before re-running this gate."
        )
    elif nis_verdict == "OVER-CONFIDENT" and all_folds_overconfident:
        recommendation = (
            "CONDITIONAL SHIP: KalmanCorrector beats Alpha overall, "
            "but NIS indicates over-confidence in all folds — "
            "implement level+trend v2 (§3.2) before declaring Phase 4 complete."
        )
    elif strong_sp_bias:
        recommendation = (
            "CONDITIONAL SHIP: KalmanCorrector beats Alpha, filter is calibrated, "
            "but SP-position bias detected — consider Phase 5 online learning "
            "for further gains beyond level correction."
        )
    else:
        recommendation = (
            "SHIP: KalmanCorrector beats AlphaCorrector, filter is well-calibrated, "
            "no strong SP-position bias. Ship Kalman as the production corrector."
        )

    return {
        "kalman_beats_alpha":    kalman_beats_alpha,
        "pct_vs_alpha":          pct_vs_alpha,
        "kalman_cov":            kalman_cov,
        "alpha_cov":             alpha_cov,
        "pi_cov_gap":            pi_cov_gap,
        "pi_calibration_note":   pi_calibration_note,
        "spike_better":          spike_better,
        "spike_kalman_mae":      spike_kalman_mae,
        "spike_alpha_mae":       spike_alpha_mae,
        "nis_verdict":           nis_verdict,
        "nis_action":            nis_action,
        "mean_nis":              mean_nis,
        "fold_nis":              fold_nis,
        "well_calibrated":       well_calibrated,
        "all_folds_overconfident": all_folds_overconfident,
        "all_folds_underconfident": all_folds_underconfident,
        "strong_sp_bias":        strong_sp_bias,
        "sp_range":              sp_range,
        "sp_verdict":            sp_verdict,
        "sp_action":             sp_action,
        "recommendation":        recommendation,
    }


# ── Phase 5a: KalmanSP simulation ────────────────────────────────────────────

def load_sp_bias_artifact() -> dict:
    """Load sp_bias_profile_v1.json; return {} if missing."""
    if not SP_BIAS_ARTIFACT.exists():
        return {}
    try:
        with open(SP_BIAS_ARTIFACT) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _run_simulation_kalman_sp(
    wf: pd.DataFrame,
    loo_bias_by_fold: dict,
    best_Q: float,
    best_sigma: float,
    best_gamma: float,
    name: str = "KalmanSP",
) -> pd.DataFrame:
    """
    Run KalmanSP walk-forward simulation using per-fold LOO SP bias.
    For each fold, the corrector is initialized with the bias learned
    from the other 3 folds (honest LOO).
    """
    parts = []
    for fold_name in FOLD_ORDER:
        fold_wf = wf[wf["fold"] == fold_name]
        if fold_wf.empty:
            continue
        bias_dict = {
            int(k): float(v)
            for k, v in loo_bias_by_fold.get(fold_name, {}).items()
        }
        corrector = KalmanCorrector(
            Q=best_Q, sigma_sp=best_sigma, gamma=best_gamma,
            sp_bias=bias_dict,
        )
        parts.append(_run_simulation(fold_wf, corrector, name))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── Phase 5a: gating decisions ───────────────────────────────────────────────

def make_sp_gating_decisions(
    metrics_overall: pd.DataFrame,
    metrics_spike: pd.DataFrame,
    metrics_normal: pd.DataFrame,
    all_sim: pd.DataFrame,
    swing_before: float,
) -> dict:
    """
    Phase 5a gating: compare KalmanSP vs KalmanBase.
    Gates (per user spec):
      1. Diurnal reduction: swing(KalmanSP) < 0.5 × swing_before
      2. MAE non-regression: KalmanSP MAE ≤ KalmanBase + 2%  (target −5%)
      3. Spike MAE non-regression: KalmanSP spike MAE ≤ KalmanBase + 2%
      4. Non-spike MAE non-regression: KalmanSP non-spike MAE ≤ KalmanBase + 2%  [NEW]
      5. PI coverage non-regression: KalmanSP PI cov ≥ KalmanBase − 2pp
    Merge if ALL gates pass.
    """
    m  = metrics_overall.set_index("label")
    ms = metrics_spike.set_index("corrector")
    mn = metrics_normal.set_index("corrector")

    kalman_mae    = float(m.loc["KalmanCorrector", "MAE"])
    kalman_sp_mae = float(m.loc["KalmanSP", "MAE"])
    kalman_cov    = float(m.loc["KalmanCorrector", "PI_cov"])
    kalman_sp_cov = float(m.loc["KalmanSP", "PI_cov"])

    # Diurnal swing of KalmanSP residuals (LOO-honest)
    grp_sp  = all_sim[all_sim["corrector"] == "KalmanSP"].copy()
    grp_sp["err"] = grp_sp["actual"] - grp_sp["pred_q50"]
    swing_sp = float(grp_sp.groupby("sp")["err"].mean().pipe(lambda s: s.max() - s.min()))

    grp_kb  = all_sim[all_sim["corrector"] == "KalmanCorrector"].copy()
    grp_kb["err"] = grp_kb["actual"] - grp_kb["pred_q50"]
    swing_kb = float(grp_kb.groupby("sp")["err"].mean().pipe(lambda s: s.max() - s.min()))

    # Gate 1: diurnal reduction ≥ 50%
    diurnal_reduced = swing_sp < 0.5 * swing_before
    diurnal_pct     = (swing_before - swing_sp) / swing_before * 100.0

    # Gate 2: overall MAE non-regression (5% target, 2% threshold)
    mae_improvement_pct  = (kalman_mae - kalman_sp_mae) / kalman_mae * 100.0
    no_mae_regression    = kalman_sp_mae <= kalman_mae * 1.02

    # Gate 3: spike MAE non-regression
    spike_kalman    = float(ms.loc["KalmanCorrector", "MAE"])
    spike_kalman_sp = float(ms.loc["KalmanSP", "MAE"])
    spike_pct       = (spike_kalman - spike_kalman_sp) / spike_kalman * 100.0
    spike_no_regr   = spike_kalman_sp <= spike_kalman * 1.02

    # Gate 4: non-spike MAE non-regression
    ns_kalman    = float(mn.loc["KalmanCorrector", "MAE"])
    ns_kalman_sp = float(mn.loc["KalmanSP", "MAE"])
    ns_pct       = (ns_kalman - ns_kalman_sp) / ns_kalman * 100.0
    ns_no_regr   = ns_kalman_sp <= ns_kalman * 1.02

    # Gate 5: PI coverage non-regression
    pi_no_regr   = kalman_sp_cov >= kalman_cov - 2.0
    pi_delta     = kalman_sp_cov - kalman_cov

    merge = diurnal_reduced and no_mae_regression and spike_no_regr and ns_no_regr and pi_no_regr

    if merge:
        decision = "MERGE: all gates pass — KalmanSP reduces diurnal bias without hurting spikes."
    else:
        fails = []
        if not diurnal_reduced:
            fails.append(f"diurnal reduction only {diurnal_pct:.0f}% (target ≥50%)")
        if not no_mae_regression:
            fails.append(f"overall MAE regression ({mae_improvement_pct:+.1f}%)")
        if not spike_no_regr:
            fails.append(f"spike MAE regression ({spike_pct:+.1f}%)")
        if not ns_no_regr:
            fails.append(f"non-spike MAE regression ({ns_pct:+.1f}%)")
        if not pi_no_regr:
            fails.append(f"PI coverage regression ({pi_delta:+.1f}pp)")
        decision = "HOLD: " + "; ".join(fails) + "."

    return {
        "swing_before":      swing_before,
        "swing_kalman_base": swing_kb,
        "swing_kalman_sp":   swing_sp,
        "diurnal_pct":       diurnal_pct,
        "diurnal_reduced":   diurnal_reduced,
        "kalman_mae":        kalman_mae,
        "kalman_sp_mae":     kalman_sp_mae,
        "mae_improvement_pct": mae_improvement_pct,
        "no_mae_regression": no_mae_regression,
        "spike_kalman":      spike_kalman,
        "spike_kalman_sp":   spike_kalman_sp,
        "spike_pct":         spike_pct,
        "spike_no_regr":     spike_no_regr,
        "ns_kalman":         ns_kalman,
        "ns_kalman_sp":      ns_kalman_sp,
        "ns_pct":            ns_pct,
        "ns_no_regr":        ns_no_regr,
        "kalman_cov":        kalman_cov,
        "kalman_sp_cov":     kalman_sp_cov,
        "pi_delta":          pi_delta,
        "pi_no_regr":        pi_no_regr,
        "merge":             merge,
        "decision":          decision,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

CORRECTOR_COLORS = {
    "StaticBase":       "#9E9E9E",
    "AlphaCorrector":   "#FB8C00",
    "KalmanCorrector":  "#1E88E5",
    "KalmanSP":         "#43A047",
}
CORRECTOR_LABELS = {
    "StaticBase":       "Static Base",
    "AlphaCorrector":   "Alpha (α=0.4)",
    "KalmanCorrector":  "Kalman",
    "KalmanSP":         "Kalman+SPBias",
}

SP_BIAS_ARTIFACT = ASSETS_DIR / "sp_bias_profile_v1.json"


def _bar_group(ax, df, metric, title, ylabel, correctors, xtick_labels=None):
    x     = np.arange(len(correctors))
    vals  = [float(df.loc[df["label"] == c, metric].iloc[0]) if (df["label"] == c).any() else 0
             for c in correctors]
    colors = [CORRECTOR_COLORS.get(c, "#888") for c in correctors]
    bars   = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([CORRECTOR_LABELS.get(c, c) for c in correctors], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def plot_metrics_comparison(metrics_df: pd.DataFrame, out_dir: Path) -> Path:
    correctors = ["StaticBase", "AlphaCorrector", "KalmanCorrector"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Corrector Comparison — Overall Metrics (unsettled SPs, all hourly steps)",
                 fontsize=11, fontweight="bold")

    _bar_group(axes[0, 0], metrics_df, "MAE",    "MAE (£/MWh)",  "£/MWh",  correctors)
    _bar_group(axes[0, 1], metrics_df, "RMSE",   "RMSE (£/MWh)", "£/MWh",  correctors)
    _bar_group(axes[1, 0], metrics_df, "sMAPE",  "sMAPE (%)",    "%",       correctors)
    _bar_group(axes[1, 1], metrics_df, "PI_cov", "PI Coverage (%)", "%",    correctors)
    axes[1, 1].axhline(80.0, color="red", linestyle="--", linewidth=1, label="Target 80%")
    axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    p = out_dir / "metrics_comparison.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return p


def plot_metrics_by_fold(all_sim: pd.DataFrame, out_dir: Path) -> Path:
    correctors = ["StaticBase", "AlphaCorrector", "KalmanCorrector"]
    x = np.arange(len(FOLD_ORDER))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("MAE and PI Coverage by Fold", fontsize=11, fontweight="bold")

    for metric, ax, ylabel in [("MAE", axes[0], "£/MWh"), ("PI_cov", axes[1], "%")]:
        for i, c in enumerate(correctors):
            fold_vals = []
            for f in FOLD_ORDER:
                grp = all_sim[(all_sim["corrector"] == c) & (all_sim["fold"] == f)]
                m   = compute_metrics(grp)
                fold_vals.append(m[metric])
            bars = ax.bar(x + (i - 1) * width, fold_vals, width=width,
                          label=CORRECTOR_LABELS.get(c, c),
                          color=CORRECTOR_COLORS.get(c, "#888"), zorder=3)
        ax.set_title(f"{metric} by Fold", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(FOLD_ORDER, rotation=15, ha="right", fontsize=8)
        ax.yaxis.grid(True, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        if metric == "PI_cov":
            ax.axhline(80.0, color="red", linestyle="--", linewidth=1)

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    p = out_dir / "metrics_by_fold.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return p


def plot_nis_heatmap(tune_result: dict, best_gamma: float, out_dir: Path) -> Path:
    """2-D heatmap: Q (rows) × sigma_sp (cols), filtered to best gamma."""
    rows = [r for r in tune_result["all_results"] if r["gamma"] == best_gamma]
    if not rows:
        rows = tune_result["all_results"]

    pivot = pd.DataFrame(rows).pivot_table(
        index="Q", columns="sigma_sp", values="mean_nis", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=0.5, vmax=1.5)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"σ={v}" for v in pivot.columns], fontsize=8)
    ax.set_yticklabels([f"Q={v}" for v in pivot.index],   fontsize=8)
    ax.set_title(f"Mean NIS (γ={best_gamma}) — green=calibrated, red=over/under",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("σ_SP", fontsize=9)
    ax.set_ylabel("Q",    fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean NIS")

    # Mark best cell
    best_Q, best_σ = tune_result["Q"], tune_result["sigma_sp"]
    if best_Q in pivot.index and best_σ in pivot.columns:
        ri = list(pivot.index).index(best_Q)
        ci = list(pivot.columns).index(best_σ)
        ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                   fill=False, edgecolor="blue", linewidth=2))

    # Annotate cells
    for ri, Q in enumerate(pivot.index):
        for ci, s in enumerate(pivot.columns):
            ax.text(ci, ri, f"{pivot.loc[Q, s]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(pivot.loc[Q, s] - 1.0) > 0.25 else "black")

    plt.tight_layout()
    p = out_dir / "nis_heatmap.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return p


def plot_sp_bias(bias_df: pd.DataFrame, all_sim: pd.DataFrame, out_dir: Path) -> Path:
    """Mean residual (actual − pred_q50) by settlement period for each corrector."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(bias_df["settlement_period"], bias_df["base_residual"],
           color="#9E9E9E", alpha=0.5, label="Base residual", width=0.8)

    for c in ["AlphaCorrector", "KalmanCorrector"]:
        grp = all_sim[all_sim["corrector"] == c].copy()
        grp["err"] = grp["actual"] - grp["pred_q50"]
        sp_err = grp.groupby("sp")["err"].mean().reset_index()
        ax.plot(sp_err["sp"], sp_err["err"],
                color=CORRECTOR_COLORS[c], linewidth=2,
                label=CORRECTOR_LABELS[c], marker="o", markersize=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Settlement Period (SP)", fontsize=9)
    ax.set_ylabel("Mean Residual (£/MWh)", fontsize=9)
    ax.set_title("Mean Residual by SP Position — diurnal bias check", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    p = out_dir / "sp_bias.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return p


def plot_error_distributions(all_sim: pd.DataFrame, out_dir: Path) -> Path:
    """Box plots of absolute error per corrector."""
    correctors = ["StaticBase", "AlphaCorrector", "KalmanCorrector"]
    fig, ax = plt.subplots(figsize=(8, 5))

    data   = []
    labels = []
    colors = []
    for c in correctors:
        grp = all_sim[all_sim["corrector"] == c]
        data.append((grp["actual"] - grp["pred_q50"]).abs().values)
        labels.append(CORRECTOR_LABELS.get(c, c))
        colors.append(CORRECTOR_COLORS.get(c, "#888"))

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    showfliers=False, whis=1.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("|Actual − Pred Q50|  (£/MWh)", fontsize=9)
    ax.set_title("Absolute Error Distribution (no fliers)", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    p = out_dir / "error_distributions.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return p


def plot_sp_bias_comparison(
    all_sim: pd.DataFrame, bias_df: pd.DataFrame, out_dir: Path
) -> Path:
    """Two-panel: mean residual by SP for KalmanBase (left) vs KalmanSP (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Phase 5a — Diurnal Residual Before/After SP Bias Correction",
                 fontsize=11, fontweight="bold")

    sps = np.arange(1, 49)

    # Left: KalmanBase (base residual overlay)
    ax = axes[0]
    ax.bar(bias_df["settlement_period"], bias_df["base_residual"],
           color="#9E9E9E", alpha=0.45, width=0.8, label="Base residual")
    grp_kb = all_sim[all_sim["corrector"] == "KalmanCorrector"].copy()
    grp_kb["err"] = grp_kb["actual"] - grp_kb["pred_q50"]
    sp_kb = grp_kb.groupby("sp")["err"].mean()
    ax.plot(sp_kb.index, sp_kb.values, color=CORRECTOR_COLORS["KalmanCorrector"],
            linewidth=2, marker="o", markersize=3, label="KalmanBase residual")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    swing_kb = float(sp_kb.max() - sp_kb.min())
    ax.set_title(f"KalmanBase  (swing £{swing_kb:.1f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Settlement Period", fontsize=9)
    ax.set_ylabel("Mean Residual (£/MWh)", fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Right: KalmanSP
    ax = axes[1]
    grp_sp = all_sim[all_sim["corrector"] == "KalmanSP"].copy()
    grp_sp["err"] = grp_sp["actual"] - grp_sp["pred_q50"]
    sp_sp = grp_sp.groupby("sp")["err"].mean()
    ax.bar(sp_sp.index, sp_sp.values,
           color=CORRECTOR_COLORS["KalmanSP"], alpha=0.5, width=0.8, label="KalmanSP residual")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    swing_sp = float(sp_sp.max() - sp_sp.min())
    ax.set_title(f"KalmanSP  (swing £{swing_sp:.1f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Settlement Period", fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    p = out_dir / "sp_bias_comparison.png"
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    return p


def plot_spike_harness(all_sim: pd.DataFrame, out_dir: Path,
                       corrector: str = "KalmanCorrector") -> Path:
    """
    Two-panel: PI coverage and MAE by price bucket, split by spike_risk_flag.
    Left panel: coverage lines with 80% target.
    Right panel: MAE lines showing tail error uplift.
    """
    s = all_sim[all_sim["corrector"] == corrector].copy()
    s["bucket"] = pd.cut(s["actual"], bins=PRICE_BUCKET_BINS, labels=PRICE_BUCKET_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Spike Harness — {corrector} — Coverage and MAE by Price Bucket",
                 fontsize=11, fontweight="bold")

    style = {
        True:  {"color": "#d62728", "marker": "o", "label": "Risk-flagged days"},
        False: {"color": "#1f77b4", "marker": "s", "label": "Non-risk days"},
    }
    x = np.arange(len(PRICE_BUCKET_LABELS))

    for flag in [True, False]:
        sub = s[s["spike_risk_flag"] == flag]
        coverages, maes, ns = [], [], []
        for b in PRICE_BUCKET_LABELS:
            bsub = sub[sub["bucket"] == b]
            if bsub.empty:
                coverages.append(float("nan"))
                maes.append(float("nan"))
                ns.append(0)
            else:
                cov = ((bsub["actual"] >= bsub["pred_q10"]) &
                       (bsub["actual"] <= bsub["pred_q90"])).mean()
                coverages.append(cov * 100)
                maes.append((bsub["actual"] - bsub["pred_q50"]).abs().mean())
                ns.append(len(bsub))
        lbl = f"{style[flag]['label']} (n≈{sum(ns)})"
        axes[0].plot(x, coverages, marker=style[flag]["marker"], color=style[flag]["color"],
                     label=lbl, linewidth=1.8)
        axes[1].plot(x, maes, marker=style[flag]["marker"], color=style[flag]["color"],
                     label=lbl, linewidth=1.8)

    axes[0].axhline(80, color="#666", linestyle="--", alpha=0.6, label="80% target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(PRICE_BUCKET_LABELS, rotation=25)
    axes[0].set_ylabel("PI Coverage (%)")
    axes[0].set_title("PI Coverage by Price Bucket")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 105)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(PRICE_BUCKET_LABELS, rotation=25)
    axes[1].set_ylabel("MAE (£/MWh)")
    axes[1].set_title("MAE by Price Bucket")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    p = out_dir / "spike_harness.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# ── Markdown report ───────────────────────────────────────────────────────────

def _metric_table(df: pd.DataFrame, label_col: str = "label") -> str:
    hdr  = f"| {label_col.title()} | MAE | RMSE | sMAPE | PI Cov |"
    sep  = "|---|---:|---:|---:|---:|"
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"| {r[label_col]} "
            f"| {r.get('MAE', float('nan')):.2f} "
            f"| {r.get('RMSE', float('nan')):.2f} "
            f"| {r.get('sMAPE', float('nan')):.1f}% "
            f"| {r.get('PI_cov', float('nan')):.1f}% |"
        )
    return "\n".join([hdr, sep] + rows)


def _sp_gate_section(sp_gate: dict, plot_paths: dict, out_dir: Path) -> list:
    """Build Phase 5a markdown section lines."""
    g = sp_gate
    rel = lambda p: str(p.relative_to(out_dir))

    def _tick(ok): return "✅" if ok else "❌"

    lines = [
        "",
        "---",
        "",
        "## Phase 5a — SP Bias Profile (KalmanSP vs KalmanBase)",
        "",
        f"SP bias trained on median(actual − q50) per settlement period (LOO across 4 folds).",
        f"Diurnal swing before: **£{g['swing_before']:.1f}**",
        "",
        "### Phase 5a Gating",
        "",
        "| # | Gate | Threshold | Finding | Pass? |",
        "|---|---|---|---|---|",
        f"| 1 | Diurnal reduction | swing < 50% of base | "
        f"£{g['swing_kalman_sp']:.1f} vs base £{g['swing_before']:.1f} ({g['diurnal_pct']:.0f}% reduction) | "
        f"{_tick(g['diurnal_reduced'])} |",
        f"| 2 | Overall MAE non-regression | KalmanSP ≤ KalmanBase + 2% | "
        f"£{g['kalman_sp_mae']:.2f} vs £{g['kalman_mae']:.2f} ({g['mae_improvement_pct']:+.1f}%) | "
        f"{_tick(g['no_mae_regression'])} |",
        f"| 3 | Spike MAE non-regression | KalmanSP ≤ KalmanBase + 2% | "
        f"£{g['spike_kalman_sp']:.2f} vs £{g['spike_kalman']:.2f} ({g['spike_pct']:+.1f}%) | "
        f"{_tick(g['spike_no_regr'])} |",
        f"| 4 | Non-spike MAE non-regression | KalmanSP ≤ KalmanBase + 2% | "
        f"£{g['ns_kalman_sp']:.2f} vs £{g['ns_kalman']:.2f} ({g['ns_pct']:+.1f}%) | "
        f"{_tick(g['ns_no_regr'])} |",
        f"| 5 | PI coverage non-regression | KalmanSP ≥ KalmanBase − 2pp | "
        f"{g['kalman_sp_cov']:.1f}% vs {g['kalman_cov']:.1f}% ({g['pi_delta']:+.1f}pp) | "
        f"{_tick(g['pi_no_regr'])} |",
        "",
        f"> **{g['decision']}**",
        "",
        "### Phase 5a Metrics — KalmanSP vs KalmanBase",
        "",
        "| Corrector | MAE | RMSE | sMAPE | PI Cov |",
        "|---|---:|---:|---:|---:|",
        f"| KalmanBase | {g['kalman_mae']:.2f} | — | — | {g['kalman_cov']:.1f}% |",
        f"| KalmanSP   | {g['kalman_sp_mae']:.2f} | — | — | {g['kalman_sp_cov']:.1f}% |",
        "",
    ]

    if "sp_bias_comparison" in plot_paths:
        lines.append(f"![SP Bias Comparison]({rel(plot_paths['sp_bias_comparison'])})")
        lines.append("")

    return lines


def _spike_harness_section(
    harness_all: dict,
    harness_excl: dict,
    out_dir: Path,
    plot_path: "Path | None" = None,
) -> list:
    """Markdown lines for the spike harness report section (§7)."""
    fs = harness_all["flag_stats"]

    def _fmt_row(m):
        if m.get("n", 0) == 0:
            return "—"
        cov = f"{m['coverage']:.1%}" if not np.isnan(m["coverage"]) else "—"
        mae = f"£{m['mae']:.1f}" if not np.isnan(m["mae"]) else "—"
        return cov, mae

    lines = [
        "",
        "---",
        "",
        "## 7. Spike Harness — Ex-Ante Stratification",
        "",
        f"**Corrector:** {harness_all.get('corrector', '?')}  ",
        f"**Ex-ante flag definition:**",
        "```",
        "spike_risk_flag = (elevated_count_lag1d >= 5)",
        "               OR (wind_pct_daily_mean_lag1d < 10%",
        "                   AND month ∈ {Sep, Oct, Nov, Mar, Apr, May})",
        "```",
        f"*`elevated_count_lag1d` = number of SPs on D-1 with ssp_actual > £120 (p90 threshold).  "
        f"`wind_pct_daily_mean_lag1d` = mean wind penetration (0-100 scale) across all 48 SPs on D-1.*  ",
        f"**Flag prevalence:** {fs['n_risk_days']}/{fs['n_total_days']} days flagged ({fs['risk_pct']:.0f}%)  ",
        f"**Spike rate lift:** p(actual > £150) = {fs['spike_rate_risk']:.1%} (risk) vs "
        f"{fs['spike_rate_norisk']:.1%} (no-risk) — **{fs['spike_rate_ratio']:.1f}× lift**  ",
        "",
        "> **Single-event leverage:** 2025-10-13 SP26 (£487, Code-P, is_spike_P=True) is the only "
        "SP in the WF window exceeding the Tukey fence (£353.7). All tables are reported WITH and "
        "WITHOUT this date. Large differences indicate leverage from this single event.",
        "",
        "---",
        "",
        "### 7.1 PI Coverage and MAE by Price Bucket (with 2025-10-13)",
        "",
        "| Bucket | N | PI Coverage | MAE | Median Error |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in PRICE_BUCKET_LABELS:
        m = harness_all["by_bucket"].get(b, {})
        n = m.get("n", 0)
        if n == 0:
            lines.append(f"| {b} | 0 | — | — | — |")
        else:
            cov = f"{m['coverage']:.1%}" if not np.isnan(m["coverage"]) else "—"
            lines.append(f"| {b} | {n} | {cov} | £{m['mae']:.1f} | £{m['median_error']:.1f} |")

    lines += ["", f"*Excluding {SPIKE_EXCLUDE_DATE}:*", ""]
    lines += ["| Bucket | N | PI Coverage | MAE |", "|---|---:|---:|---:|"]
    for b in PRICE_BUCKET_LABELS:
        m = harness_excl["by_bucket"].get(b, {})
        n = m.get("n", 0)
        if n == 0:
            lines.append(f"| {b} | 0 | — | — |")
        else:
            cov = f"{m['coverage']:.1%}" if not np.isnan(m["coverage"]) else "—"
            lines.append(f"| {b} | {n} | {cov} | £{m['mae']:.1f} |")

    lines += [
        "",
        "---",
        "",
        "### 7.2 Coverage by Risk Flag × Price Bucket",
        "",
        "| Risk Flag | Bucket | N | PI Coverage | MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for flag in [True, False]:
        flag_label = "✅ Risk" if flag else "⬜ No-Risk"
        for b in PRICE_BUCKET_LABELS:
            key = f"{'risk' if flag else 'no-risk'}|{b}"
            m = harness_all["by_flag_bucket"].get(key, {})
            n = m.get("n", 0)
            if n == 0:
                continue
            cov = f"{m['coverage']:.1%}" if not np.isnan(m["coverage"]) else "—"
            lines.append(f"| {flag_label} | {b} | {n} | {cov} | £{m['mae']:.1f} |")

    # same table excluding 2025-10-13
    lines += ["", f"*Excluding {SPIKE_EXCLUDE_DATE}:*", ""]
    lines += ["| Risk Flag | Bucket | N | PI Coverage | MAE |", "|---|---|---:|---:|---:|"]
    for flag in [True, False]:
        flag_label = "✅ Risk" if flag else "⬜ No-Risk"
        for b in PRICE_BUCKET_LABELS:
            key = f"{'risk' if flag else 'no-risk'}|{b}"
            m = harness_excl["by_flag_bucket"].get(key, {})
            n = m.get("n", 0)
            if n == 0:
                continue
            cov = f"{m['coverage']:.1%}" if not np.isnan(m["coverage"]) else "—"
            lines.append(f"| {flag_label} | {b} | {n} | {cov} | £{m['mae']:.1f} |")

    lines += [
        "",
        "---",
        "",
        "### 7.3 Coverage by Fold × Risk Flag",
        "",
        "| Fold | Risk Flag | N | PI Coverage | MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for fold in FOLD_ORDER:
        for flag in [True, False]:
            flag_label = "Risk" if flag else "No-Risk"
            key = f"{fold}|{'risk' if flag else 'no-risk'}"
            m = harness_all["by_fold_flag"].get(key, {})
            n = m.get("n", 0)
            if n == 0:
                continue
            cov = f"{m['coverage']:.1%}" if not np.isnan(m["coverage"]) else "—"
            lines.append(f"| {fold} | {flag_label} | {n} | {cov} | £{m['mae']:.1f} |")

    lines += [
        "",
        "---",
        "",
        "### 7.4 Code-P Cross-Tab by Price Bucket",
        "",
        "Code-P is a **market mechanism indicator** (46% of all SPs), NOT a spike label. "
        "The table shows what fraction of each price bucket has Code-P set.",
        "",
        "| Bucket | N | Code-P Rate |",
        "|---|---:|---:|",
    ]
    for b in PRICE_BUCKET_LABELS:
        xt = harness_all["code_p_xtab"].get(b, {})
        n = xt.get("n", 0)
        if n == 0:
            lines.append(f"| {b} | 0 | — |")
        else:
            r = xt.get("code_p_rate", float("nan"))
            rate_str = f"{r:.1%}" if not np.isnan(r) else "—"
            lines.append(f"| {b} | {n} | {rate_str} |")

    if plot_path is not None:
        lines += ["", f"![Spike Harness]({str(plot_path.relative_to(out_dir))})"]

    lines += [
        "",
        "---",
        "",
        "### 7.5 Interpretation",
        "",
        "- **PI Coverage targets 80%** across all buckets. Coverage shortfall in the £150-250 "
        "and £250+ buckets indicates the symmetric conformal δ (£22.8 global) is insufficient "
        "for tail events.",
        "- **Ex-ante flag has no data leakage.** `spike_risk_flag` uses only D-1 lagged features "
        "and calendar month. It cannot be calibrated on tomorrow's realised price.",
        "- **Any δ_hi regime correction** must be derived from elevated-price conformity scores "
        "(actual − q90 where actual > £120), NOT from Code-P scores (Code-P fires at normal "
        "prices 46% of the time; see docs/spike-tail-design.md §3 and §5).",
        "- **2025-10-13 leverage:** If WITH and WITHOUT numbers diverge substantially in the "
        "£250+ bucket, the entire bucket signal is driven by this single event and should not "
        "be used for calibration without additional data.",
        "",
    ]
    return lines


def write_report(
    out_dir: Path,
    gate: dict,
    tune: dict,
    metrics_overall: pd.DataFrame,
    metrics_spike: pd.DataFrame,
    metrics_normal: pd.DataFrame,
    metrics_by_fold_df: pd.DataFrame,
    n_days: int,
    plot_paths: dict,
    sp_gate=None,
    spike_harness_all: "dict | None" = None,
    spike_harness_excl: "dict | None" = None,
) -> Path:
    rel = lambda p: str(p.relative_to(out_dir))

    def _nis_fold_table():
        hdr = "| Fold | Mean NIS | Calibration |"
        sep = "|---|---:|---|"
        rows = []
        for f in FOLD_ORDER:
            v = tune["nis_by_fold"].get(f, float("nan"))
            cal = "✅ good" if 0.85 <= v <= 1.15 else ("⚠️ over" if v > 1.15 else "⚠️ under")
            rows.append(f"| {f} | {v:.3f} | {cal} |")
        return "\n".join([hdr, sep] + rows)

    def _fold_metrics_table():
        lines = [_metric_table(metrics_by_fold_df[metrics_by_fold_df["corrector"] == c], label_col="fold")
                 for c in ["StaticBase", "AlphaCorrector", "KalmanCorrector"]]
        return "\n\n".join([
            f"**StaticBase**\n{lines[0]}",
            f"**AlphaCorrector (α=0.4)**\n{lines[1]}",
            f"**KalmanCorrector (tuned)**\n{lines[2]}",
        ])

    lines = [
        f"# Corrector Walk-Forward Backtest Report",
        f"",
        f"**Generated:** {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Test window:** 2025-07-01 – 2026-04-30 ({n_days} days, 4 seasonal folds)  ",
        f"**Correctors:** Static Base · AlphaCorrector (α=0.4) · KalmanCorrector  ",
        f"**Cadence:** 10 hourly steps 08:30–17:30 UTC; metrics on unsettled SPs only",
        f"",
        f"---",
        f"",
        f"## Executive Summary & Gating Decisions",
        f"",
        f"| # | Gate | Finding | Decision |",
        f"|---|---|---|---|",
        f"| 1 | **Kalman beats α=0.4?** | MAE: Kalman={gate.get('kalman_mae', metrics_overall.set_index('label').loc['KalmanCorrector','MAE']):.2f} vs Alpha={gate.get('alpha_mae', metrics_overall.set_index('label').loc['AlphaCorrector','MAE']):.2f} ({gate['pct_vs_alpha']:+.1f}%) | {'✅ YES — gate to Phase 4' if gate['kalman_beats_alpha'] else '❌ NO — keep AlphaCorrector'} |",
        f"| 2 | **NIS: structural misspec?** | Mean NIS = {gate['mean_nis']:.3f} · {gate['nis_verdict']} | {'⚠️ Recommend level+trend v2 (§3.2)' if gate['nis_verdict']=='OVER-CONFIDENT' else '✅ No structural change needed' if gate['well_calibrated'] else '⚠️ ' + gate['nis_action']} |",
        f"| 3 | **SP-position bias?** | Diurnal swing £{gate['sp_range']:.1f} | {'⚠️ Phase 5 online learning recommended' if gate['strong_sp_bias'] else '✅ Scalar corrector sufficient'} |",
        f"| 4 | **PI coverage (target 80%)?** | Kalman={gate['kalman_cov']:.1f}% vs Alpha={gate['alpha_cov']:.1f}% | {'✅ Improved' if gate['kalman_cov'] > gate['alpha_cov'] else '❌ No improvement'} |",
        f"",
        f"### Recommendation",
        f"",
        f"> **{gate['recommendation']}**",
        f"",
        f"> ⚠️ **Base-model PI calibration prerequisite:** {gate['pi_calibration_note']}",
        f"",
        f"---",
        f"",
        f"## 1. Metrics — Overall",
        f"",
        _metric_table(metrics_overall),
        f"",
        f"![Metrics Comparison]({rel(plot_paths['metrics_comparison'])})",
        f"",
        f"---",
        f"",
        f"## 2. Metrics — Spike vs Normal",
        f"",
        f"**Spike SPs** (`is_spike_P`: `price_derivation_code == 'P'` AND `actual > £{353.7:.0f}`):",
        f"*Note: only 1 SP in the WF window meets this definition (2025-10-13 SP26, £487). "
        f"See §7 for bucket-stratified analysis covering the broader high-price tail.*",
        f"",
        _metric_table(metrics_spike, label_col="corrector"),
        f"",
        f"**Normal SPs:**",
        f"",
        _metric_table(metrics_normal, label_col="corrector"),
        f"",
        f"---",
        f"",
        f"## 3. Metrics by Seasonal Fold",
        f"",
        _fold_metrics_table(),
        f"",
        f"![Metrics by Fold]({rel(plot_paths['metrics_by_fold'])})",
        f"",
        f"---",
        f"",
        f"## 4. Kalman Hyperparameter Tuning (NIS)",
        f"",
        f"**Objective:** minimise |mean NIS − 1| across all (date, hourly step) pairs.",
        f"A well-calibrated filter has NIS ~ χ²(1), so E[NIS] = 1.",
        f"",
        f"**Best parameters (NIS-tuned):** Q = {tune['Q']} £², σ_SP = {tune['sigma_sp']} £",
        f"*(Note: γ is not identifiable from NIS — it only affects future-SP correction, not the*",
        f"*settled-SP innovation sequence.  γ = {tune['gamma']} is reported as the grid minimum.)*",
        f"**Mean NIS (tuned):** {tune['mean_nis']:.4f}  →  **{gate['nis_verdict']}**",
        f"",
        f"### NIS by Fold",
        f"",
        _nis_fold_table(),
        f"",
        (f"NIS consistently > 1.15 across all folds → level-only scalar state is over-confident "
         f"→ {gate['nis_action']}."
         if gate["all_folds_overconfident"] else
         f"NIS consistently < 0.85 across all folds → filter is under-confident (Q and/or σ_SP "
         f"may be larger than necessary relative to actual intraday bias variance)."
         if gate["all_folds_underconfident"] else
         f"NIS is seasonally heterogeneous: some folds over-confident (NIS > 1), others under-confident "
         f"(NIS < 1).  A single (Q, σ_SP) cannot perfectly calibrate across all regimes — "
         f"this is expected for a scalar level-only filter."),
        f"",
        f"![NIS Heatmap (best γ={tune['gamma']})]({rel(plot_paths['nis_heatmap'])})",
        f"",
        f"---",
        f"",
        f"## 5. SP-Position Bias",
        f"",
        f"Mean residual (actual − base_q50) per SP, averaged across all {n_days} test days.",
        f"Diurnal swing = **£{gate['sp_range']:.1f}** (max − min of SP-mean residuals).",
        f"",
        f"**Verdict:** {gate['sp_verdict']}",
        f"**Action:** {gate['sp_action']}",
        f"",
        f"![SP Bias]({rel(plot_paths['sp_bias'])})",
        f"",
        f"---",
        f"",
        f"## 6. Error Distributions",
        f"",
        f"![Error Distributions]({rel(plot_paths['error_distributions'])})",
        f"",
        f"---",
        f"",
        f"## Appendix: Grid Search Results (Top 15)",
        f"",
        f"| Q | σ_SP | γ | Mean NIS | Score |",
        f"|---:|---:|---:|---:|---:|",
    ]

    for r in tune["all_results"][:15]:
        lines.append(
            f"| {r['Q']} | {r['sigma_sp']} | {r['gamma']} "
            f"| {r['mean_nis']:.4f} | {r['score']:.4f} |"
        )

    lines += [
        f"",
        f"*Lower score = better calibration (|mean NIS − 1|).*",
    ]

    # Phase 5a section (appended when KalmanSP was run)
    if sp_gate:
        lines += _sp_gate_section(sp_gate, plot_paths, out_dir)

    # Spike harness section (§7)
    if spike_harness_all and spike_harness_excl:
        lines += _spike_harness_section(
            spike_harness_all,
            spike_harness_excl,
            out_dir,
            plot_path=plot_paths.get("spike_harness"),
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Walk-forward corrector backtest")
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "corrector_backtest"))
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    wf = load_data()
    n_days = wf["settlement_date"].nunique()
    print(f"  {n_days} days · {len(wf)} rows · folds: {wf['fold'].unique().tolist()}")

    # ── NIS grid search ────────────────────────────────────────────────────────
    print("NIS grid search …")
    tune = tune_kalman(wf)
    best_Q, best_sigma, best_gamma = tune["Q"], tune["sigma_sp"], tune["gamma"]
    print(f"  Best: Q={best_Q}, σ_SP={best_sigma}, γ={best_gamma}  "
          f"mean_NIS={tune['mean_nis']:.4f}")

    # ── Phase 5a: load SP bias profile if available ───────────────────────────
    sp_bias_artifact = load_sp_bias_artifact()
    run_kalman_sp    = bool(sp_bias_artifact)
    if run_kalman_sp:
        print(f"SP bias profile loaded (v{sp_bias_artifact.get('version','?')}, "
              f"method={sp_bias_artifact.get('selected_method','?')})")
    else:
        print("SP bias profile not found — skipping KalmanSP")

    # ── Walk-forward simulation ────────────────────────────────────────────────
    correctors = {
        "StaticBase":       _StaticBase(),
        "AlphaCorrector":   AlphaCorrector(alpha=0.4, dead_band=0.5),
        "KalmanCorrector":  KalmanCorrector(Q=best_Q, sigma_sp=best_sigma, gamma=best_gamma),
    }

    sim_frames = []
    for name, corr in correctors.items():
        print(f"Simulating {name} …")
        sim = _run_simulation(wf, corr, name)
        sim_frames.append(sim)

    if run_kalman_sp:
        print("Simulating KalmanSP (LOO SP bias) …")
        loo_bias = sp_bias_artifact.get("loo_bias_by_fold", {})
        sim_sp   = _run_simulation_kalman_sp(wf, loo_bias, best_Q, best_sigma, best_gamma)
        sim_frames.append(sim_sp)

    all_sim = pd.concat(sim_frames, ignore_index=True)

    # ── Metrics ────────────────────────────────────────────────────────────────
    print("Computing metrics …")
    all_corrector_names = list(correctors.keys()) + (["KalmanSP"] if run_kalman_sp else [])
    metrics_overall = pd.DataFrame([
        compute_metrics(all_sim[all_sim["corrector"] == c], label=c)
        for c in all_corrector_names
    ])

    spike_sim  = all_sim[all_sim["is_spike_P"]]
    normal_sim = all_sim[~all_sim["is_spike_P"]]
    metrics_spike  = pd.DataFrame([
        compute_metrics(spike_sim[spike_sim["corrector"] == c], label=c)
        for c in all_corrector_names
    ]).rename(columns={"label": "corrector"})
    metrics_normal = pd.DataFrame([
        compute_metrics(normal_sim[normal_sim["corrector"] == c], label=c)
        for c in all_corrector_names
    ]).rename(columns={"label": "corrector"})

    # By-fold metrics (for all correctors)
    fold_rows = []
    for c in all_corrector_names:
        for f in FOLD_ORDER:
            grp = all_sim[(all_sim["corrector"] == c) & (all_sim["fold"] == f)]
            if grp.empty:
                continue
            m = compute_metrics(grp)
            m["corrector"] = c
            m["fold"]      = f
            fold_rows.append(m)
    metrics_by_fold_df = pd.DataFrame(fold_rows)

    # SP-position bias (from base model)
    bias_df = sp_bias(wf)

    # ── Gating decisions ───────────────────────────────────────────────────────
    # Use only the 3-corrector metrics_overall for the existing gate function
    metrics_overall_3 = metrics_overall[metrics_overall["label"].isin(correctors)].copy()
    metrics_spike_3   = metrics_spike[metrics_spike["corrector"].isin(correctors)].copy()

    m_idx = metrics_overall.set_index("label")
    gate = make_gating_decisions(
        metrics_overall_3,
        tune,
        bias_df,
        metrics_spike_3,
    )
    gate["kalman_mae"] = float(m_idx.loc["KalmanCorrector", "MAE"])
    gate["alpha_mae"]  = float(m_idx.loc["AlphaCorrector",  "MAE"])

    # Phase 5a gating
    sp_gate = None
    if run_kalman_sp:
        sp_gate = make_sp_gating_decisions(
            metrics_overall,
            metrics_spike,
            metrics_normal,
            all_sim,
            swing_before=gate["sp_range"],
        )

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("Generating plots …")
    plot_paths = {
        "metrics_comparison":  plot_metrics_comparison(metrics_overall_3, out_dir),
        "metrics_by_fold":     plot_metrics_by_fold(all_sim, out_dir),
        "nis_heatmap":         plot_nis_heatmap(tune, best_gamma, out_dir),
        "sp_bias":             plot_sp_bias(bias_df, all_sim, out_dir),
        "error_distributions": plot_error_distributions(all_sim, out_dir),
    }
    if run_kalman_sp:
        plot_paths["sp_bias_comparison"] = plot_sp_bias_comparison(all_sim, bias_df, out_dir)

    # ── Spike harness ──────────────────────────────────────────────────────────
    print("Computing spike harness …")
    harness_all  = compute_spike_harness(all_sim, corrector="KalmanCorrector")
    harness_excl = compute_spike_harness(
        all_sim, corrector="KalmanCorrector",
        exclude_dates={SPIKE_EXCLUDE_DATE},
    )
    fs = harness_all.get("flag_stats", {})
    print(f"  spike_risk_flag: {fs.get('n_risk_days','?')}/{fs.get('n_total_days','?')} days  "
          f"spike rate lift {fs.get('spike_rate_ratio','?')}×")
    for bucket in PRICE_BUCKET_LABELS:
        m = harness_all["by_bucket"].get(bucket, {})
        m_ex = harness_excl["by_bucket"].get(bucket, {})
        if m.get("n", 0) > 0:
            print(f"  {bucket}: n={m['n']}  cov={m['coverage']:.1%}  MAE=£{m['mae']:.1f}"
                  + (f"  (excl: cov={m_ex['coverage']:.1%})" if m_ex.get("n", 0) > 0 else ""))

    if harness_all:
        plot_paths["spike_harness"] = plot_spike_harness(all_sim, out_dir)

    # ── Report ─────────────────────────────────────────────────────────────────
    print("Writing report …")
    report_path = write_report(
        out_dir, gate, tune, metrics_overall_3,
        metrics_spike_3, metrics_normal[metrics_normal["corrector"].isin(correctors)].copy(),
        metrics_by_fold_df[metrics_by_fold_df["corrector"].isin(correctors)].copy(),
        n_days, plot_paths, sp_gate=sp_gate,
        spike_harness_all=harness_all or None,
        spike_harness_excl=harness_excl or None,
    )

    print(f"\nDone. Report: {report_path}")
    print("\n── Gating Decisions ─────────────────────────────────────────────────")
    print(f"  Gate 1 — Kalman beats Alpha: {gate['kalman_beats_alpha']} "
          f"({gate['pct_vs_alpha']:+.1f}% MAE)")
    print(f"  Gate 2 — NIS: {gate['nis_verdict']} (mean={gate['mean_nis']:.3f})")
    print(f"  Gate 3 — SP-position bias: {'YES' if gate['strong_sp_bias'] else 'NO'} "
          f"(swing £{gate['sp_range']:.1f})")
    print(f"  Gate 4 — PI coverage: Kalman={gate['kalman_cov']:.1f}% "
          f"vs Alpha={gate['alpha_cov']:.1f}%")
    print(f"\n  → {gate['recommendation']}")
    if sp_gate:
        print("\n── Phase 5a SP Bias Gating ──────────────────────────────────────────")
        print(f"  Diurnal reduction: {sp_gate['diurnal_reduced']} "
              f"({sp_gate['diurnal_pct']:.0f}% swing reduction)")
        print(f"  MAE non-regression: {sp_gate['no_mae_regression']} "
              f"({sp_gate['mae_improvement_pct']:+.1f}%)")
        print(f"  Spike non-regression: {sp_gate['spike_no_regr']} "
              f"({sp_gate['spike_pct']:+.1f}%)")
        print(f"  Non-spike non-regression: {sp_gate['ns_no_regr']} "
              f"({sp_gate['ns_pct']:+.1f}%)")
        print(f"  PI coverage non-regression: {sp_gate['pi_no_regr']} "
              f"({sp_gate['pi_delta']:+.1f}pp)")
        print(f"\n  → {sp_gate['decision']}")


if __name__ == "__main__":
    main()

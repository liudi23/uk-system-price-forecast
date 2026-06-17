"""
Swappable intraday bias-correction layer for the Phase 3 forecast.

Interface
---------
A corrector is any callable satisfying:

    correct(forecast_df, settled_actuals, state) -> (corrected_df, new_state)

where
    forecast_df      : pd.DataFrame with 48 SP rows and columns including
                       settlement_period, ssp_predicted, ssp_q10, ssp_q50, ssp_q90.
                       Must NOT yet have settled SPs replaced by actuals.
    settled_actuals  : dict[int, float]  {settlement_period -> actual SSP £/MWh}
                       Only SPs that have published Initial Settlement values.
    state            : dict   carry-over state from the previous hourly call
                       (empty dict on the first call of a day).
    corrected_df     : copy of forecast_df with bias correction applied to
                       unsettled SPs.  Schema identical to input.
    new_state        : dict   updated state to persist for the next call.

State is persisted between GitHub Actions calls (which are cold) via
load_state() / save_state() operating on KALMAN_STATE_PATH.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Protocol, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CORRECTION_COLS: Tuple[str, ...] = (
    "ssp_predicted", "ssp_q10", "ssp_q50", "ssp_q90"
)

KALMAN_STATE_PATH     = Path(__file__).resolve().parents[2] / "model_assets" / "kalman_state.json"
CORRECTOR_CONFIG_PATH = Path(__file__).resolve().parents[2] / "model_assets" / "corrector_config.json"
SP_BIAS_PATH          = Path(__file__).resolve().parents[2] / "model_assets" / "sp_bias_profile_v1.json"


# ── Interface ─────────────────────────────────────────────────────────────────

class Corrector(Protocol):
    def __call__(
        self,
        forecast_df: pd.DataFrame,
        settled_actuals: Dict[int, float],
        state: dict,
    ) -> Tuple[pd.DataFrame, dict]: ...


# ── AlphaCorrector ────────────────────────────────────────────────────────────

class AlphaCorrector:
    """
    Flat bias-shift corrector (α=0.4, dead-band £0.50).

    Computes the mean forecast error (actual − base-forecast q50) over all
    settled SPs, then shifts every *unsettled* SP by α × mean_error.  No
    correction is applied when |shift| ≤ dead_band or no SPs are settled.

    Stateless: ignores incoming state and always returns {}.
    This is the direct extraction of the _SHAPE_ALPHA = 0.4 block that
    previously lived inline in run_forecast().
    """

    def __init__(self, alpha: float = 0.4, dead_band: float = 0.5) -> None:
        self.alpha = alpha
        self.dead_band = dead_band

    def __call__(
        self,
        forecast_df: pd.DataFrame,
        settled_actuals: Dict[int, float],
        state: dict,
    ) -> Tuple[pd.DataFrame, dict]:
        if not settled_actuals:
            return forecast_df.copy(), {}

        df = forecast_df.copy()
        unsettled_mask = ~df["settlement_period"].isin(settled_actuals)

        settled_rows = df[df["settlement_period"].isin(settled_actuals)]
        actual_vals = [settled_actuals[sp] for sp in settled_rows["settlement_period"]]
        fc_vals = settled_rows["ssp_q50"].tolist()

        mean_err = float(np.mean([a - f for a, f in zip(actual_vals, fc_vals)]))
        correction = self.alpha * mean_err

        if unsettled_mask.any() and abs(correction) > self.dead_band:
            for col in CORRECTION_COLS:
                df.loc[unsettled_mask, col] = (
                    df.loc[unsettled_mask, col] + correction
                ).round(2)
            log.info(
                "Shape correction: mean_err=£%.1f  applied=£%.1f  (%d unsettled SPs)",
                mean_err, correction, int(unsettled_mask.sum()),
            )

        return df, {}


# ── KalmanCorrector ───────────────────────────────────────────────────────────

class KalmanCorrector:
    """
    Scalar (level-only) Kalman-filter bias corrector.

    Model
    -----
    State  : x̂  — slowly-varying mean forecast bias (£/MWh)
             P   — posterior variance of x̂
    Process: x_{t+1} = x_t + w_t,   w_t ~ N(0, Q)      [random walk]
    Obs    : z_t = mean(actual − q50) over n_t settled SPs
             v_t ~ N(0, R_t),  R_t = σ²_SP / n_t

    Horizon decay applied to unsettled SPs:
        correction(h) = x̂ · γ^h        for h = 0, 1, …, (48 − n_t − 1)
    PI widening:
        ±z_alpha · sqrt(P) · γ^h  added to Q10 / Q90

    State is reset to (x̂=0, P=P₀) at the start of each new forecast date.

    v2 seam (§3.2, not implemented): promote x̂ → [b, ḃ]ᵀ with
        A = [[1, 1], [0, 1]] and H = [1, 0].  The _predict / _update
        methods below would change to matrix operations; __call__ is unchanged.
    """

    def __init__(
        self,
        Q: float = 21.0,
        sigma_sp: float = 35.0,
        gamma: float = 0.966,
        z_alpha: float = 1.28,
        z_guardrail: float = 500.0,
        sp_bias_path=None,
        sp_bias=None,
    ) -> None:
        self.Q = Q
        self.sigma_sp = sigma_sp
        self.gamma = gamma
        self.z_alpha = z_alpha
        # Skip Kalman update if |z_t| exceeds this (data-quality / unit-error guard).
        # Legitimate Code-P spike days have mean residual ≤ ~£250; £500 fires only on
        # corrupted data (wrong units, null sentinels, stale-day mismatch).
        self.z_guardrail = z_guardrail
        # Phase 5a: per-SP bias offset (empty = disabled)
        self.sp_bias: Dict[int, float] = {}
        if sp_bias is not None:
            self.sp_bias = {int(k): float(v) for k, v in sp_bias.items()}
        elif sp_bias_path is not None:
            try:
                with open(sp_bias_path) as f:
                    data = json.load(f)
                self.sp_bias = {int(k): float(v) for k, v in data["bias_by_sp"].items()}
                log.info("Loaded SP bias profile: %d SPs from %s", len(self.sp_bias), sp_bias_path)
            except (FileNotFoundError, json.JSONDecodeError, OSError, KeyError):
                log.warning("SP bias profile not loaded from %s — running without SP bias", sp_bias_path)

    @property
    def P0(self) -> float:
        """Uninformative prior: P₀ = R₀ = σ²_SP."""
        return self.sigma_sp ** 2

    # ── Phase 5a: per-SP bias ─────────────────────────────────────────────────

    def _apply_sp_bias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shift all correction columns by the per-SP offset (vectorised).
        Returns a copy; no-op if sp_bias is empty."""
        if not self.sp_bias:
            return df.copy()
        out = df.copy()
        bias_arr = out["settlement_period"].map(self.sp_bias).fillna(0.0)
        for col in CORRECTION_COLS:
            if col in out.columns:
                out[col] = (out[col] + bias_arr).round(2)
        return out

    # ── v2 seam: isolate predict / update steps ───────────────────────────────

    def _predict(self, x_hat: float, P: float) -> Tuple[float, float]:
        """Random-walk predict step.
        v2 change: x_hat_minus = A @ x_hat,  P_minus = A @ P @ A.T + Q_matrix."""
        return x_hat, P + self.Q

    def _update(
        self, x_hat_minus: float, P_minus: float, z: float, R: float
    ) -> Tuple[float, float, float]:
        """Scalar measurement update.
        v2 change: promote to S = H@P_minus@H.T + R, K = P_minus@H.T / S."""
        K = P_minus / (P_minus + R)
        x_hat = x_hat_minus + K * (z - x_hat_minus)
        P = (1.0 - K) * P_minus
        return x_hat, P, K

    # ── Main interface ────────────────────────────────────────────────────────

    def __call__(
        self,
        forecast_df: pd.DataFrame,
        settled_actuals: Dict[int, float],
        state: dict,
    ) -> Tuple[pd.DataFrame, dict]:
        # ── Daily reset ───────────────────────────────────────────────────────
        try:
            fc_date = str(forecast_df["settlement_date"].iloc[0])
        except (KeyError, IndexError):
            fc_date = state.get("forecast_date", "")

        if state.get("forecast_date") != fc_date:
            log.info(
                "Kalman reset: new date %s (was %s)",
                fc_date, state.get("forecast_date", "none"),
            )
            state = {
                "x_hat": 0.0,
                "P": self.P0,
                "forecast_date": fc_date,
                "last_update": "",
            }

        x_hat = float(state.get("x_hat", 0.0))
        P     = float(state.get("P", self.P0))

        # ── Phase 5a: apply per-SP bias before Kalman step ───────────────────
        df = self._apply_sp_bias(forecast_df)

        if not settled_actuals:
            return df, state

        # ── Observation z_t = mean(actual − SP-adjusted q50) ─────────────────
        settled_rows = df[df["settlement_period"].isin(settled_actuals)]
        actual_vals = [settled_actuals[sp] for sp in settled_rows["settlement_period"]]
        fc_vals     = settled_rows["ssp_q50"].tolist()
        n           = len(actual_vals)
        z           = float(np.mean([a - f for a, f in zip(actual_vals, fc_vals)]))
        R           = (self.sigma_sp ** 2) / n

        # ── Idempotency: skip if no new SPs since last call ───────────────────
        if n == state.get("last_n_settled"):
            log.debug("Kalman idempotent skip: n_settled=%d unchanged", n)
            return self._apply_correction(df, settled_actuals, x_hat, P), state

        # ── Guardrail: skip on anomalous z_t (data-quality check) ────────────
        if abs(z) > self.z_guardrail:
            log.warning(
                "Kalman guardrail: |z|=£%.1f > £%.1f (n=%d) — skipping update; "
                "prior x̂=£%.2f applied",
                abs(z), self.z_guardrail, n, x_hat,
            )
            guard_state = {**state, "last_update": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M")}
            return self._apply_correction(df, settled_actuals, x_hat, P), guard_state

        # ── Kalman recursion ──────────────────────────────────────────────────
        x_hat_minus, P_minus   = self._predict(x_hat, P)
        x_hat_new, P_new, K    = self._update(x_hat_minus, P_minus, z, R)

        log.info(
            "Kalman update: n_settled=%d  z=£%.2f  K=%.3f  "
            "x̂=£%.2f→£%.2f  P=%.1f→%.1f",
            n, z, K, x_hat, x_hat_new, P, P_new,
        )

        # ── Persist state ─────────────────────────────────────────────────────
        new_state = {
            "x_hat":          round(x_hat_new, 6),
            "P":              round(P_new, 6),
            "forecast_date":  fc_date,
            "last_update":    pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M"),
            "last_n_settled": n,
            "last_z":         round(z, 4),
        }
        return self._apply_correction(df, settled_actuals, x_hat_new, P_new), new_state

    # ── Shared correction application ─────────────────────────────────────────

    def _apply_correction(
        self,
        df: pd.DataFrame,
        settled_actuals: Dict[int, float],
        x_hat: float,
        P: float,
    ) -> pd.DataFrame:
        """Apply decayed bias correction + PI widening to all unsettled SPs."""
        unsettled_mask = ~df["settlement_period"].isin(settled_actuals)
        if not unsettled_mask.any():
            return df

        unsettled_df = df[unsettled_mask].sort_values("settlement_period")
        n_unsettled  = len(unsettled_df)
        h            = np.arange(n_unsettled, dtype=float)
        decay        = self.gamma ** h
        correction   = x_hat * decay
        uncertainty  = self.z_alpha * np.sqrt(P) * decay

        idx = unsettled_df.index
        df.loc[idx, "ssp_q50"]       = (unsettled_df["ssp_q50"]       + correction).round(2).values
        df.loc[idx, "ssp_predicted"] = (unsettled_df["ssp_predicted"] + correction).round(2).values
        df.loc[idx, "ssp_q90"]       = (unsettled_df["ssp_q90"]       + correction + uncertainty).round(2).values
        df.loc[idx, "ssp_q10"]       = (unsettled_df["ssp_q10"]       + correction - uncertainty).round(2).values

        # Enforce Q10 ≤ Q50 ≤ Q90
        df.loc[idx, "ssp_q10"] = df.loc[idx, ["ssp_q10", "ssp_q50"]].min(axis=1).round(2)
        df.loc[idx, "ssp_q90"] = df.loc[idx, ["ssp_q90", "ssp_q50"]].max(axis=1).round(2)
        return df


# ── Phase 4: PI calibration ───────────────────────────────────────────────────

def apply_pi_calibration(df: pd.DataFrame, pi_path: "Path | str") -> pd.DataFrame:
    """Widen Q10/Q90 bands by per-SP conformal calibration offset δ(sp).

    Implements the split-conformal symmetric widening from Phase 4:
        q10_new(sp) = q10(sp) − δ(sp)
        q90_new(sp) = q90(sp) + δ(sp)

    δ(sp) is the 80th-percentile conformity score for settlement period sp,
    learned offline from walk_forward_predictions.csv (all 4 folds).

    Applied to ALL 48 SPs before the Kalman corrector.  Settled SPs are later
    overwritten with actuals by the intraday splice, so the widening only
    persists for unsettled SPs (which is the intended target).

    δ is clipped at 0 — we only ever widen, never narrow.
    """
    with open(pi_path) as f:
        pi = json.load(f)
    delta_map = {int(k): float(v) for k, v in pi["delta_by_sp"].items()}
    out = df.copy()
    delta_arr = out["settlement_period"].map(delta_map).fillna(0.0).clip(lower=0.0)
    out["ssp_q10"] = (out["ssp_q10"] - delta_arr).round(2)
    out["ssp_q90"] = (out["ssp_q90"] + delta_arr).round(2)
    return out


# ── Phase 6a: spike-gated asymmetric PI widening ─────────────────────────────

def apply_spike_widening(
    df: pd.DataFrame,
    p_spike: float,
    tau: float,
    delta_hi_path: "Path | str",
) -> pd.DataFrame:
    """Widen Q90 asymmetrically for high-risk afternoon SPs when P(spike) > τ.

    Applied AFTER PI calibration and BEFORE Kalman.  Q10 and Q50 are unchanged —
    only Q90 is widened for the 7 afternoon-block SPs (33–40 ∩ propensity ≥ 2%).

    Returns df unchanged when p_spike ≤ tau or delta_hi_path does not exist.
    """
    if p_spike <= tau:
        return df
    delta_hi_path = Path(delta_hi_path)
    if not delta_hi_path.exists():
        log.warning("apply_spike_widening: artifact not found at %s", delta_hi_path)
        return df
    with open(delta_hi_path) as f:
        artifact = json.load(f)
    delta_map = {int(k): float(v) for k, v in artifact["delta_hi_effective_by_sp"].items()}
    out = df.copy()
    delta_arr = out["settlement_period"].map(delta_map).fillna(0.0)
    out["ssp_q90"] = (out["ssp_q90"] + delta_arr).round(2)
    # Enforce Q50 ≤ Q90 (Q50 unchanged so this is trivially satisfied after widening,
    # but guard against any edge case where the artifact has delta < 0)
    out["ssp_q90"] = out[["ssp_q90", "ssp_q50"]].max(axis=1).round(2)
    log.info(
        "Spike widening applied: P(spike)=%.3f > τ=%.2f; δ_hi=£%.2f on %d SPs",
        p_spike, tau,
        artifact.get("delta_hi_block", 0.0),
        int((delta_arr > 0).sum()),
    )
    return out


# ── State persistence ─────────────────────────────────────────────────────────

def load_state(path: Path = KALMAN_STATE_PATH) -> dict:
    """
    Load corrector state from JSON.  Returns {} if the file is missing,
    empty, or contains malformed JSON — safe to call on a cold runner.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(state: dict, path: Path = KALMAN_STATE_PATH) -> None:
    """Persist corrector state to JSON.  Creates parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_corrector(config_path: Path = CORRECTOR_CONFIG_PATH) -> "Corrector":
    """Instantiate the active corrector from corrector_config.json.

    Config keys (all optional, defaults shown):
        corrector : "kalman" | "alpha"   (default "kalman")
        Q         : float                (default 21.0)
        sigma_sp  : float                (default 35.0)
        gamma     : float                (default 0.966)
        z_alpha   : float                (default 1.28)
        alpha     : float                (AlphaCorrector only, default 0.4)
        dead_band : float                (AlphaCorrector only, default 0.5)
    """
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        cfg = {}

    corrector_type = cfg.get("corrector", "kalman")

    if corrector_type == "alpha":
        return AlphaCorrector(
            alpha=cfg.get("alpha", 0.4),
            dead_band=cfg.get("dead_band", 0.5),
        )

    # Phase 5a: load SP bias profile unless explicitly disabled in config
    use_sp_bias = cfg.get("sp_bias", True)
    sp_bias_path = SP_BIAS_PATH if (use_sp_bias and SP_BIAS_PATH.exists()) else None

    return KalmanCorrector(
        Q=cfg.get("Q", 21.0),
        sigma_sp=cfg.get("sigma_sp", 35.0),
        gamma=cfg.get("gamma", 0.966),
        z_alpha=cfg.get("z_alpha", 1.28),
        z_guardrail=cfg.get("z_guardrail", 500.0),
        sp_bias_path=sp_bias_path,
    )

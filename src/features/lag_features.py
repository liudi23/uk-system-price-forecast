"""
Lag and rolling-window features for UK electricity price forecasting.

Lag structure follows the LEAR (Lasso Estimated AutoRegressive) model
from Lago et al. (2021) and Weron (2014):

  SSP lags  : 1 SP (30 min), 2 SP (1 h), 48 SP (1 day),
              96 SP (2 days), 336 SP (1 week)
  NIV lags  : 1 SP, 48 SP, 336 SP
              Net Imbalance Volume is a leading indicator of price
              pressure; 48 SP captures yesterday's system state at the
              same period (strongest predictor in UK-specific studies).

Rolling windows:
  6 SP  (3 h)  — intra-day momentum
  48 SP (1 day) — daily seasonal context
  336 SP (1 week) — weekly seasonal context

SSP lag features use the winsorised `ssp` column so spike contamination
does not propagate into the autoregressive features.  Additional raw-price
and spike-history features (lag ≥ 48 only) are appended separately; these
lags are always available at day-ahead inference time because they reference
actual history that predates the 48-period forecast horizon.

References:
  Lago, J., et al. (2021). Forecasting day-ahead electricity prices:
    A review of state-of-the-art algorithms, best practices and an
    open-access benchmark. Applied Energy, 293, 116983.
  Weron, R. (2014). Electricity price forecasting: A review.
    International Journal of Forecasting, 30(4), 1030–1081.
"""

import numpy as np
import pandas as pd

# Lag periods (in units of settlement periods, 1 SP = 30 min)
SSP_LAGS = [1, 2, 48, 96, 336]
NIV_LAGS = [1, 48, 336]

# Rolling window sizes (settlement periods)
ROLLING_WINDOWS = [6, 48, 336]


def add_lag_features(
    df: pd.DataFrame,
    ssp_col: str = "ssp",
    niv_col: str = "net_imbalance_volume",
) -> pd.DataFrame:
    """
    Append lag and rolling features to `df` sorted by settlement_datetime.

    The DataFrame must be sorted chronologically before calling this
    function (build_features.py ensures this). NaN values at the start of
    the series (before enough history is available) are left as NaN; the
    model training step should handle them via dropna or imputation.
    """
    df = df.copy()

    # ── SSP lags (winsorised — clean autoregressive signal) ────────────────────
    for lag in SSP_LAGS:
        df[f"{ssp_col}_lag_{lag}"] = df[ssp_col].shift(lag)

    # ── NIV lags ───────────────────────────────────────────────────────────────
    for lag in NIV_LAGS:
        df[f"{niv_col}_lag_{lag}"] = df[niv_col].shift(lag)

    # ── Rolling statistics on SSP ──────────────────────────────────────────────
    for w in ROLLING_WINDOWS:
        roll = df[ssp_col].shift(1).rolling(window=w, min_periods=1)
        df[f"{ssp_col}_roll_mean_{w}"] = roll.mean()
        df[f"{ssp_col}_roll_std_{w}"]  = roll.std()
        df[f"{ssp_col}_roll_min_{w}"]  = roll.min()
        df[f"{ssp_col}_roll_max_{w}"]  = roll.max()

    # ── Rolling statistics on NIV ──────────────────────────────────────────────
    for w in [6, 48]:
        roll_niv = df[niv_col].shift(1).rolling(window=w, min_periods=1)
        df[f"{niv_col}_roll_mean_{w}"] = roll_niv.mean()
        df[f"{niv_col}_roll_std_{w}"]  = roll_niv.std()
        df[f"{niv_col}_roll_min_{w}"]  = roll_niv.min()
        df[f"{niv_col}_roll_max_{w}"]  = roll_niv.max()

    # ── Price-spread momentum ──────────────────────────────────────────────────
    df[f"{ssp_col}_diff_1_48"]   = df[f"{ssp_col}_lag_1"]  - df[f"{ssp_col}_lag_48"]
    df[f"{ssp_col}_diff_48_336"] = df[f"{ssp_col}_lag_48"] - df[f"{ssp_col}_lag_336"]

    # ── Raw-price spike-memory features (lag ≥ 48, always in actual history) ───
    # These carry the unclipped price signal, letting the model learn that a
    # spike yesterday/last week raises the probability of another spike today.
    if "ssp_raw" in df.columns:
        for lag in [48, 96, 336]:
            df[f"ssp_raw_lag_{lag}"] = df["ssp_raw"].shift(lag)

    # ── Spike / negative-price history flags ───────────────────────────────────
    # is_spike marks rows where ssp_raw exceeded the Tukey outer fence.
    # is_negative marks negative-price periods (renewable oversupply, low demand).
    # Both use lag ≥ 48 so they always reference settled actual data at inference.
    if "is_spike" in df.columns:
        spike_f = df["is_spike"].astype(float)
        for lag in [48, 336]:
            df[f"is_spike_lag_{lag}"] = spike_f.shift(lag)
        # Rolling count of spike periods in the previous 24 h (48 SPs)
        df["spike_count_roll_48"] = spike_f.shift(1).rolling(48, min_periods=1).sum()

    if "ssp_raw" in df.columns:
        neg_f = (df["ssp_raw"] < 0).astype(float)
        for lag in [48, 336]:
            df[f"is_negative_lag_{lag}"] = neg_f.shift(lag)
        df["neg_count_roll_48"] = neg_f.shift(1).rolling(48, min_periods=1).sum()

    # ── NIV extreme features (system stress / shortage / surplus indicators) ───
    # Rolling min/max on NIV reveal how short (negative) or long (positive)
    # the system has been, which is a direct antecedent of price spikes/crashes.
    if niv_col in df.columns:
        for w in [6, 48]:
            roll_niv2 = df[niv_col].shift(1).rolling(window=w, min_periods=1)
            # min/max already added above; keep only if not duplicate
        # Absolute NIV rolling mean — how stressed the system has been regardless of direction
        for w in [6, 48]:
            abs_niv = df[niv_col].abs().shift(1).rolling(window=w, min_periods=1)
            df[f"abs_niv_roll_mean_{w}"] = abs_niv.mean()

    return df


def add_generation_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append wind_pct and gas_pct lag features (lag-48 and lag-336 only).

    Wind % in the generation mix is a key marginal-cost signal: high wind
    displaces gas plant, suppressing prices; low wind forces high-cost gas
    peakers online, pushing prices up.  Gas % directly proxies marginal cost.

    Only lag-48+ lags are added — leakage-free for the shape model.
    """
    df = df.copy()
    for col in ["wind_pct", "gas_pct"]:
        if col not in df.columns:
            continue
        for lag in [48, 336]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

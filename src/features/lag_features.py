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

All lags and rolling features are computed on `ssp` (the denoised price)
so that spike contamination does not propagate into features.  The raw
price `ssp_raw` remains available as the modelling target.

References:
  Lago, J., et al. (2021). Forecasting day-ahead electricity prices:
    A review of state-of-the-art algorithms, best practices and an
    open-access benchmark. Applied Energy, 293, 116983.
  Weron, R. (2014). Electricity price forecasting: A review.
    International Journal of Forecasting, 30(4), 1030–1081.
"""

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

    # ── SSP lags ───────────────────────────────────────────────────────────────
    for lag in SSP_LAGS:
        df[f"{ssp_col}_lag_{lag}"] = df[ssp_col].shift(lag)

    # ── NIV lags ───────────────────────────────────────────────────────────────
    for lag in NIV_LAGS:
        df[f"{niv_col}_lag_{lag}"] = df[niv_col].shift(lag)

    # ── Rolling statistics on SSP ──────────────────────────────────────────────
    # min_periods=1 avoids NaN for the very first rows but note that early
    # rolling values (window not yet full) are less reliable.
    for w in ROLLING_WINDOWS:
        roll = df[ssp_col].shift(1).rolling(window=w, min_periods=1)
        df[f"{ssp_col}_roll_mean_{w}"] = roll.mean()
        df[f"{ssp_col}_roll_std_{w}"] = roll.std()
        df[f"{ssp_col}_roll_min_{w}"] = roll.min()
        df[f"{ssp_col}_roll_max_{w}"] = roll.max()

    # ── Rolling statistics on NIV ──────────────────────────────────────────────
    for w in [6, 48]:
        roll_niv = df[niv_col].shift(1).rolling(window=w, min_periods=1)
        df[f"{niv_col}_roll_mean_{w}"] = roll_niv.mean()
        df[f"{niv_col}_roll_std_{w}"] = roll_niv.std()

    # ── Price-spread momentum ──────────────────────────────────────────────────
    # Difference between current lag-1 and lag-48 captures within-day trend
    df[f"{ssp_col}_diff_1_48"] = df[f"{ssp_col}_lag_1"] - df[f"{ssp_col}_lag_48"]
    # Difference between lag-48 and lag-336 captures day-on-day change vs last week
    df[f"{ssp_col}_diff_48_336"] = df[f"{ssp_col}_lag_48"] - df[f"{ssp_col}_lag_336"]

    return df

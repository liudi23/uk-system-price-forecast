"""Set-based Kalman idempotency for out-of-order / holey Elexon feeds.

The intraday feed publishes settlement periods sparsely and out of order, so the
*count* of settled SPs can stay equal while the *set* changes (e.g. SP30 drops
out, SP34 appears). Count-based idempotency froze the bias estimate in that case;
the guard now keys off the exact set of settled SPs.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
from correctors import KalmanCorrector


def _make_forecast(n: int = 48, base_q50: float = 80.0, date: str = "2026-06-24") -> pd.DataFrame:
    return pd.DataFrame([
        {"settlement_date": date, "settlement_period": sp,
         "ssp_predicted": base_q50, "ssp_q10": base_q50 - 10.0,
         "ssp_q50": base_q50, "ssp_q90": base_q50 + 10.0, "is_actual": False}
        for sp in range(1, n + 1)
    ])


def _fresh(corr, date="2026-06-24"):
    return {"x_hat": 0.0, "P": corr.P0, "forecast_date": date, "last_update": ""}


def test_same_count_different_set_still_updates():
    """n unchanged but the SP set differs → Kalman must update, not skip."""
    corr = KalmanCorrector()
    fc = _make_forecast()

    _, s1 = corr(fc, {1: 90.0, 2: 90.0, 3: 90.0}, _fresh(corr))   # z=+10
    assert s1["last_settled_sps"] == [1, 2, 3]
    x1 = s1["x_hat"]

    # Same count (3) but SP3 replaced by SP4 with a different residual → z changes.
    _, s2 = corr(fc, {1: 90.0, 2: 90.0, 4: 70.0}, s1)             # z=+3.33
    assert s2["last_settled_sps"] == [1, 2, 4]                    # processed the new set
    assert s2["x_hat"] != x1                                       # bias actually moved
    assert s2["last_n_settled"] == 3


def test_identical_set_is_idempotent():
    """Exact same SP set on a re-fire → no state change (idempotent skip)."""
    corr = KalmanCorrector()
    fc = _make_forecast()
    _, s1 = corr(fc, {1: 90.0, 2: 90.0, 4: 70.0}, _fresh(corr))
    _, s2 = corr(fc, {1: 90.0, 2: 90.0, 4: 70.0}, s1)            # identical set
    assert s2 == s1                                               # unchanged
    assert s2["x_hat"] == s1["x_hat"]


def test_holey_set_processes_all_settled_sps():
    """A set with interior gaps (1,2,3,5 — missing 4) is processed in full."""
    corr = KalmanCorrector()
    fc = _make_forecast()
    _, s = corr(fc, {1: 95.0, 2: 95.0, 3: 95.0, 5: 95.0}, _fresh(corr))
    assert s["last_settled_sps"] == [1, 2, 3, 5]
    assert s["last_n_settled"] == 4
    assert s["x_hat"] != 0.0                                      # bias update applied

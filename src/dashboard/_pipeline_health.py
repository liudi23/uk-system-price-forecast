"""
Pipeline health signal computation — no Streamlit dependency.

Extracted here so streamlit_app.py can wrap pipeline_status() with
@st.cache_data(ttl=300) while tests import _compute_pipeline_status directly.
"""
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

UK_TZ = ZoneInfo("Europe/London")   # settlement periods are UK-local (BST/GMT)

_STALE_THRESHOLD_MIN = 360   # 6h — above P95 of same-day inter-run gaps (304 min, n=17)
                              # and above the observed same-day max (350 min, n=25 commits).
                              # 180 min (median) would false-alarm on every second missed slot.
_DELAY_THRESHOLD_MIN = 150   # data-recency: how far the LATEST received SP may lag wall-clock
                              # before flagging "delayed". Normal-day newest-SP lag maxes at
                              # 91 min (Jun 22-23 git history; == frontier lag when gap-free);
                              # 150 leaves ~60 min margin. Interior gaps are NOT a delay — a
                              # current latest SP keeps the feed "ok" with a sidebar gap note.
_ACTIVE_WINDOW_START = 7     # UTC: Kalman updates expected from 07:00
_ACTIVE_WINDOW_END   = 22    # UTC: Kalman updates expected until 22:00
_RULE_B_TOL          = 3     # tolerance for the catch-up check: n_actual vs kalman_n_settled
                             # (counts, not max-SP vs count — holey feeds have max_sp >> count)


def _compute_pipeline_status(
    fc_path: Path,
    kalman_path: Path,
    now_utc: Optional[datetime] = None,
) -> dict:
    """
    Single source of truth for all pipeline freshness and health signals.

    Parameters
    ----------
    fc_path     : path to next_day_forecast_phase3.csv
    kalman_path : path to kalman_state.json
    now_utc     : current time override (UTC-aware datetime; use in tests)

    Returns
    -------
    dict with keys:
      fc_date           : str|None  settlement_date in forecast CSV
      kalman_last_update: str|None  last_update in kalman_state.json
      kalman_fc_date    : str|None  forecast_date in kalman_state.json
      kalman_n_settled  : int       last_n_settled in kalman_state.json
      n_actual_sps      : int       count of is_actual==True rows in forecast CSV
      last_actual_sp    : int       max settlement_period where is_actual==True (0 if none)
      health            : str       "ok"|"stale"|"daily_missed"|"unknown"
      health_msg        : str|None  human-readable sidebar/banner text
      consistent        : bool      False when Rule A or Rule B is violated
      inconsistency_msg : str|None  warning text when not consistent
    """
    _now   = now_utc or datetime.now(timezone.utc)
    _today = str(_now.date())
    _hour  = _now.hour
    _in_window = _ACTIVE_WINDOW_START <= _hour < _ACTIVE_WINDOW_END

    # ── 1. Read forecast CSV (minimal columns only) ────────────────────────────
    _fc_date: Optional[str] = None
    _n_actual = 0
    _last_actual_sp = 0
    _complete_sp = 0   # contiguous frontier: largest F with SPs 1..F all settled
    if fc_path.exists():
        try:
            _wanted = {"settlement_date", "settlement_period", "is_actual"}
            _fc_df = pd.read_csv(fc_path, usecols=lambda c: c in _wanted)
            if not _fc_df.empty:
                _fc_date = str(_fc_df["settlement_date"].iloc[0])
                if "is_actual" in _fc_df.columns:
                    # Robust to both bool and "True"/"False" string CSV round-trip
                    _mask = _fc_df["is_actual"].map(
                        lambda x: str(x).strip().lower() in {"true", "1"}
                    )
                    _n_actual = int(_mask.sum())
                    if _n_actual > 0:
                        _settled = set(int(sp) for sp in _fc_df.loc[_mask, "settlement_period"])
                        _last_actual_sp = max(_settled)
                        # Contiguous complete-data frontier (stops at the first hole), so a
                        # sparse feed with high-SP outliers doesn't mask a mid-day gap.
                        while (_complete_sp + 1) in _settled:
                            _complete_sp += 1
        except Exception:
            pass

    # ── 2. Read Kalman state ───────────────────────────────────────────────────
    _kalman_last_update: Optional[str] = None
    _kalman_fc_date:     Optional[str] = None
    _kalman_n_settled    = 0
    if kalman_path.exists():
        try:
            with open(kalman_path) as _f:
                _s = json.load(_f)
            _kalman_last_update = _s.get("last_update") or None
            _kalman_fc_date     = _s.get("forecast_date") or None
            _kalman_n_settled   = int(_s.get("last_n_settled", 0))
        except Exception:
            pass

    # ── 3. Derive health state ─────────────────────────────────────────────────
    _health = "unknown"
    _health_msg: Optional[str] = None

    if _fc_date is None and _kalman_last_update is None:
        _health = "unknown"

    elif _fc_date and _fc_date < _today and _hour >= 14:
        # Daily pipeline did not run today and it's past 14:00 UTC
        _health = "daily_missed"
        _health_msg = (
            f"🔴 Daily pipeline has not refreshed today's forecast "
            f"(last forecast_date: **{_fc_date}**). "
            f"Expected by ~13:30 UTC — check GitHub Actions."
        )

    elif _kalman_last_update:
        try:
            _last_dt = datetime.fromisoformat(_kalman_last_update).replace(tzinfo=timezone.utc)
            _age_min = (_now - _last_dt).total_seconds() / 60
            if _kalman_fc_date == _today and _in_window and _age_min > _STALE_THRESHOLD_MIN:
                # Kalman has today's fc_date but hasn't updated in > 3h during trading hours
                _health = "stale"
                _health_msg = (
                    f"🔴 Intraday pipeline stale — last Kalman update "
                    f"**{_last_dt.strftime('%H:%M UTC')}** ({_age_min:.0f} min ago). "
                    f"Check GitHub Actions → Intraday Forecast Update."
                )
            else:
                _health = "ok"
                _health_msg = (
                    f"Last intraday: **{_kalman_last_update} UTC** · {_kalman_n_settled} SPs settled"
                )
        except Exception:
            _health = "unknown"

    else:
        # Kalman has no last_update but fc_date is present — daily ran, intraday hasn't yet
        _health = "ok"

    # ── 3b. Data-recency override ──────────────────────────────────────────────
    # "Delayed" means we are not receiving recent settled data — key off the LATEST
    # received SP, not the contiguous frontier. A single early hole (e.g. SP28 never
    # published) leaves the frontier stuck for hours but does NOT mean the feed is
    # stale: later SPs keep arriving and the dashboard is current. Interior gaps are
    # surfaced separately (sidebar "N missing"), not as a delay alarm. A genuine
    # stall (or silent fetch failure) shows up as an old latest-received SP.
    _latest_actual_lag_min: Optional[float] = None
    if _health == "ok" and _fc_date == _today and _in_window and _last_actual_sp > 0:
        try:
            _latest_end = (
                datetime.combine(date.fromisoformat(_fc_date), time.min, tzinfo=UK_TZ)
                + timedelta(minutes=_last_actual_sp * 30)
            )
            _latest_actual_lag_min = (_now - _latest_end).total_seconds() / 60
            if _latest_actual_lag_min > _DELAY_THRESHOLD_MIN:
                _health = "delayed"
                _gap_note = (
                    f" ({_last_actual_sp - _n_actual} earlier period(s) still missing)"
                    if _n_actual < _last_actual_sp else ""
                )
                _health_msg = (
                    f"🟠 Settled data delayed — latest settled SP{_last_actual_sp} "
                    f"(**{_latest_end.strftime('%H:%M')} {_latest_end.tzname()}**), "
                    f"{_latest_actual_lag_min / 60:.1f}h behind.{_gap_note} Upstream Elexon "
                    f"publication gap or stalled fetch; the dashboard catches up automatically "
                    f"when the feed resumes."
                )
        except Exception:
            pass

    # ── 4. Self-consistency checks (skip when pipeline is known-missed or unknown) ──
    _consistent = True
    _inconsistency_msg: Optional[str] = None

    if _health not in ("daily_missed", "unknown") and _kalman_fc_date == _today:
        if _kalman_n_settled > 0 and _n_actual == 0:
            # State recorded settled periods but the forecast CSV shows none — the
            # intraday commit pushed the Kalman state but not the forecast.
            _consistent = False
            _inconsistency_msg = (
                f"⚠️ Intraday data out of sync — the correction state recorded "
                f"{_kalman_n_settled} settled periods but none are in the published "
                f"forecast yet (a commit may have failed)."
            )
        elif _n_actual > _kalman_n_settled + _RULE_B_TOL:
            # Count-vs-count: the forecast CSV has more settled periods than the bias
            # estimate has processed. (Compared as counts, not max-SP vs count — a
            # holey/out-of-order feed routinely has max_sp >> count without any lag.)
            _consistent = False
            _inconsistency_msg = (
                f"⚠️ Intraday correction is catching up — {_n_actual - _kalman_n_settled} "
                f"settled periods are not yet in the bias estimate "
                f"({_n_actual} settled, {_kalman_n_settled} corrected)."
            )

    return {
        "fc_date":            _fc_date,
        "kalman_last_update": _kalman_last_update,
        "kalman_fc_date":     _kalman_fc_date,
        "kalman_n_settled":   _kalman_n_settled,
        "n_actual_sps":       _n_actual,
        "last_actual_sp":     _last_actual_sp,
        "complete_sp":        _complete_sp,
        "latest_actual_lag_min": _latest_actual_lag_min,
        "health":             _health,
        "health_msg":         _health_msg,
        "consistent":         _consistent,
        "inconsistency_msg":  _inconsistency_msg,
    }

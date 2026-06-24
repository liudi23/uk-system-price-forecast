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
_DELAY_THRESHOLD_MIN = 150   # data-recency: how far the contiguous complete-data frontier
                              # may lag wall-clock before flagging "delayed". Normal-day max
                              # frontier lag is 91 min (Jun 22-23 git history); 150 leaves
                              # ~60 min margin and would have caught the Jun 24 stall (max 350).
_ACTIVE_WINDOW_START = 7     # UTC: Kalman updates expected from 07:00
_ACTIVE_WINDOW_END   = 22    # UTC: Kalman updates expected until 22:00
_RULE_B_TOL          = 3     # SP tolerance for Rule B: last_actual_sp vs kalman_n_settled


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
    # A run can look fresh (recent kalman_last_update) while serving stale data:
    # the upstream Elexon feed can stall for hours, or a silently-failing fetch
    # (exit-0 hardening) can freeze the spliced actuals. Flag when the contiguous
    # complete-data frontier lags wall-clock by more than the tuned threshold.
    _frontier_lag_min: Optional[float] = None
    if _health == "ok" and _fc_date == _today and _in_window and _complete_sp > 0:
        try:
            _frontier_end = (
                datetime.combine(date.fromisoformat(_fc_date), time.min, tzinfo=UK_TZ)
                + timedelta(minutes=_complete_sp * 30)
            )
            _frontier_lag_min = (_now - _frontier_end).total_seconds() / 60
            if _frontier_lag_min > _DELAY_THRESHOLD_MIN:
                _health = "delayed"
                _health_msg = (
                    f"🟠 Settled data delayed — complete through SP{_complete_sp} "
                    f"(**{_frontier_end.strftime('%H:%M')} {_frontier_end.tzname()}**), "
                    f"{_frontier_lag_min / 60:.1f}h behind. Upstream Elexon publication gap "
                    f"or stalled fetch; the dashboard catches up automatically when the feed resumes."
                )
        except Exception:
            pass

    # ── 4. Self-consistency checks (skip when pipeline is known-missed or unknown) ──
    _consistent = True
    _inconsistency_msg: Optional[str] = None

    if _health not in ("daily_missed", "unknown") and _kalman_fc_date == _today:
        if _kalman_n_settled > 0 and _n_actual == 0:
            # Rule A: Kalman claims N settled SPs but forecast CSV has no is_actual rows
            # → intraday run updated the Kalman state but the commit didn't push the CSV
            _consistent = False
            _inconsistency_msg = (
                f"⚠️ Inconsistency (Rule A): Kalman claims {_kalman_n_settled} SPs settled "
                f"but forecast CSV has no is_actual rows — intraday commit may have failed."
            )
        elif _last_actual_sp > _kalman_n_settled + _RULE_B_TOL:
            # Rule B: forecast CSV has actual SPs well beyond what Kalman has processed
            # → CSV was committed more recently than Kalman state
            _consistent = False
            _inconsistency_msg = (
                f"⚠️ Inconsistency (Rule B): forecast CSV shows actuals through "
                f"SP {_last_actual_sp} but Kalman only processed {_kalman_n_settled} SPs "
                f"— Kalman state may be stale."
            )

    return {
        "fc_date":            _fc_date,
        "kalman_last_update": _kalman_last_update,
        "kalman_fc_date":     _kalman_fc_date,
        "kalman_n_settled":   _kalman_n_settled,
        "n_actual_sps":       _n_actual,
        "last_actual_sp":     _last_actual_sp,
        "complete_sp":        _complete_sp,
        "complete_sp_lag_min": _frontier_lag_min,
        "health":             _health,
        "health_msg":         _health_msg,
        "consistent":         _consistent,
        "inconsistency_msg":  _inconsistency_msg,
    }

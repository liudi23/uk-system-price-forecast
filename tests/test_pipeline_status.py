"""
Unit tests for _compute_pipeline_status health and consistency logic.

Tests import _pipeline_health directly — no Streamlit dependency.
"""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard._pipeline_health import _compute_pipeline_status


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_fc(path: Path, fc_date: str, rows: list[dict]) -> None:
    """Write a minimal next_day_forecast_phase3.csv."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _write_kalman(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state))


def _now(hour: int = 14, date: str = "2026-06-19") -> datetime:
    return datetime.fromisoformat(f"{date}T{hour:02d}:00:00").replace(tzinfo=timezone.utc)


TODAY = "2026-06-19"
YESTERDAY = "2026-06-18"

KALMAN_TODAY = {
    "x_hat": 0.0,
    "P": 1.0,
    "forecast_date": TODAY,
    "last_update": f"{TODAY}T10:00",
    "last_n_settled": 20,
    "last_z": 0.0,
}

KALMAN_YESTERDAY = {**KALMAN_TODAY, "forecast_date": YESTERDAY, "last_update": f"{YESTERDAY}T23:00"}


# ── Health: unknown ───────────────────────────────────────────────────────────

def test_health_unknown_no_files():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ps = _compute_pipeline_status(d / "fc.csv", d / "k.json", now_utc=_now())
    assert ps["health"] == "unknown"
    assert ps["health_msg"] is None
    assert ps["consistent"] is True   # no data → no contradiction


# ── Health: daily_missed ──────────────────────────────────────────────────────

def test_health_daily_missed():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, YESTERDAY, [
            {"settlement_date": YESTERDAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, KALMAN_YESTERDAY)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=15))
    assert ps["health"] == "daily_missed"
    assert "Daily pipeline" in ps["health_msg"]
    assert YESTERDAY in ps["health_msg"]


def test_health_daily_missed_suppressed_before_14h():
    """Before 14:00 UTC the daily_missed alert is suppressed — daily pipeline may still be running."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, YESTERDAY, [
            {"settlement_date": YESTERDAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, KALMAN_YESTERDAY)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=12))
    assert ps["health"] != "daily_missed"


# ── Health: stale ─────────────────────────────────────────────────────────────

def test_health_stale():
    """Kalman fc_date == today, in active window, age > 360 min → stale."""
    stale_kalman = {
        **KALMAN_TODAY,
        "last_update": f"{TODAY}T07:00",   # 7h ago at hour=14
    }
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, stale_kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    assert ps["health"] == "stale"
    assert "stale" in ps["health_msg"].lower()


def test_health_stale_outside_window_is_ok():
    """Outside the 07:00–22:00 window, a long gap is normal — should not be 'stale'."""
    stale_kalman = {**KALMAN_TODAY, "last_update": f"{TODAY}T05:00"}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, stale_kalman)
        # hour=6 is outside the active window
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=6))
    assert ps["health"] == "ok"


def test_health_not_stale_within_threshold():
    """Age ≤ 360 min should not trigger stale."""
    fresh_kalman = {**KALMAN_TODAY, "last_update": f"{TODAY}T08:30"}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, fresh_kalman)
        # 14:00 - 08:30 = 330 min < 360 min threshold
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    assert ps["health"] == "ok"


# ── Health: ok ────────────────────────────────────────────────────────────────

def test_health_ok_basic():
    # Settled contiguously through SP28 (=14:00 BST = 13:00 UTC); at 14:00 UTC the
    # frontier is 60 min behind (< 150) so the data-recency rule leaves it "ok".
    fresh_kalman = {**KALMAN_TODAY, "last_update": f"{TODAY}T13:00", "last_n_settled": 28}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i,
             "is_actual": i <= 28}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, fresh_kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    assert ps["health"] == "ok"
    assert "Last intraday" in ps["health_msg"]
    assert ps["n_actual_sps"] == 28
    assert ps["last_actual_sp"] == 28
    assert ps["kalman_n_settled"] == 28


# ── Consistency: Rule A ───────────────────────────────────────────────────────

def test_inconsistent_rule_a():
    """Rule A: Kalman claims N>0 settled SPs but forecast CSV has no is_actual rows."""
    # last_update at 13:30 → age = 30 min at hour=14, well under stale threshold
    kalman = {**KALMAN_TODAY, "last_n_settled": 15, "last_update": f"{TODAY}T13:30"}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        # All is_actual=False in CSV
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    assert ps["health"] == "ok"
    assert ps["consistent"] is False
    assert "out of sync" in ps["inconsistency_msg"]
    assert "15" in ps["inconsistency_msg"]


def test_consistent_when_kalman_zero_and_no_actuals():
    """Kalman n_settled==0 and n_actual==0 is normal (morning before first SP)."""
    kalman = {**KALMAN_TODAY, "last_n_settled": 0}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=8))
    assert ps["consistent"] is True


# ── Consistency: Rule B ───────────────────────────────────────────────────────

def test_inconsistent_rule_b():
    """Catch-up: CSV settled count > kalman_n_settled + tolerance."""
    kalman = {**KALMAN_TODAY, "last_n_settled": 10}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        # CSV has 20 settled SPs, but the bias estimate processed only 10
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i,
             "is_actual": i <= 20}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    # n_actual=20 (count), kalman_n_settled=10, tol=3 → 20 > 13 → catch-up flagged
    assert ps["consistent"] is False
    assert "catching up" in ps["inconsistency_msg"]


def test_consistent_rule_b_within_tolerance():
    """last_actual_sp == kalman_n_settled + 3 is at the boundary — still consistent."""
    kalman = {**KALMAN_TODAY, "last_n_settled": 17}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i,
             "is_actual": i <= 20}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    # last_actual_sp=20, kalman_n_settled=17 → 20 > 17+3=20 → False → consistent
    assert ps["consistent"] is True


# ── Consistency checks skip during daily_missed / unknown ─────────────────────

def test_no_consistency_check_during_daily_missed():
    """Rule A/B must not fire when health==daily_missed — Kalman fc_date diverges legitimately."""
    kalman = {**KALMAN_YESTERDAY, "last_n_settled": 15}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, YESTERDAY, [
            {"settlement_date": YESTERDAY, "settlement_period": i, "is_actual": False}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=16))
    assert ps["health"] == "daily_missed"
    assert ps["consistent"] is True   # checks disabled when daily_missed


# ── Health: delayed (data-recency / contiguous frontier) ───────────────────────

def _fc_rows(fc_date: str, settled: set):
    """48-SP forecast where SPs in `settled` are is_actual=True."""
    return [
        {"settlement_date": fc_date, "settlement_period": i, "is_actual": (i in settled)}
        for i in range(1, 49)
    ]


def _kalman(fc_date: str, last_update: str, n: int):
    return {"x_hat": 0.0, "P": 1.0, "forecast_date": fc_date,
            "last_update": last_update, "last_n_settled": n, "last_z": 0.0}


def test_health_delayed_when_frontier_lags():
    # Contiguous data only through SP27 (=13:30 BST = 12:30 UTC) while it's 16:00 UTC
    # → 210 min behind > 150 threshold → "delayed", even though Kalman is fresh.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc, k = d / "fc.csv", d / "k.json"
        _write_fc(fc, TODAY, _fc_rows(TODAY, set(range(1, 28))))      # SP1..27
        _write_kalman(k, _kalman(TODAY, f"{TODAY}T15:50", 27))        # fresh run
        ps = _compute_pipeline_status(fc, k, now_utc=_now(hour=16))
    assert ps["health"] == "delayed"
    assert ps["complete_sp"] == 27
    assert ps["complete_sp_lag_min"] > 150
    assert "SP27" in ps["health_msg"] and "BST" in ps["health_msg"]


def test_health_holey_frontier_stops_at_gap_and_not_flagged_at_normal_lag():
    # Sparse feed 1..27 + {29,30}: contiguous frontier is 27 (stops at the SP28 hole),
    # last_actual_sp is 30. At 14:00 UTC the frontier lag is ~90 min (< 150) → "ok".
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc, k = d / "fc.csv", d / "k.json"
        _write_fc(fc, TODAY, _fc_rows(TODAY, set(range(1, 28)) | {29, 30}))
        _write_kalman(k, _kalman(TODAY, f"{TODAY}T13:50", 29))
        ps = _compute_pipeline_status(fc, k, now_utc=_now(hour=14))
    assert ps["complete_sp"] == 27          # frontier stops at the hole
    assert ps["last_actual_sp"] == 30       # max settled still tracked
    assert ps["health"] == "ok"             # 90 min < 150 → no false alarm
    assert ps["complete_sp_lag_min"] < 150


def test_health_ok_when_data_fresh():
    # Contiguous through SP30 (=15:00 BST = 14:00 UTC); now 14:30 UTC → 30 min behind → ok.
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc, k = d / "fc.csv", d / "k.json"
        _write_fc(fc, TODAY, _fc_rows(TODAY, set(range(1, 31))))      # SP1..30
        _write_kalman(k, _kalman(TODAY, f"{TODAY}T14:20", 30))
        ps = _compute_pipeline_status(fc, k, now_utc=_now(hour=14) + __import__("datetime").timedelta(minutes=30))
    assert ps["health"] == "ok"
    assert ps["complete_sp"] == 30


# ── Consistency: holey feed must NOT false-alarm (count-vs-count) ───────────────

def test_holey_feed_not_flagged_as_catchup():
    # Sparse settled set {1..27, 29, 30, 34, 38, 39}: count=32, max_sp=39.
    # Old rule (max_sp 39 > n_settled 32 + 3) false-alarmed; count-vs-count
    # (32 vs 32) does not. frontier SP27 (12:30 UTC) at 13:00 UTC → not delayed.
    settled = set(range(1, 28)) | {29, 30, 34, 38, 39}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc, k = d / "fc.csv", d / "k.json"
        _write_fc(fc, TODAY, _fc_rows(TODAY, settled))
        _write_kalman(k, _kalman(TODAY, f"{TODAY}T12:55", len(settled)))   # 32
        ps = _compute_pipeline_status(fc, k, now_utc=_now(hour=13))
    assert ps["last_actual_sp"] == 39
    assert ps["kalman_n_settled"] == 32
    assert ps["consistent"] is True
    assert ps["inconsistency_msg"] is None


def test_genuine_catchup_is_flagged_plain_language():
    # CSV settled through SP39 (count 39) but the bias estimate processed only 30
    # → genuine lag → flagged, with plain wording (no "Rule B" jargon).
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc, k = d / "fc.csv", d / "k.json"
        _write_fc(fc, TODAY, _fc_rows(TODAY, set(range(1, 40))))           # 1..39
        _write_kalman(k, _kalman(TODAY, f"{TODAY}T18:50", 30))
        ps = _compute_pipeline_status(fc, k, now_utc=_now(hour=19))
    assert ps["consistent"] is False
    assert "catching up" in ps["inconsistency_msg"]
    assert "Rule B" not in ps["inconsistency_msg"]
    assert "39 settled" in ps["inconsistency_msg"] and "30 corrected" in ps["inconsistency_msg"]

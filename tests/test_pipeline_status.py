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
    fresh_kalman = {**KALMAN_TODAY, "last_update": f"{TODAY}T13:00"}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i,
             "is_actual": i <= 20}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, fresh_kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    assert ps["health"] == "ok"
    assert "Last intraday" in ps["health_msg"]
    assert ps["n_actual_sps"] == 20
    assert ps["last_actual_sp"] == 20
    assert ps["kalman_n_settled"] == 20


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
    assert "Rule A" in ps["inconsistency_msg"]
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
    """Rule B: CSV has actual SPs > kalman_n_settled + tolerance."""
    kalman = {**KALMAN_TODAY, "last_n_settled": 10}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        fc_path = d / "fc.csv"
        k_path  = d / "k.json"
        # CSV shows actuals through SP 20, but Kalman only processed 10
        _write_fc(fc_path, TODAY, [
            {"settlement_date": TODAY, "settlement_period": i,
             "is_actual": i <= 20}
            for i in range(1, 49)
        ])
        _write_kalman(k_path, kalman)
        ps = _compute_pipeline_status(fc_path, k_path, now_utc=_now(hour=14))
    # last_actual_sp=20, kalman_n_settled=10, tol=3 → 20 > 13 → Rule B
    assert ps["consistent"] is False
    assert "Rule B" in ps["inconsistency_msg"]


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

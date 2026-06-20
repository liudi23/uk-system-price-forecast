"""
Unit tests for the nowcast SP day-rollover logic.

The nowcast panel computes SP+h labels and 80% bands for the next 3 horizons
after the last settled SP.  When last_settled + h > 48 the target falls on the
next calendar day; the persistence value and horizon bands (h1/h2/h3) remain
valid — the band lookup is horizon-indexed, not absolute-SP-indexed.

Run with:  pytest tests/test_nowcast_rollover.py -v
"""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BANDS_JSON = REPO / "model_assets" / "nowcast_bands.json"


# ── Pure logic extracted from streamlit_app.py (lines 396-423 after fix) ─────

def _sp_to_time_str(sp: int) -> str:
    """Convert SP number (1–48) to HH:MM clock string."""
    mins = (sp - 1) * 30
    return f"{mins // 60:02d}:{mins % 60:02d}"


def compute_nowcast_horizon(last_sp: int, h: int, fc_date_str: str,
                             y0: float, horizons: dict, regime: str = "overall"):
    """
    Compute nowcast output for horizon h after last_sp.

    Returns a dict with:
      header      : str  — display label
      value       : float — persistence point (= y0)
      p10         : int   — 80% band lower
      p90         : int   — 80% band upper
      is_next_day : bool  — True when target crosses midnight
      target_sp   : int   — SP number on its own day (1-48)
      target_date : str   — YYYY-MM-DD of the target SP
    """
    target_sp_raw = last_sp + h
    h_key = f"h{h}"
    bnd = horizons[h_key].get(regime, horizons[h_key]["overall"])
    p10 = round(y0 + bnd["p10"])
    p90 = round(y0 + bnd["p90"])

    if target_sp_raw > 48:
        nd_sp   = target_sp_raw - 48
        nd_date = pd.Timestamp(fc_date_str) + pd.Timedelta(days=1)
        nd_time = _sp_to_time_str(nd_sp)
        date_lbl = nd_date.strftime("%b") + " " + str(nd_date.day)
        header = f"SP+{h} · SP {nd_sp} · {date_lbl} · {nd_time}"
        return dict(header=header, value=y0, p10=p10, p90=p90,
                    is_next_day=True, target_sp=nd_sp,
                    target_date=nd_date.strftime("%Y-%m-%d"))
    else:
        sp_time = _sp_to_time_str(target_sp_raw)
        header = f"SP+{h} · SP {target_sp_raw} · {sp_time}"
        return dict(header=header, value=y0, p10=p10, p90=p90,
                    is_next_day=False, target_sp=target_sp_raw,
                    target_date=fc_date_str)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def horizons():
    assert BANDS_JSON.exists(), f"Missing {BANDS_JSON}"
    return json.load(open(BANDS_JSON))["horizons"]


Y0 = 95.0
FC_DATE = "2026-06-20"
NEXT_DATE = "2026-06-21"


# ── Tests: last_settled = 46 (SP+1=47, SP+2=48, SP+3=next-day SP 1) ─────────

class TestLastSP46:
    def test_h1_same_day(self, horizons):
        r = compute_nowcast_horizon(46, 1, FC_DATE, Y0, horizons)
        assert not r["is_next_day"]
        assert r["target_sp"] == 47
        assert r["target_date"] == FC_DATE
        assert "SP 47" in r["header"]
        assert "23:00" in r["header"]

    def test_h2_same_day(self, horizons):
        r = compute_nowcast_horizon(46, 2, FC_DATE, Y0, horizons)
        assert not r["is_next_day"]
        assert r["target_sp"] == 48
        assert r["target_date"] == FC_DATE
        assert "SP 48" in r["header"]
        assert "23:30" in r["header"]

    def test_h3_next_day(self, horizons):
        r = compute_nowcast_horizon(46, 3, FC_DATE, Y0, horizons)
        assert r["is_next_day"]
        assert r["target_sp"] == 1
        assert r["target_date"] == NEXT_DATE
        assert "SP 1" in r["header"]
        assert "00:00" in r["header"]
        assert "Jun 21" in r["header"]


# ── Tests: last_settled = 47 (SP+1=48, SP+2=next-day SP 1, SP+3=next-day SP 2)

class TestLastSP47:
    def test_h1_same_day(self, horizons):
        r = compute_nowcast_horizon(47, 1, FC_DATE, Y0, horizons)
        assert not r["is_next_day"]
        assert r["target_sp"] == 48
        assert "23:30" in r["header"]

    def test_h2_next_day_sp1(self, horizons):
        r = compute_nowcast_horizon(47, 2, FC_DATE, Y0, horizons)
        assert r["is_next_day"]
        assert r["target_sp"] == 1
        assert r["target_date"] == NEXT_DATE
        assert "00:00" in r["header"]
        assert "Jun 21" in r["header"]

    def test_h3_next_day_sp2(self, horizons):
        r = compute_nowcast_horizon(47, 3, FC_DATE, Y0, horizons)
        assert r["is_next_day"]
        assert r["target_sp"] == 2
        assert r["target_date"] == NEXT_DATE
        assert "00:30" in r["header"]
        assert "Jun 21" in r["header"]


# ── Tests: last_settled = 48 (all three horizons on next day) ─────────────────

class TestLastSP48:
    def test_h1_next_day_sp1(self, horizons):
        r = compute_nowcast_horizon(48, 1, FC_DATE, Y0, horizons)
        assert r["is_next_day"]
        assert r["target_sp"] == 1
        assert "00:00" in r["header"]

    def test_h2_next_day_sp2(self, horizons):
        r = compute_nowcast_horizon(48, 2, FC_DATE, Y0, horizons)
        assert r["is_next_day"]
        assert r["target_sp"] == 2
        assert "00:30" in r["header"]

    def test_h3_next_day_sp3(self, horizons):
        r = compute_nowcast_horizon(48, 3, FC_DATE, Y0, horizons)
        assert r["is_next_day"]
        assert r["target_sp"] == 3
        assert "01:00" in r["header"]


# ── Tests: bands are valid (non-NaN, sensible) for all cases ─────────────────

@pytest.mark.parametrize("last_sp,h", [
    (46, 1), (46, 2), (46, 3),
    (47, 1), (47, 2), (47, 3),
    (48, 1), (48, 2), (48, 3),
])
def test_bands_valid(horizons, last_sp, h):
    r = compute_nowcast_horizon(last_sp, h, FC_DATE, Y0, horizons)
    assert isinstance(r["value"], float)
    assert r["value"] == Y0
    assert isinstance(r["p10"], int)
    assert isinstance(r["p90"], int)
    assert r["p10"] < r["p90"], f"p10 {r['p10']} not < p90 {r['p90']} for SP+{h} from {last_sp}"
    assert -500 < r["p10"] < 2000
    assert -500 < r["p90"] < 2000


# ── Tests: band lookup is horizon-keyed, not SP-keyed ────────────────────────

def test_bands_horizon_keyed_not_sp_keyed(horizons):
    """SP+2 from last_sp=46 (→SP48) and SP+2 from last_sp=48 (→next-day SP2)
    must return the same h2 band — confirming lookup is on horizon, not target SP."""
    r_same_day  = compute_nowcast_horizon(46, 2, FC_DATE, Y0, horizons)
    r_next_day  = compute_nowcast_horizon(48, 2, FC_DATE, Y0, horizons)
    assert r_same_day["p10"] == r_next_day["p10"]
    assert r_same_day["p90"] == r_next_day["p90"]


# ── Tests: no next-day CSV required ──────────────────────────────────────────

def test_no_next_day_csv_required(horizons, tmp_path):
    """The computation only uses y0, horizons dict, and fc_date — no file I/O."""
    r = compute_nowcast_horizon(48, 3, FC_DATE, Y0, horizons)
    assert r["is_next_day"]
    assert isinstance(r["value"], float)

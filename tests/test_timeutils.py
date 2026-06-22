"""Unit tests for the shared UK-time helper and fetch_intraday's use of it.

Settlement periods/days are UK-local (Europe/London). The bug these guard
against: deriving the settlement day from the runner's UTC date, which in the
23:00–00:00 UTC window during BST points at the *previous* UK day.

Imports use the same bare module names the production code uses (`timeutils`,
`fetch_intraday`), so monkeypatching `timeutils.datetime` propagates into
fetch_intraday's imported `uk_today`.
"""
from datetime import date, datetime, timezone
from pathlib import Path

import sys

import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "data"))

import timeutils          # noqa: E402  (bare name — matches fetch_intraday's import)
import fetch_intraday     # noqa: E402


def _frozen_datetime(fixed_utc: datetime):
    """Return a datetime subclass whose now(tz) is pinned to `fixed_utc`."""
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_utc.astimezone(tz) if tz is not None else fixed_utc.replace(tzinfo=None)
    return _Frozen


# ── uk_today() ─────────────────────────────────────────────────────────────────

def test_uk_today_bst_late_night_rolls_to_next_uk_day(monkeypatch):
    # 2026-06-21 23:30 UTC = 2026-06-22 00:30 BST → the UK settlement day is the 22nd.
    fixed = datetime(2026, 6, 21, 23, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(timeutils, "datetime", _frozen_datetime(fixed))
    assert timeutils.uk_today() == date(2026, 6, 22)
    # Guard: this is exactly where a naive UTC date() would be wrong.
    assert fixed.date() == date(2026, 6, 21)
    assert timeutils.uk_today() != fixed.date()


def test_uk_today_winter_matches_utc(monkeypatch):
    # In January, Europe/London == GMT == UTC, so the dates agree.
    fixed = datetime(2026, 1, 15, 23, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(timeutils, "datetime", _frozen_datetime(fixed))
    assert timeutils.uk_today() == date(2026, 1, 15)
    assert timeutils.uk_today() == fixed.date()


def test_uk_tzname_summer_winter():
    assert timeutils.uk_tzname(date(2026, 6, 22)) == "BST"
    assert timeutils.uk_tzname(date(2026, 1, 15)) == "GMT"


# ── fetch_intraday wiring ───────────────────────────────────────────────────────

def test_fetch_intraday_fetches_uk_local_day_in_bst(monkeypatch):
    """A 23:30 UTC run in BST must fetch the new UK-local day, not yesterday."""
    fixed = datetime(2026, 6, 21, 23, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(timeutils, "datetime", _frozen_datetime(fixed))

    captured = {}

    def _fake_fetch_day(day, session):
        captured["day"] = day
        return pd.DataFrame()  # empty → main() returns without writing a file

    monkeypatch.setattr(fetch_intraday, "fetch_day", _fake_fetch_day)
    fetch_intraday.main()
    assert captured["day"] == "2026-06-22"


def test_fetch_intraday_fetches_same_day_in_winter(monkeypatch):
    fixed = datetime(2026, 1, 15, 23, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(timeutils, "datetime", _frozen_datetime(fixed))

    captured = {}

    def _fake_fetch_day(day, session):
        captured["day"] = day
        return pd.DataFrame()

    monkeypatch.setattr(fetch_intraday, "fetch_day", _fake_fetch_day)
    fetch_intraday.main()
    assert captured["day"] == "2026-01-15"

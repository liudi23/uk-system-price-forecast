"""Shared UK settlement-time helpers.

Settlement periods are defined on the UK-local clock (GMT in winter, BST in
summer): SP1 = 00:00–00:30 local. CI runners and most stamps are UTC, so any
SP number or settlement-day boundary MUST be derived in Europe/London — never
from the UTC clock — or it drifts by one hour (and one calendar day across the
BST midnight window, e.g. 23:00–00:00 UTC = 00:00–01:00 BST of the next UK day).
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

UK_TZ = ZoneInfo("Europe/London")


def uk_now() -> datetime:
    """Current time as a timezone-aware Europe/London datetime."""
    return datetime.now(UK_TZ)


def uk_today() -> date:
    """Current UK-local settlement date (Europe/London), not the UTC date."""
    return uk_now().date()


def uk_tzname(d: date | None = None) -> str:
    """'BST' or 'GMT' for a given UK date (defaults to now). Noon is used to
    avoid ambiguity around the DST-transition hour."""
    if d is None:
        return uk_now().tzname()
    return datetime(d.year, d.month, d.day, 12, tzinfo=UK_TZ).tzname()

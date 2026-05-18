"""
Calendar and temporal features for UK electricity price forecasting.

Feature set follows Weron (2014) and the epftoolbox LEAR benchmark:
  - Raw temporals : hour, settlement_period, day_of_week, month, week_of_year
  - Cyclic encoding : sin/cos transforms preserve circular structure
                      (hour 23 and hour 0 are adjacent, not distant)
  - Binary indicators : is_weekend, is_uk_holiday, is_peak, is_business_day
  - Peak definition : Elexon / National Grid convention — periods 13–36
                      (06:00–18:00 UTC), often called "peak" in UK trading

Reference:
  Weron, R. (2014). Electricity price forecasting: A review.
  International Journal of Forecasting, 30(4), 1030–1081.
"""

import numpy as np
import pandas as pd

try:
    import holidays as hols
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False


def _uk_holidays(years) -> set:
    if _HOLIDAYS_AVAILABLE:
        uk = hols.country_holidays("GB", subdiv="ENG")
        return {d for y in years for d in hols.country_holidays("GB", subdiv="ENG", years=y)}
    # Fallback: hardcode 2025–2026 England bank holidays
    return {
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp("2025-04-18").date(),
        pd.Timestamp("2025-04-21").date(),
        pd.Timestamp("2025-05-05").date(),
        pd.Timestamp("2025-05-26").date(),
        pd.Timestamp("2025-08-25").date(),
        pd.Timestamp("2025-12-25").date(),
        pd.Timestamp("2025-12-26").date(),
        pd.Timestamp("2026-01-01").date(),
        pd.Timestamp("2026-04-03").date(),
        pd.Timestamp("2026-04-06").date(),
        pd.Timestamp("2026-05-04").date(),
        pd.Timestamp("2026-05-25").date(),
        pd.Timestamp("2026-08-31").date(),
        pd.Timestamp("2026-12-25").date(),
        pd.Timestamp("2026-12-28").date(),
    }


def add_calendar_features(df: pd.DataFrame, datetime_col: str = "settlement_datetime") -> pd.DataFrame:
    """
    Add calendar and temporal features in-place and return the DataFrame.

    Expects `datetime_col` to be a datetime column (UTC settlement start time).
    Also works if `settlement_date` and `settlement_period` are present without
    a pre-built datetime column.
    """
    df = df.copy()

    if datetime_col not in df.columns:
        df[datetime_col] = pd.to_datetime(df["settlement_date"].astype(str)) + pd.to_timedelta(
            (df["settlement_period"] - 1) * 30, unit="min"
        )

    dt = pd.to_datetime(df[datetime_col])

    # ── Raw temporals ──────────────────────────────────────────────────────────
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek          # 0=Mon … 6=Sun
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["day_of_year"] = dt.dt.dayofyear           # 1–365/366

    # ── Cyclic encoding ────────────────────────────────────────────────────────
    # settlement_period cycles 1→48 each day
    sp = df["settlement_period"]
    df["sin_sp"] = np.sin(2 * np.pi * (sp - 1) / 48)
    df["cos_sp"] = np.cos(2 * np.pi * (sp - 1) / 48)

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # ── Annual modulation (confirmed via 5-year Kruskal-Wallis, p=5e-11) ──────
    # UK electricity prices peak in winter (Nov–Jan, heating demand) and
    # trough in spring (Apr–May). A W-shape requires both the 12-month
    # and 6-month harmonics to capture the secondary summer peak.
    doy = df["day_of_year"].astype(np.float64)
    df["sin_annual"]      = np.sin(2 * np.pi * doy / 365)   # 12-month cycle
    df["cos_annual"]      = np.cos(2 * np.pi * doy / 365)
    df["sin_half_annual"] = np.sin(4 * np.pi * doy / 365)   # 6-month cycle
    df["cos_half_annual"] = np.cos(4 * np.pi * doy / 365)

    # ── Binary indicators ──────────────────────────────────────────────────────
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)

    # UK peak: periods 13–36 → 06:00–17:30 UTC (Elexon convention)
    df["is_peak"] = ((sp >= 13) & (sp <= 36)).astype(np.int8)

    # Evening ramp: periods 33–40 → 16:00–19:30 (demand peak)
    df["is_evening_ramp"] = ((sp >= 33) & (sp <= 40)).astype(np.int8)

    # UK bank holidays (England)
    years = dt.dt.year.unique().tolist()
    uk_hol_set = _uk_holidays(years)
    dates = dt.dt.date
    df["is_uk_holiday"] = dates.apply(lambda d: d in uk_hol_set).astype(np.int8)

    # Business day: weekday and not a holiday
    df["is_business_day"] = ((df["is_weekend"] == 0) & (df["is_uk_holiday"] == 0)).astype(np.int8)

    return df

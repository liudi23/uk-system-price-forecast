"""
Phase 3 day-ahead SSP forecast: two-stage level-shape inference.

No recursive loop — both stages use only data settled before the forecast day:

  Stage 1 — Daily level model
      Inputs  : rolling daily SSP/NIV stats from history + day-ahead weather
      Output  : predicted daily mean SSP for target date (P10 / P50 / P90)

  Stage 2 — Intra-day shape model
      Inputs  : fixed-point lag-48+ features for each of the 48 SPs
                (ssp_lag_48/96/336, weather_lag_48, niv_lag_48, calendar)
      Output  : predicted deviation from daily mean per SP (P50)

  Final forecast per SP : level_P50 + deviation_P50
  Uncertainty band      : [level_P10 + deviation, level_P90 + deviation]

Output: model_assets/next_day_forecast_phase3.csv
Columns: settlement_datetime, settlement_date, settlement_period,
         ssp_predicted, ssp_q10, ssp_q50, ssp_q90

Usage:
    python src/models/forecast_phase3.py
    python src/models/forecast_phase3.py --date 2026-05-20
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from fetch_generation import fetch_range as _fetch_ci_range
from fetch_bmrs_forecasts import get_wind_pct_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ASSETS_DIR           = Path(__file__).resolve().parents[2] / "model_assets"
FEATURES_FILE        = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_5yr.csv"
FEATURES_RECENT_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_recent.csv"
RAW_PRICES_FILE      = Path(__file__).resolve().parents[2] / "data" / "raw" / "system_prices.csv"
OUTPUT_FILE     = ASSETS_DIR / "next_day_forecast_phase3.csv"       # H+1: today
OUTPUT_FILE_H2  = ASSETS_DIR / "day2_forecast_phase3.csv"           # H+2: tomorrow

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
UK_LOCATIONS = [
    {"lat": 52.5, "lon": -1.5, "weight": 0.6},
    {"lat": 56.5, "lon": -4.0, "weight": 0.2},
    {"lat": 52.3, "lon": -3.7, "weight": 0.2},
]
WEATHER_VARS   = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "precipitation"]
WEATHER_RENAME = {
    "temperature_2m":      "temp_c",
    "wind_speed_10m":      "wind_ms",
    "shortwave_radiation": "solar_wm2",
    "precipitation":       "precip_mm",
}
HEATING_BASE = 15.5
COOLING_BASE = 22.0
HISTORY_DAYS = 40   # rolling window for daily features (28-day max lag + buffer)


# ── Weather ───────────────────────────────────────────────────────────────────

def fetch_forecast_weather(target_date: date) -> pd.DataFrame:
    start = str(target_date - timedelta(days=2))
    end   = str(target_date)
    frames = []
    for loc in UK_LOCATIONS:
        params = {
            "latitude": loc["lat"], "longitude": loc["lon"],
            "hourly": ",".join(WEATHER_VARS),
            "start_date": start, "end_date": end,
            "timezone": "UTC", "wind_speed_unit": "ms",
        }
        r = requests.get(FORECAST_URL, params=params, timeout=30)
        r.raise_for_status()
        hourly = r.json()["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
        for v in WEATHER_VARS:
            df[v] = df[v] * loc["weight"]
        frames.append(df[["time"] + WEATHER_VARS])
    combined = frames[0].copy()
    for v in WEATHER_VARS:
        combined[v] = sum(f[v] for f in frames)
    combined = (
        combined.set_index("time").resample("30min").ffill().reset_index()
        .rename(columns={**WEATHER_RENAME, "time": "dt"})
    )
    return combined


def _day_weather(weather_df: pd.DataFrame, d: date) -> dict:
    """Extract 48-SP weather arrays for a given date."""
    start = pd.Timestamp(d)
    end   = start + pd.Timedelta(hours=23, minutes=30)
    day   = weather_df[(weather_df["dt"] >= start) & (weather_df["dt"] <= end)].reset_index(drop=True)
    while len(day) < 48:
        day = pd.concat([day, day.iloc[[-1]]], ignore_index=True)
    return {col: day[col].values[:48] for col in ["temp_c", "wind_ms", "solar_wm2", "precip_mm"]}


# ── Calendar ──────────────────────────────────────────────────────────────────

def _calendar(dt: pd.Timestamp) -> dict:
    sp  = dt.hour * 2 + dt.minute // 30 + 1
    doy = dt.day_of_year
    return {
        "hour":            dt.hour,
        "day_of_week":     dt.dayofweek,
        "month":           dt.month,
        "week_of_year":    int(dt.isocalendar().week),
        "quarter":         dt.quarter,
        "day_of_year":     doy,
        "sin_sp":          np.sin(2 * np.pi * sp / 48),
        "cos_sp":          np.cos(2 * np.pi * sp / 48),
        "sin_hour":        np.sin(2 * np.pi * dt.hour / 24),
        "cos_hour":        np.cos(2 * np.pi * dt.hour / 24),
        "sin_dow":         np.sin(2 * np.pi * dt.dayofweek / 7),
        "cos_dow":         np.cos(2 * np.pi * dt.dayofweek / 7),
        "sin_month":       np.sin(2 * np.pi * dt.month / 12),
        "cos_month":       np.cos(2 * np.pi * dt.month / 12),
        "sin_annual":      np.sin(2 * np.pi * doy / 365),
        "cos_annual":      np.cos(2 * np.pi * doy / 365),
        "sin_half_annual": np.sin(4 * np.pi * doy / 365),
        "cos_half_annual": np.cos(4 * np.pi * doy / 365),
        "is_weekend":      int(dt.dayofweek >= 5),
        "is_peak":         int(13 <= sp <= 36),
        "is_evening_ramp": int(33 <= sp <= 40),
        "is_uk_holiday":   0,
        "is_business_day": int(dt.dayofweek < 5),
    }


# ── Stage 1: level features from rolling daily history ────────────────────────

def build_level_features(daily_hist: pd.DataFrame, target_date: date,
                          target_weather: dict) -> dict:
    """
    Compute level model features for target_date from the daily history
    DataFrame (one row per past day).

    daily_hist must be sorted ascending and contain:
        date, ssp_daily_mean, ssp_raw_daily_max, niv_daily_mean, is_spike_count
        and optionally temp_c_daily_mean, wind_ms_daily_mean, solar_wm2_daily_mean
    """
    h = daily_hist.sort_values("date").reset_index(drop=True)

    # Calendar for target date
    td  = pd.Timestamp(target_date)
    doy = td.day_of_year
    dow = td.dayofweek
    mon = td.month

    feat: dict = {
        "day_of_week":     dow,
        "month":           mon,
        "day_of_year":     doy,
        "quarter":         td.quarter,
        "is_weekend":      int(dow >= 5),
        "is_business_day": int(dow < 5),
        "sin_dow":         np.sin(2 * np.pi * dow / 7),
        "cos_dow":         np.cos(2 * np.pi * dow / 7),
        "sin_month":       np.sin(2 * np.pi * mon / 12),
        "cos_month":       np.cos(2 * np.pi * mon / 12),
        "sin_annual":      np.sin(2 * np.pi * doy / 365),
        "cos_annual":      np.cos(2 * np.pi * doy / 365),
        "sin_half_annual": np.sin(4 * np.pi * doy / 365),
        "cos_half_annual": np.cos(4 * np.pi * doy / 365),
    }

    # SSP daily lag features (shift ≥ 1 day)
    ssp_series  = h["ssp_daily_mean"].values
    raw_max_ser = h["ssp_raw_daily_max"].values
    raw_min_ser = h["ssp_raw_daily_min"].values if "ssp_raw_daily_min" in h.columns else ssp_series

    for lag_d, idx in [(1, -1), (2, -2), (7, -7), (14, -14), (28, -28)]:
        feat[f"ssp_daily_mean_lag{lag_d}d"]    = float(ssp_series[idx])   if len(ssp_series) >= abs(idx) else np.nan
        feat[f"ssp_raw_daily_max_lag{lag_d}d"] = float(raw_max_ser[idx])  if len(raw_max_ser) >= abs(idx) else np.nan
        feat[f"ssp_raw_daily_min_lag{lag_d}d"] = float(raw_min_ser[idx])  if len(raw_min_ser) >= abs(idx) else np.nan

    for w in [7, 14, 28]:
        win = ssp_series[-w:] if len(ssp_series) >= w else ssp_series
        feat[f"ssp_daily_roll_mean_{w}d"] = float(win.mean())
        feat[f"ssp_daily_roll_std_{w}d"]  = float(win.std(ddof=1)) if len(win) > 1 else 0.0
        feat[f"ssp_daily_roll_max_{w}d"]  = float(win.max())
        feat[f"ssp_daily_roll_min_{w}d"]  = float(win.min())
        raw_win = raw_max_ser[-w:] if len(raw_max_ser) >= w else raw_max_ser
        feat[f"ssp_raw_max_roll_mean_{w}d"] = float(raw_win.mean())

    # NIV daily lags
    niv_series = h["niv_daily_mean"].values
    for lag_d, idx in [(1, -1), (7, -7)]:
        feat[f"niv_daily_mean_lag{lag_d}d"] = float(niv_series[idx]) if len(niv_series) >= abs(idx) else np.nan
    for w in [7, 14]:
        niv_win = niv_series[-w:] if len(niv_series) >= w else niv_series
        feat[f"niv_daily_roll_mean_{w}d"] = float(niv_win.mean())
        feat[f"niv_daily_roll_std_{w}d"]  = float(niv_win.std(ddof=1)) if len(niv_win) > 1 else 0.0

    # Spike count lags
    spk_series = h["is_spike_count"].values if "is_spike_count" in h.columns else np.zeros(len(h))
    for lag_d, idx in [(1, -1), (7, -7)]:
        feat[f"spike_count_lag{lag_d}d"] = float(spk_series[idx]) if len(spk_series) >= abs(idx) else 0.0
    spk_win = spk_series[-7:] if len(spk_series) >= 7 else spk_series
    feat["spike_count_roll_7d"] = float(spk_win.sum())

    # Negative-price SP count lags — renewable oversupply regime signal
    neg_series = h["neg_sp_daily_count"].values if "neg_sp_daily_count" in h.columns else np.zeros(len(h))
    feat["neg_sp_count_lag1d"]   = float(neg_series[-1])  if len(neg_series) >= 1 else 0.0
    feat["neg_sp_count_lag7d"]   = float(neg_series[-7])  if len(neg_series) >= 7 else 0.0
    feat["neg_sp_count_roll_7d"] = float(neg_series[-7:].sum()) if len(neg_series) >= 1 else 0.0

    # Weather for target_date (day-ahead forecast)
    feat["temp_c_daily_mean"]   = float(np.mean(target_weather["temp_c"]))
    feat["temp_c_daily_max"]    = float(np.max(target_weather["temp_c"]))
    feat["wind_ms_daily_mean"]  = float(np.mean(target_weather["wind_ms"]))
    feat["wind_ms_daily_max"]   = float(np.max(target_weather["wind_ms"]))
    feat["solar_wm2_daily_mean"] = float(np.mean(target_weather["solar_wm2"]))
    feat["solar_wm2_daily_max"]  = float(np.max(target_weather["solar_wm2"]))
    feat["precip_mm_daily_mean"] = float(np.mean(target_weather["precip_mm"]))

    # Weather lag1d (yesterday's daily mean — from history)
    if "temp_c_daily_mean" in h.columns:
        feat["temp_c_daily_mean_lag1d"]    = float(h["temp_c_daily_mean"].values[-1])
        feat["temp_c_daily_mean_lag7d"]    = float(h["temp_c_daily_mean"].values[-7]) if len(h) >= 7 else float(h["temp_c_daily_mean"].values[0])
        feat["temp_c_daily_max_lag1d"]     = float(h["temp_c_daily_max"].values[-1])  if "temp_c_daily_max" in h.columns else feat["temp_c_daily_mean_lag1d"]
    if "wind_ms_daily_mean" in h.columns:
        feat["wind_ms_daily_mean_lag1d"]   = float(h["wind_ms_daily_mean"].values[-1])
        feat["wind_ms_daily_mean_lag7d"]   = float(h["wind_ms_daily_mean"].values[-7]) if len(h) >= 7 else float(h["wind_ms_daily_mean"].values[0])
        feat["wind_ms_daily_max_lag1d"]    = float(h["wind_ms_daily_max"].values[-1])  if "wind_ms_daily_max" in h.columns else feat["wind_ms_daily_mean_lag1d"]
    if "solar_wm2_daily_mean" in h.columns:
        feat["solar_wm2_daily_mean_lag1d"] = float(h["solar_wm2_daily_mean"].values[-1])
        feat["solar_wm2_daily_mean_lag7d"] = float(h["solar_wm2_daily_mean"].values[-7]) if len(h) >= 7 else float(h["solar_wm2_daily_mean"].values[0])
        feat["solar_wm2_daily_max_lag1d"]  = float(h["solar_wm2_daily_max"].values[-1]) if "solar_wm2_daily_max" in h.columns else feat["solar_wm2_daily_mean_lag1d"]

    # Generation mix lag features (wind %, gas %) — low wind → high gas → high prices
    for _gcol, _prefix in [("wind_pct_daily_mean", "wind_pct"), ("gas_pct_daily_mean", "gas_pct")]:
        if _gcol in h.columns and h[_gcol].notna().any():
            _vals = h[_gcol].values
            feat[f"{_gcol}_lag1d"] = float(_vals[-1]) if not np.isnan(_vals[-1]) else 0.0
            feat[f"{_gcol}_lag7d"] = (float(_vals[-7]) if len(_vals) >= 7 and not np.isnan(_vals[-7]) else feat[f"{_gcol}_lag1d"])
            _w7 = _vals[-7:][~np.isnan(_vals[-7:])]
            feat[f"{_prefix}_daily_roll_mean_7d"] = float(_w7.mean()) if len(_w7) > 0 else 0.0

    return feat


# ── Stage 2: shape features for one SP ───────────────────────────────────────

def build_shape_row(h_idx: int, dt: pd.Timestamp,
                    ssp_hist: np.ndarray, ssp_raw_hist: np.ndarray,
                    is_spike_hist: np.ndarray, niv_hist: np.ndarray,
                    weather_hist: dict,
                    daily_mean_lag1d: float, daily_mean_lag7d: float,
                    daily_stats: dict = None,
                    gen_hist: dict = None,
                    wind_pct_forecast: np.ndarray = None,
                    solar_forecast: np.ndarray = None) -> dict:
    """
    Build shape feature vector for one SP (h_idx = 0-based index within
    the forecast day, so lag_48 references the same SP of yesterday).

    All features use fixed lags ≥ 48 SPs — guaranteed leakage-free for
    every SP in the 48-period forecast window.

    wind_pct_forecast : 48-element array of day-ahead wind % from BMRS WINDFOR/TSDF.
        Substitutes into wind_pct_lag_48 — genuine day-ahead forecast instead of
        CI lag-48 actual. Same pattern as weather: actual at training, forecast at inference.

    solar_forecast : 48-element array of day-ahead solar irradiance (W/m²) from
        Open-Meteo hourly data. Substitutes into solar_wm2_lag_48.
        At training: solar_wm2_lag_48 = yesterday's actual irradiance (proxy).
        At inference: today's Open-Meteo day-ahead forecast for each SP.
    """
    HIST = len(ssp_hist)  # must be ≥ 336 (7 days of actual history)
    row = _calendar(dt)

    # ── Fixed-point SSP lags (winsorised) ─────────────────────────────────
    row["ssp_lag_48"]  = float(ssp_hist[HIST - 48  + h_idx])   # same SP yesterday
    row["ssp_lag_96"]  = float(ssp_hist[HIST - 96  + h_idx])
    row["ssp_lag_336"] = float(ssp_hist[HIST - 336 + h_idx])   # same SP last week
    row["ssp_diff_48_336"] = row["ssp_lag_48"] - row["ssp_lag_336"]

    # ── Raw SSP and spike flags (actual history, lag ≥ 48) ────────────────
    row["ssp_raw_lag_48"]   = float(ssp_raw_hist[HIST - 48  + h_idx])
    row["ssp_raw_lag_96"]   = float(ssp_raw_hist[HIST - 96  + h_idx])
    row["ssp_raw_lag_336"]  = float(ssp_raw_hist[HIST - 336 + h_idx])
    row["is_spike_lag_48"]  = float(is_spike_hist[HIST - 48  + h_idx])
    row["is_spike_lag_336"] = float(is_spike_hist[HIST - 336 + h_idx])
    row["is_negative_lag_48"]  = float(ssp_raw_hist[HIST - 48  + h_idx] < 0)
    row["is_negative_lag_336"] = float(ssp_raw_hist[HIST - 336 + h_idx] < 0)

    # ── NIV at lag-48 (same SP yesterday) ────────────────────────────────
    row["net_imbalance_volume_lag_48"]  = float(niv_hist[HIST - 48  + h_idx])
    row["net_imbalance_volume_lag_336"] = float(niv_hist[HIST - 336 + h_idx])

    # ── Weather at lag-48 (same SP yesterday's actual weather) ───────────
    for wvar in ["temp_c", "wind_ms", "solar_wm2", "precip_mm"]:
        arr = weather_hist[wvar]
        row[f"{wvar}_lag_48"]  = float(arr[HIST - 48  + h_idx])
        row[f"{wvar}_lag_336"] = float(arr[HIST - 336 + h_idx])

    temp_lag48 = weather_hist["temp_c"][HIST - 48 + h_idx]
    row["heating_degree"] = float(max(0.0, HEATING_BASE - temp_lag48))
    row["cooling_degree"] = float(max(0.0, temp_lag48 - COOLING_BASE))

    # ── Daily-level lag features (from level_features) ────────────────────
    row["ssp_daily_mean_lag1d"]    = daily_mean_lag1d
    row["ssp_daily_mean_lag7d"]    = daily_mean_lag7d
    row["ssp_lag48_deviation"]     = row["ssp_lag_48"]  - daily_mean_lag1d
    row["ssp_lag336_deviation"]    = row["ssp_lag_336"] - daily_mean_lag7d

    # ── Additional daily-aggregated shape features ────────────────────────
    if daily_stats:
        row.update(daily_stats)

    # ── Generation mix lag features (lag-48 and lag-336) ─────────────────
    if gen_hist:
        for _gvar in ["wind_pct", "gas_pct"]:
            _garr = gen_hist.get(_gvar, np.zeros(HIST))
            row[f"{_gvar}_lag_48"]  = float(_garr[HIST - 48  + h_idx])
            row[f"{_gvar}_lag_336"] = float(_garr[HIST - 336 + h_idx]) if HIST >= 336 else 0.0

    # ── WINDFOR substitution: replace wind_pct_lag_48 with day-ahead forecast ──
    if wind_pct_forecast is not None and np.isfinite(wind_pct_forecast[h_idx]):
        row["wind_pct_lag_48"] = float(wind_pct_forecast[h_idx])

    # ── Solar substitution: replace solar_wm2_lag_48 with day-ahead forecast ──
    # Open-Meteo hourly shortwave_radiation already fetched for the target date.
    # solar_wm2_lag_48 is the top-ranked shape feature (importance 0.11);
    # providing the genuine day-ahead value instead of yesterday's actual is
    # the largest single inference-time signal improvement available.
    if solar_forecast is not None and np.isfinite(solar_forecast[h_idx]):
        row["solar_wm2_lag_48"] = float(solar_forecast[h_idx])

    return row


def build_shape_row_h2(h_idx: int, dt: pd.Timestamp,
                       ssp_hist: np.ndarray, ssp_raw_hist: np.ndarray,
                       is_spike_hist: np.ndarray, niv_hist: np.ndarray,
                       weather_hist: dict,
                       daily_mean_lag2d: float, daily_mean_lag7d: float,
                       daily_stats_h2: dict = None,
                       gen_hist: dict = None,
                       wind_pct_forecast_h2: np.ndarray = None,
                       solar_forecast_h2: np.ndarray = None) -> dict:
    """
    Build H+2 shape feature vector for one SP of target_date+1 (tomorrow).

    All features use lag ≥ 96 SPs so they reference settled actuals from
    at least D (today's settled data), not D+1 which won't be available.

    lag-96  from D+2 = same SP of D   (today's settled prices)
    lag-336 from D+2 = same SP of D-5 (7 days before D+2)
    """
    HIST = len(ssp_hist)
    row  = _calendar(dt)

    # ── Lag-96/336 as primary same-SP signals ─────────────────────────────
    row["ssp_lag_96"]  = float(ssp_hist[HIST - 96  + h_idx])
    row["ssp_lag_336"] = float(ssp_hist[HIST - 336 + h_idx])

    row["ssp_raw_lag_96"]   = float(ssp_raw_hist[HIST - 96  + h_idx])
    row["ssp_raw_lag_336"]  = float(ssp_raw_hist[HIST - 336 + h_idx])
    row["is_spike_lag_336"] = float(is_spike_hist[HIST - 336 + h_idx])
    row["is_negative_lag_336"] = float(ssp_raw_hist[HIST - 336 + h_idx] < 0)

    row["ssp_diff_48_336"]  = row["ssp_lag_96"] - row["ssp_lag_336"]  # lag96 as proxy

    # ── NIV lag-336 ───────────────────────────────────────────────────────
    row["net_imbalance_volume_lag_336"] = float(niv_hist[HIST - 336 + h_idx])

    # ── Weather lag-336 ───────────────────────────────────────────────────
    for wvar in ["temp_c", "wind_ms", "solar_wm2", "precip_mm"]:
        arr = weather_hist[wvar]
        row[f"{wvar}_lag_336"] = float(arr[HIST - 336 + h_idx])

    temp_lag336 = weather_hist["temp_c"][HIST - 336 + h_idx]
    row["heating_degree"] = float(max(0.0, HEATING_BASE - temp_lag336))
    row["cooling_degree"] = float(max(0.0, temp_lag336 - COOLING_BASE))

    # ── Daily-level H+2 features (lag-2d, lag-7d relative to D+2) ─────────
    row["ssp_daily_mean_lag2d"] = daily_mean_lag2d
    row["ssp_daily_mean_lag7d"] = daily_mean_lag7d
    row["ssp_lag96_deviation"]  = row["ssp_lag_96"]  - daily_mean_lag2d
    row["ssp_lag336_deviation"] = row["ssp_lag_336"] - daily_mean_lag7d

    if daily_stats_h2:
        row.update(daily_stats_h2)

    # ── Generation lags (lag-336 only — lag-48 not safe for H+2) ─────────
    if gen_hist:
        for _gvar in ["wind_pct", "gas_pct"]:
            _garr = gen_hist.get(_gvar, np.zeros(HIST))
            row[f"{_gvar}_lag_336"] = float(_garr[HIST - 336 + h_idx]) if HIST >= 336 else 0.0

    # ── Forecast substitutions for H+2 (D+2 day-ahead signals) ───────────
    if wind_pct_forecast_h2 is not None and np.isfinite(wind_pct_forecast_h2[h_idx]):
        row["wind_pct_lag_336"] = float(wind_pct_forecast_h2[h_idx])
    if solar_forecast_h2 is not None and np.isfinite(solar_forecast_h2[h_idx]):
        row["solar_wm2_lag_336"] = float(solar_forecast_h2[h_idx])

    return row


# ── Main forecast ─────────────────────────────────────────────────────────────

def run_forecast(target_date: date = None, features_file: str = None) -> pd.DataFrame:
    # ── Load models ───────────────────────────────────────────────────────
    level_q10 = joblib.load(ASSETS_DIR / "level_q10.pkl")
    level_q50 = joblib.load(ASSETS_DIR / "level_q50.pkl")
    level_q90 = joblib.load(ASSETS_DIR / "level_q90.pkl")
    shape_q50 = joblib.load(ASSETS_DIR / "shape_q50.pkl")

    with open(ASSETS_DIR / "level_feature_cols.json") as f:
        level_feat_cols = json.load(f)
    with open(ASSETS_DIR / "shape_feature_cols.json") as f:
        shape_feat_cols = json.load(f)
    with open(ASSETS_DIR / "tukey_fence.json") as f:
        fence = json.load(f)

    # H+2 shape model (optional — graceful fallback if not trained yet)
    _h2_pkl  = ASSETS_DIR / "shape_h2_q50.pkl"
    _h2_json = ASSETS_DIR / "shape_h2_feature_cols.json"
    shape_h2_q50       = joblib.load(_h2_pkl)              if _h2_pkl.exists()  else None
    shape_h2_feat_cols = json.load(open(_h2_json))         if _h2_json.exists() else []

    # Optional: negative-price regime classifier
    _neg_clf_path    = ASSETS_DIR / "neg_day_classifier.pkl"
    _neg_feats_path  = ASSETS_DIR / "neg_day_classifier_feats.json"
    neg_clf          = joblib.load(_neg_clf_path)   if _neg_clf_path.exists()   else None
    neg_clf_feats    = json.load(open(_neg_feats_path)) if _neg_feats_path.exists() else []
    upper_fence = fence["upper"]

    # ── Load history ──────────────────────────────────────────────────────
    _features_path = Path(features_file) if features_file else FEATURES_FILE
    if not _features_path.exists() and FEATURES_RECENT_FILE.exists():
        log.info("Full features not found — falling back to features_recent.csv")
        _features_path = FEATURES_RECENT_FILE
    log.info("Loading feature history from %s …", _features_path.name)
    base = pd.read_csv(_features_path, parse_dates=["settlement_datetime"])
    base = base.sort_values("settlement_datetime").reset_index(drop=True)
    last_dt = base["settlement_datetime"].max()

    # Extend with any newer rows from raw prices file
    extra = pd.DataFrame()
    if RAW_PRICES_FILE.exists():
        raw = pd.read_csv(RAW_PRICES_FILE, parse_dates=["settlement_date"])
        raw["settlement_datetime"] = (
            raw["settlement_date"]
            + pd.to_timedelta((raw["settlement_period"] - 1) * 30, unit="min")
        )
        extra = raw[raw["settlement_datetime"] > last_dt].sort_values("settlement_datetime").copy()
        if not extra.empty:
            extra["ssp_raw"]  = extra["ssp"]
            extra["is_spike"] = extra["ssp_raw"] > upper_fence
            log.info("Extending history with %d rows (%s → %s)",
                     len(extra), extra["settlement_datetime"].min().date(),
                     extra["settlement_datetime"].max().date())

    sp_all = pd.concat([base, extra], ignore_index=True).sort_values("settlement_datetime")
    combined_last_dt = sp_all["settlement_datetime"].max()

    # ── Fetch recent CI generation mix for extra rows not in features_5yr ──
    if "wind_pct" not in sp_all.columns or sp_all["wind_pct"].isna().any():
        _ci_start = (combined_last_dt - pd.Timedelta(days=9)).date()
        _ci_end   = combined_last_dt.date()
        if _ci_start <= _ci_end:
            log.info("Fetching Carbon Intensity generation mix %s → %s …", _ci_start, _ci_end)
            try:
                _ci = _fetch_ci_range(_ci_start, _ci_end, delay=0.2)
                if not _ci.empty:
                    _ci["settlement_datetime"] = (
                        pd.to_datetime(_ci["settlement_date"]) +
                        pd.to_timedelta((_ci["settlement_period"] - 1) * 30, unit="min")
                    )
                    _ci_merge = _ci[["settlement_datetime", "wind_pct", "gas_pct"]].rename(
                        columns={"wind_pct": "_ci_wind", "gas_pct": "_ci_gas"})
                    for _col in ["wind_pct", "gas_pct"]:
                        if _col not in sp_all.columns:
                            sp_all[_col] = np.nan
                    sp_all = sp_all.merge(_ci_merge, on="settlement_datetime", how="left")
                    sp_all["wind_pct"] = sp_all["wind_pct"].fillna(sp_all.pop("_ci_wind"))
                    sp_all["gas_pct"]  = sp_all["gas_pct"].fillna(sp_all.pop("_ci_gas"))
                    log.info("  wind_pct coverage after CI fill: %.1f%%",
                             sp_all["wind_pct"].notna().mean() * 100)
            except Exception as _e:
                log.warning("CI fetch failed: %s — wind/gas features will be missing", _e)

    if target_date is None:
        target_date = (combined_last_dt + pd.Timedelta(minutes=30)).date()
    log.info("Forecasting %s (history ends %s)", target_date, combined_last_dt.date())

    # ── Fetch BMRS WINDFOR day-ahead wind % forecast ─────────────────────
    wind_pct_forecast = None
    try:
        log.info("Fetching BMRS WINDFOR + TSDF day-ahead wind forecast for %s …", target_date)
        wind_pct_forecast = get_wind_pct_forecast(target_date)
        # Also update level feature: wind_pct_daily_mean_lag1d → use forecast mean
        # (same substitution as Open-Meteo weather: actual at training, forecast at inference)
        _wf_daily_mean = float(np.nanmean(wind_pct_forecast))
        log.info("  WINDFOR daily mean for %s: %.1f%% (vs CI lag-1d proxy)", target_date, _wf_daily_mean)
    except Exception as _e:
        log.warning("WINDFOR fetch failed: %s — using CI lag-48 proxy for wind", _e)
        wind_pct_forecast = None
        _wf_daily_mean = None

    # ── Fetch day-ahead weather ───────────────────────────────────────────
    log.info("Fetching Open-Meteo weather …")
    weather_df = fetch_forecast_weather(target_date)
    target_weather = _day_weather(weather_df, target_date)

    # ── Build daily history for level features ────────────────────────────
    sp_all["_date"] = sp_all["settlement_datetime"].dt.date

    raw_col = "_raw_temp_c" if "_raw_temp_c" in sp_all.columns else None
    agg_spec: dict = {
        "ssp_daily_mean":    ("ssp",                  "mean"),
        "ssp_raw_daily_max": ("ssp_raw",               "max"),
        "ssp_raw_daily_min": ("ssp_raw",               "min"),
        "niv_daily_mean":    ("net_imbalance_volume",  "mean"),
        "is_spike_count":    ("is_spike",              "sum"),
    }
    if raw_col:
        agg_spec["temp_c_daily_mean"]    = (raw_col,            "mean")
        agg_spec["temp_c_daily_max"]     = (raw_col,            "max")
    for wc, wn in [("_raw_wind_ms","wind_ms"), ("_raw_solar_wm2","solar_wm2")]:
        if wc in sp_all.columns:
            agg_spec[f"{wn}_daily_mean"] = (wc, "mean")
            agg_spec[f"{wn}_daily_max"]  = (wc, "max")
    for _gv in ["wind_pct", "gas_pct"]:
        if _gv in sp_all.columns:
            agg_spec[f"{_gv}_daily_mean"] = (_gv, "mean")
    # Negative-price SP count per day (leading indicator of renewable oversupply regime)
    if "ssp_raw" in sp_all.columns:
        sp_all["_is_neg"] = (sp_all["ssp_raw"] < 0).astype(float)
        agg_spec["neg_sp_daily_count"] = ("_is_neg", "sum")

    daily_hist = (
        sp_all[sp_all["_date"] < target_date]
        .groupby("_date").agg(**agg_spec)
        .reset_index()
        .rename(columns={"_date": "date"})
        .sort_values("date")
        .tail(HISTORY_DAYS)
        .reset_index(drop=True)
    )
    daily_hist["date"] = pd.to_datetime(daily_hist["date"])

    daily_mean_lag1d = float(daily_hist["ssp_daily_mean"].values[-1])
    daily_mean_lag7d = float(daily_hist["ssp_daily_mean"].values[-7]) if len(daily_hist) >= 7 else daily_mean_lag1d

    # Daily stats passed to build_shape_row() for the 8 daily-aggregated shape features
    _tail7 = daily_hist["ssp_daily_mean"].values[-7:]
    daily_stats: dict = {
        "ssp_raw_daily_max_lag1d": float(daily_hist["ssp_raw_daily_max"].values[-1]),
        "niv_daily_mean_lag1d":    float(daily_hist["niv_daily_mean"].values[-1]),
        "spike_count_lag1d":       float(daily_hist["is_spike_count"].values[-1]),
        "spike_count_roll_7d":     float(daily_hist["is_spike_count"].values[-7:].sum()),
        "ssp_daily_roll_mean_7d":  float(_tail7.mean()),
        "ssp_daily_roll_std_7d":   float(_tail7.std(ddof=1)) if len(_tail7) > 1 else 0.0,
    }
    for _wv in ("temp_c", "wind_ms", "solar_wm2"):
        _col = f"{_wv}_daily_mean"
        if _col in daily_hist.columns:
            daily_stats[f"{_wv}_daily_mean_lag1d"] = float(daily_hist[_col].values[-1])

    # ── Stage 1: predict daily level ─────────────────────────────────────
    lf = build_level_features(daily_hist, target_date, target_weather)

    # Intraday realized features: SP1-25 partial-day actuals from today's API data.
    # Provides a strong anchor for the level prediction — SP1-25 cover ~52% of the day.
    _intraday_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "intraday_prices.csv"
    _id_today_all: pd.DataFrame = pd.DataFrame()   # all settled SPs — used for post-processing
    if _intraday_path.exists():
        try:
            _id = pd.read_csv(_intraday_path)
            _id_today = str(target_date)
            if "settlement_date" in _id.columns:
                _id = _id[_id["settlement_date"].astype(str) == _id_today]
            # SP ≤ 25 for level model (consistent with training-time cutoff at 12:30 UTC)
            _id_partial = _id[_id["settlement_period"] <= 25]
            if not _id_partial.empty:
                _price_col = "ssp" if "ssp" in _id_partial.columns else "systemSellPrice"
                lf["intraday_realized_mean"] = float(_id_partial[_price_col].mean())
                if "net_imbalance_volume" in _id_partial.columns:
                    lf["intraday_realized_niv_mean"] = float(_id_partial["net_imbalance_volume"].mean())
                lf["intraday_n_settled"] = len(_id_partial)
                if "price_derivation_code" in _id_partial.columns:
                    lf["intraday_pct_P"] = float((_id_partial["price_derivation_code"] == "P").mean())
                log.info("Intraday features: realized_mean=£%.1f  niv_mean=%.0f  n_settled=%d",
                         lf["intraday_realized_mean"],
                         lf.get("intraday_realized_niv_mean", float("nan")),
                         lf["intraday_n_settled"])
            # All available SPs — for result post-processing below (Options 1 & 2)
            _id_today_all = _id.copy()
        except Exception as _e:
            log.warning("Could not load intraday features: %s", _e)

    # Substitute WINDFOR daily mean into level feature wind_pct_daily_mean_lag1d
    if _wf_daily_mean is not None and "wind_pct_daily_mean_lag1d" in level_feat_cols:
        lf["wind_pct_daily_mean_lag1d"] = _wf_daily_mean
        log.info("  Level feature wind_pct_daily_mean_lag1d → WINDFOR %.1f%% (was CI lag-1d)", _wf_daily_mean)

    # Negative-price regime classifier: compute P(≥3 negative SPs tomorrow)
    # and inject as level feature so the level model can predict low daily means
    # on high-renewable-oversupply days.
    if neg_clf is not None and neg_clf_feats:
        try:
            _x_neg = np.array([[lf.get(f, 0.0) for f in neg_clf_feats]])
            lf["neg_price_risk_prob"] = float(neg_clf.predict(_x_neg)[0])
            log.info("  neg_price_risk_prob = %.3f", lf["neg_price_risk_prob"])
        except Exception as _e:
            log.warning("Neg-day classifier inference failed: %s", _e)
    x_lvl = np.array([[lf.get(c, 0.0) for c in level_feat_cols]])
    pred_l10 = float(level_q10.predict(x_lvl)[0])
    pred_l50 = float(level_q50.predict(x_lvl)[0])
    pred_l90 = float(level_q90.predict(x_lvl)[0])
    pred_l10 = min(pred_l10, pred_l50)
    pred_l90 = max(pred_l90, pred_l50)
    log.info("Level prediction: P10=£%.1f  P50=£%.1f  P90=£%.1f (daily mean)",
             pred_l10, pred_l50, pred_l90)

    # ── Build SP-level history arrays for shape features ──────────────────
    HIST_SPS = 400
    sp_hist = sp_all[sp_all["_date"] < target_date].tail(HIST_SPS).reset_index(drop=True)

    ssp_arr      = sp_hist["ssp"].values.astype(float)
    ssp_raw_arr  = sp_hist["ssp_raw"].values.astype(float) if "ssp_raw" in sp_hist.columns else ssp_arr
    is_spike_arr = sp_hist["is_spike"].values.astype(float) if "is_spike" in sp_hist.columns else (ssp_raw_arr > upper_fence).astype(float)
    niv_arr      = sp_hist["net_imbalance_volume"].values.astype(float)

    # Weather history arrays (SP-level)
    def _weather_arr(col):
        if col in sp_hist.columns:
            return sp_hist[col].values.astype(float)
        return np.zeros(len(sp_hist))

    weather_hist = {
        "temp_c":    _weather_arr("_raw_temp_c"),
        "wind_ms":   _weather_arr("_raw_wind_ms"),
        "solar_wm2": _weather_arr("_raw_solar_wm2"),
        "precip_mm": _weather_arr("_raw_precip_mm"),
    }
    gen_hist = {
        "wind_pct": _weather_arr("wind_pct"),
        "gas_pct":  _weather_arr("gas_pct"),
    }

    n_hist = len(sp_hist)
    if n_hist < 336:
        raise RuntimeError(f"Insufficient SP history: need 336, have {n_hist}")

    # ── Stage 2: predict shape deviations for all 48 SPs ─────────────────
    base_dt = pd.Timestamp(target_date)
    records = []

    shape_rows = []
    for h in range(48):
        dt  = base_dt + pd.Timedelta(minutes=30 * h)
        row = build_shape_row(
            h_idx=h, dt=dt,
            ssp_hist=ssp_arr, ssp_raw_hist=ssp_raw_arr,
            is_spike_hist=is_spike_arr, niv_hist=niv_arr,
            weather_hist=weather_hist,
            daily_mean_lag1d=daily_mean_lag1d,
            daily_mean_lag7d=daily_mean_lag7d,
            daily_stats=daily_stats,
            gen_hist=gen_hist,
            wind_pct_forecast=wind_pct_forecast,
            solar_forecast=target_weather["solar_wm2"],
        )
        shape_rows.append(row)

    X_shape = np.array([[r.get(c, 0.0) for c in shape_feat_cols] for r in shape_rows])
    deviations = shape_q50.predict(X_shape)

    for h in range(48):
        dt  = base_dt + pd.Timedelta(minutes=30 * h)
        dev = float(deviations[h])
        records.append({
            "settlement_datetime": dt,
            "settlement_date":     str(target_date),
            "settlement_period":   h + 1,
            "ssp_predicted":       round(pred_l50 + dev, 2),
            "ssp_q10":             round(pred_l10 + dev, 2),
            "ssp_q50":             round(pred_l50 + dev, 2),
            "ssp_q90":             round(pred_l90 + dev, 2),
            "pred_daily_level":    round(pred_l50, 2),
            "pred_deviation":      round(dev, 2),
        })

    result = pd.DataFrame(records)
    result["is_actual"] = False   # populated below when intraday actuals are available

    # ── Intraday post-processing: splice actuals + shape correction ───────
    # Runs only when intraday_prices.csv exists and has data for today.
    #
    # Option 1 — Replace settled SPs with actual prices.
    #   The orange zone on the dashboard shows real prices, not forecasts.
    #
    # Option 2 — Shape correction for unsettled SPs.
    #   Mean forecast error over settled SPs → dampened correction applied to
    #   remaining SPs (α=0.4).  If morning was £10 higher than forecast, the
    #   afternoon shifts up by £4.  Decays to zero correction when no SPs are
    #   settled yet (daily pipeline run) so it never degrades the pure forecast.
    _SHAPE_ALPHA = 0.4   # dampening: don't over-commit to morning signal
    if not _id_today_all.empty:
        try:
            _pc = "ssp" if "ssp" in _id_today_all.columns else "systemSellPrice"
            _actual = _id_today_all[_id_today_all[_pc].notna()].copy()
            _actual_map = dict(zip(
                _actual["settlement_period"].astype(int),
                _actual[_pc].astype(float),
            ))
            if _actual_map:
                _settled = set(_actual_map)
                _unsettled_mask = ~result["settlement_period"].isin(_settled)

                # Option 2: compute correction BEFORE overwriting with actuals
                _settled_rows  = result[result["settlement_period"].isin(_settled)]
                _actual_vals   = [_actual_map[sp] for sp in _settled_rows["settlement_period"]]
                _fc_vals       = _settled_rows["ssp_q50"].tolist()
                _mean_err      = float(np.mean([a - f for a, f in zip(_actual_vals, _fc_vals)]))
                _correction    = _SHAPE_ALPHA * _mean_err
                if _unsettled_mask.any() and abs(_correction) > 0.5:
                    for _col in ("ssp_predicted", "ssp_q10", "ssp_q50", "ssp_q90"):
                        result.loc[_unsettled_mask, _col] = (
                            result.loc[_unsettled_mask, _col] + _correction
                        ).round(2)
                    log.info("Shape correction: mean_err=£%.1f  applied=£%.1f  (%d unsettled SPs)",
                             _mean_err, _correction, int(_unsettled_mask.sum()))

                # Option 1: replace settled SPs with actuals
                for _sp, _av in _actual_map.items():
                    _m = result["settlement_period"] == _sp
                    if _m.any():
                        for _col in ("ssp_predicted", "ssp_q10", "ssp_q50", "ssp_q90"):
                            result.loc[_m, _col] = round(_av, 2)
                        result.loc[_m, "is_actual"] = True
                log.info("Intraday splice: %d SPs replaced with actuals", len(_actual_map))
        except Exception as _e:
            log.warning("Intraday post-processing failed: %s", _e)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)
    log.info("H+1 forecast saved → %s", OUTPUT_FILE)

    archive_dir = OUTPUT_FILE.parent / "forecasts"
    archive_dir.mkdir(exist_ok=True)
    result.to_csv(archive_dir / f"forecast_phase3_{target_date}.csv", index=False)

    # ── H+2: forecast for target_date + 1 (tomorrow) ─────────────────────
    if shape_h2_q50 is not None and shape_h2_feat_cols:
        h2_date   = target_date + timedelta(days=1)
        base_h2   = pd.Timestamp(h2_date)

        # Level for H+2: use lag-2d features (available 2 days before H+2)
        lf_h2 = build_level_features(daily_hist, h2_date, target_weather)
        # WINDFOR for H+2 (tomorrow)
        try:
            log.info("Fetching BMRS WINDFOR for H+2 date %s …", h2_date)
            wind_pct_h2  = get_wind_pct_forecast(h2_date)
            wf_h2_mean   = float(np.nanmean(wind_pct_h2))
            if "wind_pct_daily_mean_lag1d" in level_feat_cols:
                lf_h2["wind_pct_daily_mean_lag1d"] = wf_h2_mean
            log.info("  H+2 WINDFOR daily mean: %.1f%%", wf_h2_mean)
        except Exception as _e:
            wind_pct_h2 = None
            log.warning("H+2 WINDFOR fetch failed: %s", _e)

        # Open-Meteo day+2 solar for H+2 shape rows
        try:
            solar_h2 = _day_weather(weather_df, h2_date)["solar_wm2"]
        except Exception:
            solar_h2 = None

        # H+2 level prediction
        x_h2 = np.array([[lf_h2.get(c, 0.0) for c in level_feat_cols]])
        h2_l10 = min(float(level_q10.predict(x_h2)[0]), float(level_q50.predict(x_h2)[0]))
        h2_l50 = float(level_q50.predict(x_h2)[0])
        h2_l90 = max(float(level_q90.predict(x_h2)[0]), h2_l50)
        log.info("H+2 level: P10=£%.1f  P50=£%.1f  P90=£%.1f", h2_l10, h2_l50, h2_l90)

        # H+2 lag-2d daily stats
        daily_mean_lag2d = float(daily_hist["ssp_daily_mean"].values[-2]) if len(daily_hist) >= 2 else daily_mean_lag1d
        daily_mean_lag7d_h2 = float(daily_hist["ssp_daily_mean"].values[-7]) if len(daily_hist) >= 7 else daily_mean_lag2d
        daily_stats_h2: dict = {
            "ssp_raw_daily_max_lag2d": float(daily_hist["ssp_raw_daily_max"].values[-2]) if len(daily_hist) >= 2 else 0.0,
            "spike_count_lag7d":       float(daily_hist["is_spike_count"].values[-7])    if len(daily_hist) >= 7 else 0.0,
            "spike_count_roll_7d":     float(daily_hist["is_spike_count"].values[-7:].sum()),
            "ssp_daily_roll_mean_7d":  float(daily_hist["ssp_daily_mean"].values[-7:].mean()),
        }

        h2_shape_rows = []
        for h in range(48):
            dt_h2 = base_h2 + pd.Timedelta(minutes=30 * h)
            row_h2 = build_shape_row_h2(
                h_idx=h, dt=dt_h2,
                ssp_hist=ssp_arr, ssp_raw_hist=ssp_raw_arr,
                is_spike_hist=is_spike_arr, niv_hist=niv_arr,
                weather_hist=weather_hist,
                daily_mean_lag2d=daily_mean_lag2d,
                daily_mean_lag7d=daily_mean_lag7d_h2,
                daily_stats_h2=daily_stats_h2,
                gen_hist=gen_hist,
                wind_pct_forecast_h2=wind_pct_h2,
                solar_forecast_h2=solar_h2,
            )
            h2_shape_rows.append(row_h2)

        X_h2   = np.array([[r.get(c, 0.0) for c in shape_h2_feat_cols] for r in h2_shape_rows])
        devs_h2 = shape_h2_q50.predict(X_h2)

        h2_records = []
        for h in range(48):
            dt_h2 = base_h2 + pd.Timedelta(minutes=30 * h)
            dev   = float(devs_h2[h])
            h2_records.append({
                "settlement_datetime": dt_h2,
                "settlement_date":     str(h2_date),
                "settlement_period":   h + 1,
                "ssp_predicted":       round(h2_l50 + dev, 2),
                "ssp_q10":             round(h2_l10 + dev, 2),
                "ssp_q50":             round(h2_l50 + dev, 2),
                "ssp_q90":             round(h2_l90 + dev, 2),
                "pred_daily_level":    round(h2_l50, 2),
                "pred_deviation":      round(dev, 2),
            })

        result_h2 = pd.DataFrame(h2_records)
        result_h2.to_csv(OUTPUT_FILE_H2, index=False)
        result_h2.to_csv(archive_dir / f"forecast_phase3_{h2_date}.csv", index=False)
        log.info("H+2 forecast saved → %s  (daily level P50=£%.1f)", OUTPUT_FILE_H2, h2_l50)
    else:
        result_h2 = pd.DataFrame()
        log.info("H+2 model not available — skipping H+2 forecast")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 3 two-stage day-ahead SSP forecast")
    parser.add_argument("--date",     default=None, help="Target date YYYY-MM-DD")
    parser.add_argument("--features", default=str(FEATURES_FILE))
    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else None
    result = run_forecast(target, features_file=args.features if args.features != str(FEATURES_FILE) else None)
    print(f"\nPhase 3 forecast for {result['settlement_date'].iloc[0]} "
          f"(daily level P50 = £{result['pred_daily_level'].iloc[0]:.1f}/MWh):")
    print(result[["settlement_datetime", "settlement_period",
                  "ssp_q10", "ssp_q50", "ssp_q90"]].to_string(index=False))


if __name__ == "__main__":
    main()

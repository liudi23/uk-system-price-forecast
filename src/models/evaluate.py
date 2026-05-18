"""
Evaluation metrics for electricity price forecasting.

Standard metrics from Weron (2014) and the epftoolbox benchmark:
  MAE   — Mean Absolute Error (£/MWh), scale-dependent
  RMSE  — Root Mean Squared Error (£/MWh), penalises large errors more
  MAPE  — Mean Absolute Percentage Error; undefined at zero so rows where
           |y_true| < 1 are excluded from the MAPE calculation
  sMAPE — Symmetric MAPE; bounded [0, 200%], handles near-zero prices
           better than MAPE

Usage:
    from evaluate import metrics, print_report
    m = metrics(y_true, y_pred)
    print_report("LightGBM", m)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, min_abs: float = 1.0) -> float:
    mask = np.abs(y_true) >= min_abs
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    return float(np.mean(vals) * 100)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(mae(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAPE": round(mape(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
        "n": int(len(y_true)),
    }


def print_report(name: str, m: Dict[str, float]) -> None:
    print(f"\n{'─'*44}")
    print(f"  {name}")
    print(f"{'─'*44}")
    print(f"  MAE   : {m['MAE']:>8.3f} £/MWh")
    print(f"  RMSE  : {m['RMSE']:>8.3f} £/MWh")
    print(f"  MAPE  : {m['MAPE']:>8.2f} %")
    print(f"  sMAPE : {m['sMAPE']:>8.2f} %")
    print(f"  n     : {m['n']:>8d} periods")
    print(f"{'─'*44}")


def save_metrics(all_metrics: Dict[str, Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved → {path}")


def compare_table(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, m in all_metrics.items():
        rows.append({"Model": name, **{k: v for k, v in m.items() if k != "n"}})
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("MAE")

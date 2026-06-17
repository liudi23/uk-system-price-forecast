"""
Train a day-level spike probability classifier for use as a PI-widening gate.

Target
------
spike_day = 1 if any SP on that day has ssp_raw > £150 else 0.

Features (all ≥ 48 SP lagged — guaranteed leakage-free)
---------------------------------------------------------
All 9 features map 1-to-1 to the `lf` dict in forecast_phase3.py so there is
no additional code path at inference: the classifier reads from the same dict
already built by build_level_features().

  ssp_raw_daily_max_lag1d   max raw SSP on D-1 (continuous; key price signal)
  spike_count_lag1d         Tukey-fence (>£353.7) spikes on D-1 (sparse, non-zero
                            only on true extreme days)
  spike_count_roll_7d       7-day rolling Tukey-fence spike count
  wind_pct_daily_mean_lag1d CI wind % on D-1 [TRAIN]; substituted with NG ESO
                            WINDFOR (day-ahead forecast for D+1) at inference
  gas_pct_daily_mean_lag1d  gas % on D-1
  niv_daily_mean_lag1d      net imbalance volume on D-1
  wind_ms_daily_mean_lag1d  wind speed m/s on D-1
  sin_month, cos_month      cyclical month encoding
  is_business_day           0/1 flag for the FORECAST day (D+1)

Train/serve wind skew
---------------------
wind_pct_daily_mean_lag1d is CI D-1 actual wind at training time but WINDFOR
(NG ESO day-ahead wind forecast for D+1) at inference. This is the same pattern
used by the level model. At inference, WINDFOR is a forecast of D+1 wind whereas
training uses D-1 actual wind as a proxy. The skew has the same sign but opposite
direction relative to the level model: on low-wind days, WINDFOR < CI_lag1d
(persistence over-estimates tomorrow's wind given a ramp-down), which means the
classifier sees a lower wind value at inference than it saw during training for
equivalent market conditions. This makes it LESS likely to flag spike risk than
warranted, so the gate is biased toward conservatism (missed spikes rather than
false alarms). Document in eval JSON; revisit when WINDFOR history is available
for backfilling training.

Training window
---------------
2024-01-01 to 2025-06-30. Base rate ≈ 14% (matches WF window), post energy-crisis.
2022 excluded (96.7% spike rate = misleading). 2023 excluded pending regime weighting.

Evaluation
----------
2025-07-01 to 2026-04-30 (WF window, 119 days). Out-of-sample on the same period
used for PI calibration and the corrector backtest.

Outputs
-------
model_assets/spike_classifier_v1.pkl         sklearn Pipeline (predict_proba)
model_assets/spike_classifier_v1_feats.json  ordered feature name list
model_assets/spike_classifier_v1_eval.json   WF window predictions + metrics

Usage
-----
    python src/models/train_spike_classifier.py
"""

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT      = Path(__file__).resolve().parents[2]
FEAT_PATH = ROOT / "data" / "processed" / "features_5yr.csv"
WF_PATH   = ROOT / "model_assets" / "walk_forward_predictions.csv"
PKL_OUT   = ROOT / "model_assets" / "spike_classifier_v1.pkl"
FEATS_OUT = ROOT / "model_assets" / "spike_classifier_v1_feats.json"
EVAL_OUT  = ROOT / "model_assets" / "spike_classifier_v1_eval.json"

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END   = pd.Timestamp("2025-06-30")
EVAL_START  = pd.Timestamp("2025-07-01")
EVAL_END    = pd.Timestamp("2026-04-30")

SPIKE_THRESHOLD = 150.0   # £/MWh — target definition

FEATURE_NAMES = [
    "ssp_raw_daily_max_lag1d",
    "spike_count_lag1d",
    "spike_count_roll_7d",
    "wind_pct_daily_mean_lag1d",
    "gas_pct_daily_mean_lag1d",
    "niv_daily_mean_lag1d",
    "wind_ms_daily_mean_lag1d",
    "sin_month",
    "cos_month",
    "is_business_day",
]


def build_day_features(feat: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SP-level features_5yr.csv to one row per day.
    All aggregations use lag_48 columns (= D-1 same-SP values) so the
    result represents D-1 information for the forecast of day D.
    """
    feat = feat.sort_values(["settlement_date", "settlement_period"])

    daily = feat.groupby("settlement_date").agg(
        # Target: any SP on THIS day with raw SSP > SPIKE_THRESHOLD
        spike_day               = ("ssp_raw",                    lambda x: int((x > SPIKE_THRESHOLD).any())),
        # Feature: max raw SSP on D-1 (from ssp_raw_lag_48 at row (D, sp) = ssp_raw(D-1, sp))
        ssp_raw_daily_max_lag1d = ("ssp_raw_lag_48",             "max"),
        # Feature: Tukey-fence spikes on D-1 (is_spike_lag_48 uses raw SSP > fence)
        spike_count_lag1d       = ("is_spike_lag_48",            "sum"),
        # Weather/energy mix on D-1 (lag_48 columns)
        wind_pct_daily_mean_lag1d = ("wind_pct_lag_48",          "mean"),
        gas_pct_daily_mean_lag1d  = ("gas_pct_lag_48",           "mean"),
        niv_daily_mean_lag1d      = ("net_imbalance_volume_lag_48", "mean"),
        wind_ms_daily_mean_lag1d  = ("wind_ms_lag_48",           "mean"),
        # Calendar (for target day D — first SP of the day, all SPs identical)
        sin_month               = ("sin_month",                   "first"),
        cos_month               = ("cos_month",                   "first"),
        is_business_day         = ("is_business_day",             "first"),
    ).reset_index()

    daily["settlement_date"] = pd.to_datetime(daily["settlement_date"])

    # spike_count_roll_7d: 7-day rolling sum of daily spike_count_lag1d
    daily = daily.sort_values("settlement_date").reset_index(drop=True)
    daily["spike_count_roll_7d"] = (
        daily["spike_count_lag1d"]
        .rolling(7, min_periods=1)
        .sum()
        .shift(0)   # shift(0): rolling over D-6..D (includes current-day D-1 count, which is fine)
    )

    return daily


def main() -> None:
    if not FEAT_PATH.exists():
        print(f"ERROR: {FEAT_PATH} not found", file=sys.stderr)
        sys.exit(1)

    # ── Load and aggregate features ───────────────────────────────────────────
    print("Loading features_5yr.csv …")
    need_cols = [
        "settlement_date", "settlement_period",
        "ssp_raw", "ssp_raw_lag_48", "is_spike_lag_48",
        "wind_pct_lag_48", "gas_pct_lag_48",
        "net_imbalance_volume_lag_48", "wind_ms_lag_48",
        "sin_month", "cos_month", "is_business_day",
    ]
    feat = pd.read_csv(FEAT_PATH, usecols=need_cols)
    feat["settlement_date"] = pd.to_datetime(feat["settlement_date"])
    print(f"  {len(feat)} SP rows, {feat['settlement_date'].nunique()} days")

    daily = build_day_features(feat)
    print(f"  Aggregated to {len(daily)} day-level rows")

    # ── Train / eval split ────────────────────────────────────────────────────
    train = daily[(daily["settlement_date"] >= TRAIN_START) &
                  (daily["settlement_date"] <= TRAIN_END)].copy()
    ev    = daily[(daily["settlement_date"] >= EVAL_START) &
                  (daily["settlement_date"] <= EVAL_END)].copy()

    # Restrict eval to dates present in WF backtest window (119 days from WF CSV)
    wf_dates = None
    if WF_PATH.exists():
        wf_dates = set(pd.read_csv(WF_PATH, usecols=["settlement_date"])["settlement_date"].unique())
        ev = ev[ev["settlement_date"].dt.strftime("%Y-%m-%d").isin(wf_dates)].copy()

    print(f"\nTraining: {TRAIN_START.date()} – {TRAIN_END.date()}  "
          f"({len(train)} days, {train['spike_day'].mean():.1%} spike rate)")
    print(f"Eval WF:  {EVAL_START.date()} – {EVAL_END.date()}  "
          f"({len(ev)} days, {ev['spike_day'].mean():.1%} spike rate)")

    # ── Drop rows with too many NaNs in features ──────────────────────────────
    train = train.dropna(subset=FEATURE_NAMES)
    ev_clean = ev.dropna(subset=FEATURE_NAMES)
    print(f"\nAfter NaN drop: train={len(train)}, eval={len(ev_clean)}")

    X_train = train[FEATURE_NAMES].values
    y_train = train["spike_day"].values
    X_eval  = ev_clean[FEATURE_NAMES].values
    y_eval  = ev_clean["spike_day"].values

    # ── Train model ───────────────────────────────────────────────────────────
    print("\nTraining CalibratedClassifierCV(LogisticRegression, cv=5, method=isotonic) …")
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, class_weight="balanced",
                                      max_iter=1000, random_state=42,
                                      solver="lbfgs")),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CalibratedClassifierCV(base_pipe, cv=5, method="isotonic")
        model.fit(X_train, y_train)
    print("  Training complete.")

    # ── Evaluate on WF window ─────────────────────────────────────────────────
    p_spike_eval = model.predict_proba(X_eval)[:, 1]

    brier = float(brier_score_loss(y_eval, p_spike_eval))
    ap    = float(average_precision_score(y_eval, p_spike_eval))
    base_rate = float(y_eval.mean())
    brier_baseline = base_rate * (1 - base_rate)   # Brier of constant-rate predictor
    brier_skill = 1 - brier / brier_baseline

    print(f"\nWF evaluation ({len(ev_clean)} days):")
    print(f"  Spike-day base rate : {base_rate:.1%}")
    print(f"  Brier score         : {brier:.4f}  (baseline={brier_baseline:.4f}, skill={brier_skill:.3f})")
    print(f"  Average precision   : {ap:.4f}")
    print(f"\n  Train/serve wind skew: wind_pct_daily_mean_lag1d = CI D-1 actual in training,")
    print(f"  WINDFOR (NG ESO day-ahead forecast) at inference. Skew biases toward conservatism")
    print(f"  (WINDFOR more accurate on ramp-down days → classifier under-flags low-wind risk).")

    taus = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    tau_metrics = {}
    print(f"\n  {'τ':>5}  {'Flagged':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
    for tau in taus:
        flagged = (p_spike_eval >= tau).sum()
        tp = ((p_spike_eval >= tau) & (y_eval == 1)).sum()
        fp = ((p_spike_eval >= tau) & (y_eval == 0)).sum()
        fn = ((p_spike_eval <  tau) & (y_eval == 1)).sum()
        prec   = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1     = 2 * prec * recall / max(prec + recall, 1e-9)
        print(f"  {tau:>5.2f}  {flagged:>8d}  {prec:>10.1%}  {recall:>8.1%}  {f1:>6.3f}")
        tau_metrics[str(tau)] = {
            "n_flagged": int(flagged),
            "precision": round(float(prec), 4),
            "recall":    round(float(recall), 4),
            "f1":        round(float(f1), 4),
        }

    # ── Save per-day predictions for backtest gate ────────────────────────────
    p_spike_all = model.predict_proba(daily[FEATURE_NAMES].fillna(0).values)[:, 1]
    daily["p_spike"] = p_spike_all
    wf_preds = {
        row["settlement_date"].strftime("%Y-%m-%d"): round(float(row["p_spike"]), 6)
        for _, row in daily[
            (daily["settlement_date"] >= EVAL_START) &
            (daily["settlement_date"] <= EVAL_END)
        ].iterrows()
    }

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(model, PKL_OUT)
    print(f"\nSaved model → {PKL_OUT}")

    json.dump(FEATURE_NAMES, open(FEATS_OUT, "w"), indent=2)
    print(f"Saved features → {FEATS_OUT}")

    eval_artifact = {
        "version":           "v1",
        "target":            f"spike_day_{int(SPIKE_THRESHOLD)}",
        "model_type":        "logistic_regression_calibrated_isotonic",
        "training_start":    str(TRAIN_START.date()),
        "training_end":      str(TRAIN_END.date()),
        "eval_start":        str(EVAL_START.date()),
        "eval_end":          str(EVAL_END.date()),
        "n_train_days":      int(len(train)),
        "n_spike_days_train":int(y_train.sum()),
        "train_spike_rate":  round(float(y_train.mean()), 4),
        "n_eval_days":       int(len(ev_clean)),
        "n_spike_days_eval": int(y_eval.sum()),
        "eval_spike_rate":   round(base_rate, 4),
        "brier_score":       round(brier, 6),
        "brier_baseline":    round(brier_baseline, 6),
        "brier_skill":       round(brier_skill, 4),
        "average_precision": round(ap, 4),
        "tau_metrics":       tau_metrics,
        "feature_names":     FEATURE_NAMES,
        "train_serve_wind_skew": (
            "wind_pct_daily_mean_lag1d = CI D-1 actual at training, "
            "WINDFOR (NG ESO D+1 forecast) at inference. "
            "Classifier biased toward conservatism on low-wind days: "
            "WINDFOR more accurate on ramp-downs so inference sees lower wind "
            "than equivalent training observations."
        ),
        "wf_predictions":    wf_preds,
        "created":           str(date.today()),
    }
    json.dump(eval_artifact, open(EVAL_OUT, "w"), indent=2)
    print(f"Saved eval → {EVAL_OUT}")


if __name__ == "__main__":
    main()

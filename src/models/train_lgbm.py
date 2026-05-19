"""
Phase 2 HGBR training: quantile regression + spike classifier.

Key changes vs Phase 1
──────────────────────
* Training target is now `ssp_raw` (actual unclipped price) instead of the
  Tukey-winsorised `ssp`. This lifts the prediction ceiling from £354 to the
  true market maximum (£4 038 in training data), allowing the model to learn
  spike behaviour.

* Three quantile HGBR models are trained (P10 / P50 / P90):
    - P50  = median point forecast (replaces the old single-model forecast)
    - P10  = downside / crash-risk bound
    - P90  = upside / spike-risk bound

* A binary HistGradientBoostingClassifier is trained to predict `is_spike`
  (SSP_raw > Tukey upper fence). Its calibrated probabilities are surfaced in
  the dashboard as a settlement-period spike-risk indicator.

* Tukey fence parameters (lower/upper) are saved to `model_assets/tukey_fence.json`
  so forecast.py can recompute `is_spike` for newly ingested raw rows that
  haven't passed through build_dataset.py.

* Autoregressive lag features (`ssp_lag_1`, `ssp_lag_2`, …) continue to use the
  winsorised `ssp` column so spike contamination does not corrupt the recursive
  feature signal during inference.

Outputs (model_assets/)
──────────────────────
  hgbr_q10.pkl               — quantile-10 HGBR (£/MWh lower bound)
  hgbr_q50.pkl               — quantile-50 HGBR (median point forecast)
  hgbr_q90.pkl               — quantile-90 HGBR (£/MWh upper bound)
  spike_classifier.pkl       — binary HGBR classifier for is_spike
  hgbr_metrics.json          — test-set evaluation metrics
  hgbr_feature_importance.csv
  feature_cols.json
  tukey_fence.json            — {"lower": float, "upper": float}

Usage:
    python src/models/train_lgbm.py
    python src/models/train_lgbm.py --features data/processed/features_5yr.csv
    python src/models/train_lgbm.py --test-days 7 --val-days 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import metrics, print_report, save_metrics

FEATURES_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "features_5yr.csv"
ASSETS_DIR    = Path(__file__).resolve().parents[2] / "model_assets"

TEST_DAYS = 7
VAL_DAYS  = 3

# Columns that must never enter the feature matrix
EXCLUDE_COLS = {
    # Identifiers
    "settlement_date", "settlement_datetime", "settlement_period",
    # Contemporaneous values — unknown at prediction time
    "net_imbalance_volume",
    "abs_imbalance_volume",
    "price_derivation_code",
    "price_derivation_code_P",
    # Target leakage: replacement_price == SSP for all P-code rows
    "replacement_price",
    "sell_price_adjustment",
    "buy_price_adjustment",
    # Raw weather columns (renamed with _raw_ prefix by weather_features.py)
    "_raw_temp_c", "_raw_wind_ms", "_raw_solar_wm2", "_raw_precip_mm",
    # Both target forms — we pick the one we want as y, exclude the other
    "ssp",
    "ssp_raw",
    # is_spike is the classification target; exclude from regression features
    "is_spike",
}

# Quantile model specs
QUANTILES = [0.10, 0.50, 0.90]
QUANTILE_NAMES = {0.10: "q10", 0.50: "q50", 0.90: "q90"}

BASE_PARAMS = {
    "max_iter":          1000,
    "learning_rate":     0.05,
    "max_leaf_nodes":    31,
    "min_samples_leaf":  20,
    "l2_regularization": 0.1,
    "max_bins":          255,
    "early_stopping":    True,
    "n_iter_no_change":  50,
    "random_state":      42,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_usable(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["settlement_datetime"])
    df = df.sort_values("settlement_datetime").reset_index(drop=True)
    lag_cols = [c for c in df.columns if "_lag_" in c or "_roll_" in c or "_diff_" in c]
    return df[df[lag_cols].notna().all(axis=1)].reset_index(drop=True)


def split(df: pd.DataFrame, test_days: int, val_days: int):
    t_max = df["settlement_datetime"].max()
    test_cut = t_max - pd.Timedelta(days=test_days)
    val_cut  = test_cut - pd.Timedelta(days=val_days)

    train = df[df["settlement_datetime"] <= val_cut]
    val   = df[(df["settlement_datetime"] > val_cut) & (df["settlement_datetime"] <= test_cut)]
    test  = df[df["settlement_datetime"] > test_cut]

    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        log.info("%s: %d rows (%s → %s)", name, len(part),
                 part["settlement_datetime"].min().date(),
                 part["settlement_datetime"].max().date())
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def _make_regressor(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile,
        **BASE_PARAMS,
    )


def train_quantile_models(train_df, val_df, feature_cols):
    """Train P10, P50, P90 quantile regressors on ssp_raw target."""
    combined   = pd.concat([train_df, val_df], ignore_index=True)
    X          = combined[feature_cols].values
    y          = combined["ssp_raw"].values
    val_frac   = len(val_df) / len(combined)

    models = {}
    for q in QUANTILES:
        log.info("Training quantile=%.2f model…", q)
        m = _make_regressor(q)
        # Inject val_fraction into params — early stopping on actual val set
        m.set_params(validation_fraction=val_frac)
        m.fit(X, y)
        log.info("  q=%.2f best_iter=%d", q, m.n_iter_)
        models[q] = m
    return models


def train_spike_classifier(train_df, val_df, feature_cols):
    """Binary classifier for is_spike (SSP_raw > Tukey upper fence)."""
    combined  = pd.concat([train_df, val_df], ignore_index=True)
    X         = combined[feature_cols].values
    y         = combined["is_spike"].astype(int).values
    val_frac  = len(val_df) / len(combined)

    # class_weight='balanced' adjusts for the ~2.4% spike rate
    clf = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=val_frac,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)
    log.info("Spike classifier best_iter=%d", clf.n_iter_)
    return clf


def compute_feature_importance(model, X_val, y_val, feature_cols, n_repeats=10):
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    return pd.DataFrame({
        "feature":          feature_cols,
        "importance_mean":  result.importances_mean,
        "importance_std":   result.importances_std,
    }).sort_values("importance_mean", ascending=False)


def save_predictions(test_df, preds_dict, assets_dir):
    """Save test-set predictions for all quantiles."""
    pred_path = assets_dir / "test_predictions.csv"
    out = test_df[["settlement_datetime", "settlement_date", "settlement_period",
                   "ssp", "ssp_raw"]].copy()
    out = out.rename(columns={"ssp_raw": "ssp_actual", "ssp": "ssp_winsorised"})
    for name, y_pred in preds_dict.items():
        out[f"ssp_{name}"] = y_pred
    # Backwards-compatible alias
    out["ssp_predicted"] = out["ssp_q50"]
    out["error"]         = out["ssp_predicted"] - out["ssp_actual"]
    out["abs_error"]     = out["error"].abs()
    out.to_csv(pred_path, index=False)
    log.info("Predictions saved → %s", pred_path)


def save_outputs(models_dict, clf, fi, test_metrics, assets_dir, feature_cols, tukey_fence):
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Quantile models
    for q, model in models_dict.items():
        name = QUANTILE_NAMES[q]
        path = assets_dir / f"hgbr_{name}.pkl"
        joblib.dump(model, path)
        log.info("Saved %s → %s", name, path)

    # Backwards-compatible alias: hgbr_model.pkl → q50
    joblib.dump(models_dict[0.50], assets_dir / "hgbr_model.pkl")

    # Spike classifier
    joblib.dump(clf, assets_dir / "spike_classifier.pkl")
    log.info("Spike classifier saved → %s", assets_dir / "spike_classifier.pkl")

    # Feature importance
    fi_path = assets_dir / "hgbr_feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    # Feature column list
    with open(assets_dir / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)

    # Tukey fence for inference-time is_spike computation
    with open(assets_dir / "tukey_fence.json", "w") as f:
        json.dump(tukey_fence, f)
    log.info("Tukey fence saved → %s", assets_dir / "tukey_fence.json")

    save_metrics({"HGBR_q50": test_metrics}, assets_dir / "hgbr_metrics.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=TEST_DAYS)
    parser.add_argument("--val-days",  type=int, default=VAL_DAYS)
    parser.add_argument("--features",  default=str(FEATURES_FILE))
    args = parser.parse_args()

    df = load_usable(Path(args.features))
    train_df, val_df, test_df = split(df, args.test_days, args.val_days)
    feature_cols = get_feature_cols(df)
    log.info("Features: %d", len(feature_cols))

    # Recover Tukey fence from the processed data (fence was applied in build_dataset.py)
    q1  = df["ssp"].quantile(0.25)   # winsorised == raw below upper fence
    q3  = df["ssp"].quantile(0.75)
    iqr = q3 - q1
    tukey_fence = {"lower": round(q1 - 3 * iqr, 4), "upper": round(q3 + 3 * iqr, 4)}
    log.info("Tukey fence: lower=%.2f  upper=%.2f", tukey_fence["lower"], tukey_fence["upper"])

    # ── Quantile models ───────────────────────────────────────────────────────
    log.info("Training quantile regressors (P10 / P50 / P90) on ssp_raw…")
    models = train_quantile_models(train_df, val_df, feature_cols)

    # ── Spike classifier ──────────────────────────────────────────────────────
    log.info("Training spike classifier…")
    clf = train_spike_classifier(train_df, val_df, feature_cols)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    X_test  = test_df[feature_cols].values
    y_true  = test_df["ssp_raw"].values

    preds = {}
    all_metrics = {}
    for q, model in models.items():
        name   = QUANTILE_NAMES[q]
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        m = metrics(y_true, y_pred)
        all_metrics[f"HGBR_{name}"] = m
        print_report(f"HGBR quantile={q:.2f} (on ssp_raw)", m)

    # Spike classification report
    spike_true = test_df["is_spike"].astype(int).values
    spike_prob = clf.predict_proba(X_test)[:, 1]
    spike_pred = (spike_prob >= 0.5).astype(int)
    n_actual   = spike_true.sum()
    n_caught   = (spike_pred & spike_true).sum()
    log.info("Spike classifier: %d actual spikes, %d caught (recall=%.1f%%)",
             n_actual, n_caught, 100 * n_caught / max(n_actual, 1))

    # ── Feature importance on val set (q50 model) ────────────────────────────
    log.info("Computing permutation importance on val set…")
    fi = compute_feature_importance(
        models[0.50],
        val_df[feature_cols].values, val_df["ssp_raw"].values,
        feature_cols,
    )

    # ── Save everything ───────────────────────────────────────────────────────
    save_predictions(test_df, preds, ASSETS_DIR)
    save_outputs(models, clf, fi, all_metrics.get("HGBR_q50", {}),
                 ASSETS_DIR, feature_cols, tukey_fence)

    print("\nTop-10 features by permutation importance (MAE reduction on ssp_raw):")
    print(fi.head(10)[["feature", "importance_mean", "importance_std"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""
Histogram-based gradient boosting model for UK system price (SSP) forecasting.

Uses sklearn's HistGradientBoostingRegressor (HGBR), which implements the
same histogram-based algorithm as LightGBM and requires no external libs.

To swap to LightGBM once `libomp` is installed (brew install libomp):
    Replace the HGBR import and MODEL_PARAMS block with:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50)])

Setup:
  Target     : ssp (Tukey-winsorised price from build_dataset.py)
  Features   : 38 calendar, lag, rolling, imbalance and price-formation columns
  Split      : train / val (early-stopping) / test — time-ordered, no shuffle
  Loss       : MAE (robust to remaining price spikes)

Outputs (model_assets/):
  hgbr_model.pkl                — serialised model (joblib)
  hgbr_metrics.json             — test-set evaluation metrics
  hgbr_feature_importance.csv   — permutation importance on val set

Usage:
    python src/models/train_lgbm.py
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import metrics, print_report, save_metrics

FEATURES_FILE = Path(__file__).resolve().parents[2] / "data" / "processed" / "features.csv"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "model_assets"

TEST_DAYS = 7
VAL_DAYS = 3

EXCLUDE_COLS = {
    # Identifiers
    "settlement_date", "settlement_datetime", "settlement_period",
    # Raw / derived target
    "ssp_raw", "is_spike",
    # Contemporaneous values — unknown at prediction time for a future period
    "net_imbalance_volume",       # use lagged NIV features instead
    "abs_imbalance_volume",       # = |net_imbalance_volume|, same period leakage
    "price_derivation_code",      # string; determined after settlement
    "price_derivation_code_P",    # binary version; same contemporaneous leakage
    # Critical target leakage:
    # replacement_price == SSP for all P-code rows (API) and all N-code nulls
    # (imputed with SSP in build_dataset.py). Correlation with target = 0.9999.
    "replacement_price",
    # Near-zero throughout, low signal
    "sell_price_adjustment",
    "buy_price_adjustment",
    # Raw contemporaneous weather columns (renamed with _raw_ prefix by weather_features.py)
    "_raw_temp_c", "_raw_wind_ms", "_raw_solar_wm2", "_raw_precip_mm",
}

MODEL_PARAMS = {
    "loss": "absolute_error",   # MAE — robust to remaining price spikes
    "max_iter": 1000,
    "learning_rate": 0.05,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "max_bins": 255,
    "early_stopping": True,
    "validation_fraction": None,  # we pass our own val set via warm_start workaround
    "n_iter_no_change": 50,
    "random_state": 42,
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
    val_cut = test_cut - pd.Timedelta(days=val_days)

    train = df[df["settlement_datetime"] <= val_cut]
    val = df[(df["settlement_datetime"] > val_cut) & (df["settlement_datetime"] <= test_cut)]
    test = df[df["settlement_datetime"] > test_cut]

    for name, part in [("Train", train), ("Val", val), ("Test", test)]:
        log.info("%s: %d rows (%s → %s)", name, len(part),
                 part["settlement_datetime"].min().date(),
                 part["settlement_datetime"].max().date())
    return train, val, test


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_COLS and c != "ssp"]


def train_model(train_df, val_df, feature_cols):
    # HGBR's built-in early stopping works on a fraction of training data.
    # We concatenate train+val so the model sees early-stopping signals on val.
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X = combined[feature_cols].values
    y = combined["ssp"].values

    # Tell HGBR to use the last len(val_df) rows as the validation set
    val_fraction = len(val_df) / len(combined)
    params = {**MODEL_PARAMS, "validation_fraction": val_fraction}

    model = HistGradientBoostingRegressor(**params)
    model.fit(X, y)
    log.info("Best iteration: %d", model.n_iter_)
    return model


def compute_feature_importance(model, X_val, y_val, feature_cols, n_repeats=10):
    result = permutation_importance(
        model, X_val, y_val, n_repeats=n_repeats,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return fi


def evaluate_dayahead_recursive(model, df, test_df, feature_cols):
    """
    Honest day-ahead evaluation: simulate the recursive multi-step forecast for
    each test day using only information genuinely available at forecast time.

    Batch prediction on pre-computed features is misleading because ssp_lag_1,
    ssp_lag_2, ssp_roll_*_6, ssp_diff_1_48, and niv_lag_1 all reference prices
    or volumes from within the forecast window — information unavailable when
    the day-ahead forecast is generated. This function overrides those features
    with recursively computed values (running prediction buffer), matching
    exactly what forecast.py does at deployment time.
    """
    HISTORY = 400

    leaky_names = {
        "ssp_lag_1", "ssp_lag_2",
        "ssp_roll_mean_6", "ssp_roll_std_6", "ssp_roll_min_6", "ssp_roll_max_6",
        "ssp_diff_1_48",
        "net_imbalance_volume_lag_1",
        "net_imbalance_volume_roll_mean_6", "net_imbalance_volume_roll_std_6",
    }
    col_idx  = {f: i for i, f in enumerate(feature_cols)}
    leaky_idx = {f: col_idx[f] for f in leaky_names if f in col_idx}
    lag48_idx = col_idx.get("ssp_lag_48")

    test_dates = sorted(test_df["settlement_datetime"].dt.date.unique())
    y_true_all, y_pred_all = [], []

    for target_date in test_dates:
        hist = df[df["settlement_datetime"].dt.date < target_date].tail(HISTORY).reset_index(drop=True)
        if len(hist) < HISTORY:
            log.warning("Skipping %s — only %d history rows", target_date, len(hist))
            continue

        ssp_buf     = list(hist["ssp"].values)
        niv_actual  = hist["net_imbalance_volume"].values
        niv_virtual = np.concatenate([niv_actual, niv_actual[-48:]])

        day_rows = (
            test_df[test_df["settlement_datetime"].dt.date == target_date]
            .sort_values("settlement_datetime")
        )

        for h_idx, (_, row_s) in enumerate(day_rows.iterrows()):
            idx = HISTORY + h_idx

            x = np.array([row_s.get(f, 0.0) for f in feature_cols], dtype=float)

            ssp_arr = np.array(ssp_buf)

            if "ssp_lag_1" in leaky_idx:
                x[leaky_idx["ssp_lag_1"]] = float(ssp_arr[-1])
            if "ssp_lag_2" in leaky_idx:
                x[leaky_idx["ssp_lag_2"]] = float(ssp_arr[-2])

            win6 = ssp_arr[-6:]
            for k, fn in [
                ("mean", np.mean),
                ("std",  lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0),
                ("min",  np.min),
                ("max",  np.max),
            ]:
                key = f"ssp_roll_{k}_6"
                if key in leaky_idx:
                    x[leaky_idx[key]] = float(fn(win6))

            if "ssp_diff_1_48" in leaky_idx:
                lag48 = x[lag48_idx] if lag48_idx is not None else float(ssp_arr[-48])
                x[leaky_idx["ssp_diff_1_48"]] = float(ssp_arr[-1]) - lag48

            niv6 = niv_virtual[idx - 6 : idx]
            if "net_imbalance_volume_lag_1" in leaky_idx:
                x[leaky_idx["net_imbalance_volume_lag_1"]] = float(niv_virtual[idx - 1])
            if "net_imbalance_volume_roll_mean_6" in leaky_idx:
                x[leaky_idx["net_imbalance_volume_roll_mean_6"]] = float(niv6.mean())
            if "net_imbalance_volume_roll_std_6" in leaky_idx:
                x[leaky_idx["net_imbalance_volume_roll_std_6"]] = (
                    float(np.std(niv6, ddof=1)) if len(niv6) > 1 else 0.0
                )

            pred = float(model.predict(x.reshape(1, -1))[0])
            ssp_buf.append(pred)
            y_pred_all.append(pred)
            y_true_all.append(float(row_s["ssp"]))

    return np.array(y_true_all), np.array(y_pred_all)


def save_predictions(test_df, y_pred, assets_dir):
    pred_path = assets_dir / "test_predictions.csv"
    out = test_df[["settlement_datetime", "settlement_date", "settlement_period", "ssp"]].copy()
    out = out.rename(columns={"ssp": "ssp_actual"})
    out["ssp_predicted"] = y_pred
    out["error"] = out["ssp_predicted"] - out["ssp_actual"]
    out["abs_error"] = out["error"].abs()
    out.to_csv(pred_path, index=False)
    log.info("Predictions saved → %s", pred_path)


def save_outputs(model, fi, test_metrics, assets_dir, feature_cols):
    import json
    assets_dir.mkdir(parents=True, exist_ok=True)

    model_path = assets_dir / "hgbr_model.pkl"
    joblib.dump(model, model_path)
    log.info("Model saved → %s", model_path)

    fi_path = assets_dir / "hgbr_feature_importance.csv"
    fi.to_csv(fi_path, index=False)
    log.info("Feature importance saved → %s", fi_path)

    fc_path = assets_dir / "feature_cols.json"
    with open(fc_path, "w") as f:
        json.dump(feature_cols, f)
    log.info("Feature columns saved → %s", fc_path)

    save_metrics({"HGBR": test_metrics}, assets_dir / "hgbr_metrics.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=TEST_DAYS)
    parser.add_argument("--val-days", type=int, default=VAL_DAYS)
    parser.add_argument("--features", default=str(FEATURES_FILE), help="Path to features CSV")
    args = parser.parse_args()
    FEATURES_FILE_USE = Path(args.features)

    df = load_usable(FEATURES_FILE_USE)
    train_df, val_df, test_df = split(df, args.test_days, args.val_days)
    feature_cols = get_feature_cols(df)
    log.info("Features: %d", len(feature_cols))

    log.info("Training HGBR…")
    model = train_model(train_df, val_df, feature_cols)

    # ── Recursive day-ahead evaluation (honest) ───────────────────────────────
    # Batch predict(test_df) is misleading: ssp_lag_1 uses actual within-day
    # prices (future info unavailable at forecast time). The recursive eval
    # overrides short-lag features with running model predictions, matching
    # exactly what forecast.py does at deployment.
    log.info("Running recursive day-ahead evaluation on test set…")
    y_true, y_pred = evaluate_dayahead_recursive(model, df, test_df, feature_cols)

    m = metrics(y_true, y_pred)
    print_report("HistGradientBoosting (HGBR) — recursive day-ahead (honest)", m)

    # Log the inflated batch metric so the gap is visible
    y_pred_batch = model.predict(test_df[feature_cols].values)
    m_batch = metrics(test_df["ssp"].values, y_pred_batch)
    log.info(
        "Reference batch eval (leaky): MAE=%.2f  |  "
        "Honest recursive MAE=%.2f  |  gap=%.2f (ssp_lag_1 contamination)",
        m_batch["MAE"], m["MAE"], m["MAE"] - m_batch["MAE"],
    )

    save_metrics({"HGBR": m}, ASSETS_DIR / "hgbr_metrics.json")

    log.info("Computing permutation importance on val set…")
    fi = compute_feature_importance(
        model, val_df[feature_cols].values, val_df["ssp"].values, feature_cols
    )

    save_outputs(model, fi, m, ASSETS_DIR, feature_cols)
    save_predictions(test_df, y_pred, ASSETS_DIR)

    print("\nTop-10 features by permutation importance (MAE reduction):")
    print(fi.head(10)[["feature", "importance_mean", "importance_std"]].to_string(index=False))


if __name__ == "__main__":
    main()

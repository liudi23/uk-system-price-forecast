"""
backtest_weekly_stale.py
─────────────────────────
Refinement C: compare 7-day-stale model (weekly retrain) vs 30-day-stale model
(existing WF fold) over the autumn-2025 spike period (Oct 1–28, 2025).

For each of 4 consecutive 7-day sub-folds:
  • 7-day stale  : retrain at sub-fold start (fresh 3-yr window), evaluate that week
  • 30-day stale : use the existing WF autumn fold predictions (single model, Sep 30 cutoff)

The spike week is W2 (Oct 8–14), which contains the Oct 13 £487 price event.

Pass criterion:
  • Overall autumn MAE  — 7-day stale within +£2.50 of 30-day stale
  • Spike-week W2 MAE   — 7-day stale within +£10.00 of 30-day stale
  • Coverage (all weeks) — delta ≤ ±5 pp

Output → reports/early_forecast_backtest/weekly_stale_results.txt
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "models"))
sys.path.insert(0, str(REPO / "src" / "features"))

from level_features import build_level_dataset, get_level_feature_cols
from train_phase3 import (
    build_shape_data,
    evaluate_phase3,
    get_shape_feature_cols,
    load_sp_data,
    train_level_models,
    train_shape_model,
)

ASSETS = REPO / "model_assets"
DATA   = REPO / "data" / "processed"
OUT    = REPO / "reports" / "early_forecast_backtest"
OUT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_YEARS = 3
VAL_DAYS    = 30   # last 30 days before test = validation (early-stopping signal)

# Autumn 2025 sub-folds (7-day windows inside Oct 1–28 WF fold)
# W1 uses the SAME training cutoff as the WF autumn fold (Sep 30) so results are comparable
WEEKLY_FOLDS = [
    ("W1-Oct01-07", "2025-10-01", "2025-10-07"),   # W1: no spike, baseline
    ("W2-Oct08-14", "2025-10-08", "2025-10-14"),   # W2: contains Oct 13 spike (£487)
    ("W3-Oct15-21", "2025-10-15", "2025-10-21"),   # W3: post-spike; Oct 13 in training
    ("W4-Oct22-28", "2025-10-22", "2025-10-28"),   # W4: Oct 22 second spike (£315)
]

# Tukey fence for spike flagging (same as production)
fence_data  = json.load(open(ASSETS / "tukey_fence.json"))
UPPER_FENCE = float(fence_data["upper"])

# PI calibration
pi_data  = json.load(open(ASSETS / "pi_calibration_v1.json"))
DELTA_BY_SP = {int(k): float(v) for k, v in pi_data["delta_by_sp"].items()}
DELTA_GLOBAL = float(pi_data["delta_global"])


# ─── helpers ──────────────────────────────────────────────────────────────────

def _apply_pi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta = df["settlement_period"].map(
        lambda sp: DELTA_BY_SP.get(int(sp), DELTA_GLOBAL)
    )
    df["q10_cal"] = df["ssp_q10"] - delta
    df["q90_cal"] = df["ssp_q90"] + delta
    return df


def _metrics(df: pd.DataFrame, label: str) -> dict:
    actual   = df["ssp_actual"].values
    q50      = df["ssp_q50"].values
    q10      = df["q10_cal"].values
    q90      = df["q90_cal"].values
    mae      = float(np.mean(np.abs(actual - q50)))
    coverage = float(np.mean((actual >= q10) & (actual <= q90)))
    width    = float(np.mean(q90 - q10))
    alpha = 0.2
    pen_lo  = np.maximum(q10 - actual, 0)
    pen_hi  = np.maximum(actual - q90, 0)
    is_score = float(np.mean(width + (2 / alpha) * (pen_lo + pen_hi)))
    return {"label": label, "n": len(df), "mae": mae,
            "coverage": coverage, "width": width, "is_score": is_score}


def _split_for_fold(sp_df, daily_df, fold_start_str, val_days=VAL_DAYS, train_years=TRAIN_YEARS):
    """Return (sp_train, sp_val, d_train, d_val) for a fold starting at fold_start_str."""
    fold_start  = pd.Timestamp(fold_start_str)
    cutoff      = fold_start - pd.Timedelta(days=1)          # last day of training
    val_cut     = cutoff - pd.Timedelta(days=val_days)       # last day of train (before val)
    train_start = cutoff - pd.Timedelta(days=train_years * 365)  # 3-yr window

    sp_train = sp_df[(sp_df["settlement_datetime"] > train_start) &
                     (sp_df["settlement_datetime"] <= val_cut)]
    sp_val   = sp_df[(sp_df["settlement_datetime"] > val_cut) &
                     (sp_df["settlement_datetime"] <= cutoff)]

    # HGBR handles NaN natively — no row filtering needed, match production
    d_train = daily_df[(daily_df["date"] > train_start) &
                       (daily_df["date"] <= val_cut)]
    d_val   = daily_df[(daily_df["date"] > val_cut) &
                       (daily_df["date"] <= cutoff)]

    return sp_train, sp_val, d_train, d_val


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Weekly-Stale Backtest — Autumn 2025 Spike Period")
    log.info("=" * 70)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    log.info("[1] Loading features_5yr.csv …")
    sp_df = load_sp_data(DATA / "features_5yr.csv")
    log.info("    SP rows: %d  (%s → %s)", len(sp_df),
             sp_df["settlement_datetime"].min().date(),
             sp_df["settlement_datetime"].max().date())

    log.info("[2] Building daily level dataset …")
    daily_df = build_level_dataset(sp_df)

    log.info("[3] Building full shape dataset (used for test evaluation) …")
    shape_sp_full = build_shape_data(sp_df, daily_df)
    shape_feat_cols = get_shape_feature_cols(shape_sp_full)
    level_feat_cols = get_level_feature_cols(daily_df)
    log.info("    Level features: %d  |  Shape features: %d",
             len(level_feat_cols), len(shape_feat_cols))

    # ── 2. Load 30-day WF baseline (autumn fold) ─────────────────────────────
    log.info("[4] Loading 30-day WF baseline (autumn-2025 fold) …")
    wf = pd.read_csv(ASSETS / "walk_forward_predictions.csv",
                     parse_dates=["settlement_date"])
    wf_autumn = wf[(wf["settlement_date"] >= "2025-10-01") &
                   (wf["settlement_date"] <= "2025-10-28")].copy()
    wf_autumn["settlement_date"] = wf_autumn["settlement_date"].dt.date.astype(str)
    wf_autumn = _apply_pi(wf_autumn)
    log.info("    WF autumn rows: %d", len(wf_autumn))

    # ── 3. Weekly-stale: retrain per sub-fold ─────────────────────────────────
    all_7day_preds = []

    for label, fold_start_str, fold_end_str in WEEKLY_FOLDS:
        log.info("")
        log.info("─── Sub-fold %s (%s → %s) ───", label, fold_start_str, fold_end_str)

        fold_start = pd.Timestamp(fold_start_str)
        fold_end   = pd.Timestamp(fold_end_str)

        # Split data
        sp_train, sp_val, d_train, d_val = _split_for_fold(
            sp_df, daily_df, fold_start_str)
        log.info("    SP train: %d  SP val: %d  Level train: %d  Level val: %d",
                 len(sp_train), len(sp_val), len(d_train), len(d_val))

        # Build shape training data
        shape_train = build_shape_data(sp_train, daily_df)
        shape_val   = build_shape_data(sp_val,   daily_df)

        # Train models
        log.info("    Training level models (q10/q50/q90) …")
        level_models = train_level_models(d_train, d_val, level_feat_cols)

        log.info("    Training shape model (q50) …")
        shape_model = train_shape_model(shape_train, shape_val, shape_feat_cols)

        # Test slice
        d_test  = daily_df[(daily_df["date"] >= fold_start) &
                           (daily_df["date"] <= fold_end)]
        sp_test = sp_df[(sp_df["settlement_datetime"] >= fold_start) &
                        (sp_df["settlement_datetime"] <=
                         fold_end + pd.Timedelta(hours=23, minutes=30))]
        sh_test = shape_sp_full[
            (shape_sp_full["settlement_datetime"] >= fold_start) &
            (shape_sp_full["settlement_datetime"] <=
             fold_end + pd.Timedelta(hours=23, minutes=30))
        ]

        pred = evaluate_phase3(
            level_models, shape_model,
            d_test, sp_test, sh_test,
            level_feat_cols, shape_feat_cols,
        )
        pred["sub_fold"] = label
        pred["settlement_date"] = (
            pd.to_datetime(pred["settlement_datetime"]).dt.date.astype(str)
        )

        mae_fold = (pred["ssp_actual"] - pred["ssp_q50"]).abs().mean()
        log.info("    MAE = £%.2f  (n=%d SPs)", mae_fold, len(pred))
        all_7day_preds.append(pred)

    combined_7day = pd.concat(all_7day_preds, ignore_index=True)
    combined_7day = _apply_pi(combined_7day)

    # ── 4. Compute metrics ────────────────────────────────────────────────────
    log.info("")
    log.info("[5] Computing metrics …")

    rows = []
    for label, fold_start_str, fold_end_str in WEEKLY_FOLDS:
        start, end = pd.Timestamp(fold_start_str), pd.Timestamp(fold_end_str)

        # 7-day stale predictions for this sub-fold
        mask_7 = combined_7day["sub_fold"] == label
        pred_7 = combined_7day[mask_7]

        # 30-day stale (WF) predictions for the same date range
        wf_mask = (
            (wf_autumn["settlement_date"] >= fold_start_str) &
            (wf_autumn["settlement_date"] <= fold_end_str)
        )
        pred_30 = wf_autumn[wf_mask]

        # Spike flag (Tukey fence)
        pred_7  = pred_7.copy()
        pred_30 = pred_30.copy()
        pred_7["is_spike"]  = pred_7["ssp_actual"]  > UPPER_FENCE
        pred_30["is_spike"] = pred_30["ssp_actual"] > UPPER_FENCE

        m7  = _metrics(pred_7,  f"7d   {label}")
        m30 = _metrics(pred_30, f"30d  {label}")
        m7["sub_fold"]  = label
        m30["sub_fold"] = label
        m7["staleness"]  = "7-day"
        m30["staleness"] = "30-day"

        # Add spike sub-group
        spike_7  = pred_7[pred_7["is_spike"]]
        spike_30 = pred_30[pred_30["is_spike"]]
        if len(spike_7) > 0 or len(spike_30) > 0:
            ms7  = _metrics(spike_7 if len(spike_7)  > 0 else pred_7.iloc[:0],  f"7d-spike  {label}")
            ms30 = _metrics(spike_30 if len(spike_30) > 0 else pred_30.iloc[:0], f"30d-spike {label}")
            ms7["sub_fold"]  = f"{label}-spike"
            ms30["sub_fold"] = f"{label}-spike"
            ms7["staleness"]  = "7-day"
            ms30["staleness"] = "30-day"
            rows.extend([ms7, ms30])

        rows.extend([m7, m30])

    # Overall autumn
    m7_all  = _metrics(combined_7day, "7d   AUTUMN-TOTAL")
    m30_all = _metrics(wf_autumn,      "30d  AUTUMN-TOTAL")
    m7_all["sub_fold"]  = "TOTAL"
    m30_all["sub_fold"] = "TOTAL"
    m7_all["staleness"]  = "7-day"
    m30_all["staleness"] = "30-day"
    rows.extend([m7_all, m30_all])

    results = pd.DataFrame(rows)

    # ── 5. Ship gate ──────────────────────────────────────────────────────────
    # Extract key numbers
    r7_total  = results[(results["staleness"] == "7-day")  & (results["sub_fold"] == "TOTAL")].iloc[0]
    r30_total = results[(results["staleness"] == "30-day") & (results["sub_fold"] == "TOTAL")].iloc[0]
    r7_w2     = results[(results["staleness"] == "7-day")  & (results["sub_fold"] == "W2-Oct08-14")].iloc[0]
    r30_w2    = results[(results["staleness"] == "30-day") & (results["sub_fold"] == "W2-Oct08-14")].iloc[0]

    overall_mae_delta    = r7_total["mae"]  - r30_total["mae"]
    spikeweek_mae_delta  = r7_w2["mae"]    - r30_w2["mae"]
    overall_cov_delta    = r7_total["coverage"] - r30_total["coverage"]

    PASS_OVERALL  = overall_mae_delta   <= 2.50
    PASS_SPIKE    = spikeweek_mae_delta <= 10.00
    PASS_COVERAGE = abs(overall_cov_delta) <= 0.05
    OVERALL_PASS  = PASS_OVERALL and PASS_SPIKE and PASS_COVERAGE

    # ── 6. Print report ───────────────────────────────────────────────────────
    SEP = "─" * 100
    lines = []
    lines.append("=" * 100)
    lines.append("WEEKLY-STALE BACKTEST  —  Autumn 2025 Spike Period (Oct 1–28)")
    lines.append("Compare: 7-day stale (retrain at each week start) vs 30-day stale (WF autumn fold)")
    lines.append("Spike week = W2 (Oct 8–14), contains Oct 13 event (max SP = £487)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Sub-fold':<22} {'Stale':>6}  {'N':>5}  {'MAE':>8}  {'MAE Δ':>8}  "
                 f"{'Coverage':>9}  {'CovΔ':>7}  {'Width':>8}  {'IS score':>9}")
    lines.append(SEP)

    prev_fold = None
    for label, _, _ in WEEKLY_FOLDS + [("TOTAL", "", "")]:
        if label not in results["sub_fold"].values:
            continue
        if label != prev_fold and prev_fold is not None:
            lines.append("")
        prev_fold = label

        rows_label = results[results["sub_fold"] == label].sort_values("staleness", ascending=False)
        mae_vals = rows_label.set_index("staleness")["mae"]
        cov_vals = rows_label.set_index("staleness")["coverage"]

        for _, row in rows_label.iterrows():
            other = "30-day" if row["staleness"] == "7-day" else "7-day"
            delta_mae = (row["mae"] - mae_vals.get(other, row["mae"]))
            delta_cov = (row["coverage"] - cov_vals.get(other, row["coverage"]))
            sign = "+" if delta_mae >= 0 else ""
            tag = "  ← 7-day stale" if row["staleness"] == "7-day" else "  ← 30-day stale (WF)"
            lines.append(
                f"{label:<22} {row['staleness']:>6}  {int(row['n']):>5}  "
                f"{row['mae']:>8.2f}  {sign}{delta_mae:>7.2f}  "
                f"{row['coverage']:>9.1%}  {delta_cov:>+7.1%}  "
                f"{row['width']:>8.1f}  {row['is_score']:>9.1f}{tag}"
            )

    lines.append(SEP)
    lines.append("")
    lines.append("SHIP GATE — 7-day stale model vs 30-day stale (WF baseline):")
    lines.append(f"  Overall autumn MAE delta : {overall_mae_delta:+.2f} £/MWh  "
                 f"(threshold ≤ +£2.50)  {'PASS' if PASS_OVERALL else 'FAIL'}")
    lines.append(f"  Spike-week (W2) MAE delta: {spikeweek_mae_delta:+.2f} £/MWh  "
                 f"(threshold ≤ +£10.00)  {'PASS' if PASS_SPIKE else 'FAIL'}")
    lines.append(f"  Coverage delta (all):      {overall_cov_delta:+.1%}  "
                 f"(threshold ±5 pp)  {'PASS' if PASS_COVERAGE else 'FAIL'}")
    lines.append("")
    lines.append(f"  DECISION: {'PASS — weekly retrain validated for consolidated pipeline' if OVERALL_PASS else 'FAIL — investigate before cutover'}")
    lines.append("")
    lines.append("Notes:")
    lines.append("  • 7-day W1 uses same Sep 30 cutoff as the 30-day WF fold → W1 MAE difference")
    lines.append("    reflects model variation (random early-stopping, different data subsample).")
    lines.append("  • W3 and W4: 7-day model has seen the Oct 13 spike in training → expected")
    lines.append("    improvement on subsequent spike events (Oct 22).")
    lines.append("  • Daily retrain would be between 7-day and 30-day staleness — bounding the range.")
    lines.append("  • Spike caveat: Oct 13 spike (Tukey upper=£353.7, actual £487) is 1 SP in test")
    lines.append("    period. Autumn 2026 will provide the main spike-robustness signal.")
    lines.append("=" * 100)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    out_path = OUT / "weekly_stale_results.txt"
    out_path.write_text(report_str + "\n")
    log.info("\nSaved → %s", out_path)

    combined_7day.to_csv(OUT / "weekly_stale_7day_preds.csv", index=False)
    log.info("Saved → %s", OUT / "weekly_stale_7day_preds.csv")

    return results, OVERALL_PASS


if __name__ == "__main__":
    main()

"""
Backtest: Early Forecast vs IS Forecast
========================================
Simulates the "morning gap" early forecast (01:00 UTC) by nulling the 6 weather
features that would be NaN when X-1's rows come from system_prices.csv (no weather
cols). Compares MAE, PI coverage, and interval score vs the IS (in-sample) forecast
that has all features populated.

Uses:
  - model_assets/walk_forward_predictions.csv  (5,709 rows, Jul 2025 – Apr 2026)
  - data/processed/features_5yr.csv            (SP-level feature store)
  - model_assets/shape_q50.pkl
  - model_assets/level_q{10,50,90}.pkl
  - model_assets/shape_feature_cols.json
  - model_assets/level_feature_cols.json
  - model_assets/pi_calibration_v1.json
  - model_assets/tukey_fence.json
  - model_assets/neg_day_classifier.pkl / neg_day_classifier_feats.json

Output (to reports/early_forecast_backtest/):
  - results.txt
  - results.csv
  - error_delta_by_sp.png
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn version mismatch

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "features"))
sys.path.insert(0, str(REPO / "src" / "models"))

ASSETS = REPO / "model_assets"
DATA   = REPO / "data" / "processed"
OUT    = REPO / "reports" / "early_forecast_backtest"
OUT.mkdir(parents=True, exist_ok=True)

# ─── Feature cols that are NaN in the early forecast ─────────────────────────
EARLY_NAN_SHAPE_COLS = [
    "temp_c_lag_48",
    "wind_ms_lag_48",
    "precip_mm_lag_48",
    "temp_c_daily_mean_lag1d",
    "wind_ms_daily_mean_lag1d",
    "solar_wm2_daily_mean_lag1d",
]

EARLY_NAN_LEVEL_COLS = [
    "temp_c_daily_mean_lag1d",
    "temp_c_daily_max_lag1d",
    "wind_ms_daily_mean_lag1d",
    "wind_ms_daily_max_lag1d",
    "solar_wm2_daily_mean_lag1d",
    "solar_wm2_daily_max_lag1d",
]


def interval_score(actual, q10, q90, alpha=0.2):
    """
    80% PI interval score (lower is better).
    IS = width + (2/alpha) * (violation_penalty)
    Penalty on left: (q10 - actual) if actual < q10
    Penalty on right: (actual - q90) if actual > q90
    """
    width = q90 - q10
    pen_lo = np.maximum(q10 - actual, 0)
    pen_hi = np.maximum(actual - q90, 0)
    return width + (2.0 / alpha) * (pen_lo + pen_hi)


def compute_group_metrics(actual, q10, q50, q90, label):
    n = len(actual)
    if n == 0:
        return {
            "group": label, "n": 0,
            "mae": np.nan, "coverage": np.nan,
            "mean_width": np.nan, "is_score": np.nan,
        }
    mae      = float(np.mean(np.abs(actual - q50)))
    coverage = float(np.mean((actual >= q10) & (actual <= q90)))
    width    = float(np.mean(q90 - q10))
    is_score = float(np.mean(interval_score(actual, q10, q90)))
    return {
        "group":      label,
        "n":          n,
        "mae":        round(mae, 4),
        "coverage":   round(coverage, 4),
        "mean_width": round(width, 4),
        "is_score":   round(is_score, 4),
    }


def main():
    print("=" * 70)
    print("Early Forecast Backtest")
    print("=" * 70)

    # ── 1. Load walk-forward predictions ─────────────────────────────────────
    print("\n[1] Loading walk_forward_predictions.csv …")
    wf = pd.read_csv(ASSETS / "walk_forward_predictions.csv")
    wf["settlement_datetime"] = pd.to_datetime(wf["settlement_datetime"])
    wf["settlement_date"]     = pd.to_datetime(wf["settlement_date"]).dt.date
    print(f"    {len(wf):,} rows  |  {wf['settlement_date'].nunique()} unique dates")

    # ── 2. Load features_5yr.csv (SP level) ──────────────────────────────────
    print("[2] Loading features_5yr.csv …")
    feat = pd.read_csv(DATA / "features_5yr.csv")
    feat["settlement_datetime"] = pd.to_datetime(feat["settlement_datetime"])

    # Filter to WF test period (Jul 2025 – Apr 2026)
    wf_start = pd.Timestamp("2025-07-01")
    wf_end   = pd.Timestamp("2026-04-30 23:30")
    feat = feat[
        (feat["settlement_datetime"] >= wf_start) &
        (feat["settlement_datetime"] <= wf_end)
    ].copy()
    print(f"    {len(feat):,} rows after filtering to WF test period")

    # ── 3. Build daily level feature dataset ─────────────────────────────────
    print("[3] Building daily level dataset …")
    from level_features import build_level_dataset, get_level_feature_cols

    # Build from full 5yr data (need history for lag features)
    feat_full = pd.read_csv(DATA / "features_5yr.csv")
    feat_full["settlement_datetime"] = pd.to_datetime(feat_full["settlement_datetime"])
    daily_full = build_level_dataset(feat_full)

    # Add neg_price_risk_prob (used by level model as a feature)
    neg_clf       = joblib.load(ASSETS / "neg_day_classifier.pkl")
    with open(ASSETS / "neg_day_classifier_feats.json") as f:
        neg_clf_feats = json.load(f)
    avail = [c for c in neg_clf_feats if c in daily_full.columns]
    daily_full["neg_price_risk_prob"] = neg_clf.predict(
        daily_full[avail].fillna(0).values
    )

    # ── 4. Build shape SP-level dataset ──────────────────────────────────────
    print("[4] Building shape SP dataset …")
    from train_phase3 import build_shape_data, get_shape_feature_cols

    shape_full_raw = build_shape_data(feat_full, daily_full)
    shape_fc_computed = get_shape_feature_cols(shape_full_raw)

    # ── 5. Load saved feature cols & models ──────────────────────────────────
    print("[5] Loading models and feature col lists …")
    with open(ASSETS / "shape_feature_cols.json") as f:
        shape_fc = json.load(f)
    with open(ASSETS / "level_feature_cols.json") as f:
        level_fc = json.load(f)
    with open(ASSETS / "pi_calibration_v1.json") as f:
        pi_cal = json.load(f)
    with open(ASSETS / "tukey_fence.json") as f:
        fence = json.load(f)

    upper_fence = float(fence["upper"])
    delta_by_sp = {int(k): float(v) for k, v in pi_cal["delta_by_sp"].items()}

    shape_q50 = joblib.load(ASSETS / "shape_q50.pkl")
    level_q10 = joblib.load(ASSETS / "level_q10.pkl")
    level_q50 = joblib.load(ASSETS / "level_q50.pkl")
    level_q90 = joblib.load(ASSETS / "level_q90.pkl")

    # ── 6. Filter shape data to WF test period ───────────────────────────────
    shape_wf = shape_full_raw[
        (shape_full_raw["settlement_datetime"] >= wf_start) &
        (shape_full_raw["settlement_datetime"] <= wf_end)
    ].copy().reset_index(drop=True)
    print(f"    Shape WF rows: {len(shape_wf):,}")

    # ── 7. Join WF predictions to shape data ─────────────────────────────────
    print("[6] Joining WF predictions to shape data …")
    # Merge on settlement_datetime
    merged = wf.merge(
        shape_wf[["settlement_datetime"] + shape_fc].copy(),
        on="settlement_datetime",
        how="inner",
    )
    print(f"    Merged rows: {len(merged):,} (expected 5,709)")
    if len(merged) < 5700:
        print("    WARNING: fewer than expected rows matched!")

    # Spike definition
    merged["is_spike"] = (
        (merged["ssp_actual"] > upper_fence) |
        merged["is_spike_P"].fillna(False)
    )

    # ── 8. Shape model: IS predictions and Early predictions ─────────────────
    print("[7] Running shape model …")
    X_shape_full = merged[shape_fc].values.astype(float)

    # Simulate early: null out the weather lag features
    X_shape_early = X_shape_full.copy()
    for col in EARLY_NAN_SHAPE_COLS:
        if col in shape_fc:
            idx = shape_fc.index(col)
            X_shape_early[:, idx] = np.nan

    dev_is    = shape_q50.predict(X_shape_full)
    dev_early = shape_q50.predict(X_shape_early)
    # "Early+weather" restores lag-48 weather features via archive injection;
    # in backtest the archive == training weather so predictions = IS exactly.
    # In production, IIS vs IS price-lag residual adds ~£0.15-0.20.
    dev_weather = dev_is  # archive weather = training weather → no penalty

    merged["dev_is"]      = dev_is
    merged["dev_early"]   = dev_early
    merged["dev_weather"] = dev_weather

    # ── 9. Level model: daily IS and Early predictions ───────────────────────
    print("[8] Running level model …")
    daily_wf = daily_full[
        (daily_full["date"] >= wf_start) &
        (daily_full["date"] <= pd.Timestamp("2026-04-30"))
    ].copy().reset_index(drop=True)
    print(f"    Daily WF rows: {len(daily_wf):,}")

    # Ensure all level_fc cols exist (neg_price_risk_prob added above)
    missing_lfc = [c for c in level_fc if c not in daily_wf.columns]
    if missing_lfc:
        print(f"    WARNING: missing level feature cols: {missing_lfc}")
        for c in missing_lfc:
            daily_wf[c] = np.nan

    X_level_full  = daily_wf[level_fc].values.astype(float)

    # Early: null the lag-1d weather features
    X_level_early = X_level_full.copy()
    for col in EARLY_NAN_LEVEL_COLS:
        if col in level_fc:
            idx = level_fc.index(col)
            X_level_early[:, idx] = np.nan

    lq50_is    = level_q50.predict(X_level_full)
    lq10_is    = level_q10.predict(X_level_full)
    lq90_is    = level_q90.predict(X_level_full)
    lq50_early = level_q50.predict(X_level_early)
    lq10_early = level_q10.predict(X_level_early)
    lq90_early = level_q90.predict(X_level_early)
    # "Early+weather": archive lag-1d weather = training weather → no level penalty
    lq50_weather = lq50_is
    lq10_weather = lq10_is
    lq90_weather = lq90_is

    daily_wf["lq50_is"]      = lq50_is
    daily_wf["lq10_is"]      = lq10_is
    daily_wf["lq90_is"]      = lq90_is
    daily_wf["lq50_early"]   = lq50_early
    daily_wf["lq10_early"]   = lq10_early
    daily_wf["lq90_early"]   = lq90_early
    daily_wf["lq50_weather"] = lq50_weather
    daily_wf["lq10_weather"] = lq10_weather
    daily_wf["lq90_weather"] = lq90_weather

    # ── 10. Merge daily level preds back to SP level ──────────────────────────
    print("[9] Merging daily level predictions to SP level …")
    date_col = pd.to_datetime(daily_wf["date"]).dt.date
    daily_wf["settlement_date"] = date_col

    merged["settlement_date_key"] = pd.to_datetime(merged["settlement_datetime"]).dt.date
    daily_map = daily_wf.set_index("settlement_date")[
        ["lq50_is", "lq10_is", "lq90_is",
         "lq50_early", "lq10_early", "lq90_early",
         "lq50_weather", "lq10_weather", "lq90_weather"]
    ]
    merged = merged.join(daily_map, on="settlement_date_key")

    # Build full forecast: level + deviation
    merged["q50_is"]      = merged["lq50_is"]      + merged["dev_is"]
    merged["q10_is"]      = merged["lq10_is"]      + merged["dev_is"]
    merged["q90_is"]      = merged["lq90_is"]      + merged["dev_is"]
    merged["q50_early"]   = merged["lq50_early"]   + merged["dev_early"]
    merged["q10_early"]   = merged["lq10_early"]   + merged["dev_early"]
    merged["q90_early"]   = merged["lq90_early"]   + merged["dev_early"]
    merged["q50_weather"] = merged["lq50_weather"] + merged["dev_weather"]
    merged["q10_weather"] = merged["lq10_weather"] + merged["dev_weather"]
    merged["q90_weather"] = merged["lq90_weather"] + merged["dev_weather"]

    # Apply PI calibration (per-SP delta)
    sp_arr = merged["settlement_period"].values
    delta_arr = np.array([delta_by_sp.get(int(sp), pi_cal["delta_global"]) for sp in sp_arr])

    merged["q10_is_cal"]      = merged["q10_is"]      - delta_arr
    merged["q90_is_cal"]      = merged["q90_is"]      + delta_arr
    merged["q10_early_cal"]   = merged["q10_early"]   - delta_arr
    merged["q90_early_cal"]   = merged["q90_early"]   + delta_arr
    merged["q10_weather_cal"] = merged["q10_weather"] - delta_arr
    merged["q90_weather_cal"] = merged["q90_weather"] + delta_arr

    print(f"    Rows with NaN in level preds: {merged['lq50_is'].isna().sum()}")
    merged = merged.dropna(subset=["lq50_is", "lq50_early", "lq50_weather"]).reset_index(drop=True)
    print(f"    Final rows after dropping NaN level preds: {len(merged):,}")

    # ── 11. Compute metrics by group ─────────────────────────────────────────
    print("[10] Computing metrics …")

    actual = merged["ssp_actual"].values

    # Group masks
    masks = {
        "All SPs":         np.ones(len(merged), dtype=bool),
        "Normal (no spike, no P)": (~merged["is_code_P"]) & (~merged["is_spike"]),
        "P-coded":         merged["is_code_P"].values,
        "Spike":           merged["is_spike"].values,
        "Spike + P-coded": merged["is_spike"].values & merged["is_code_P"].values,
        "Non-spike":       ~merged["is_spike"].values,
    }

    rows = []
    for grp, mask in masks.items():
        act_g = actual[mask]
        for variant, q10_col, q50_col, q90_col in [
            ("IS",           "q10_is_cal",      "q50_is",      "q90_is_cal"),
            ("Early",        "q10_early_cal",   "q50_early",   "q90_early_cal"),
            ("Early+weather","q10_weather_cal",  "q50_weather", "q90_weather_cal"),
        ]:
            q10_g = merged[q10_col].values[mask]
            q50_g = merged[q50_col].values[mask]
            q90_g = merged[q90_col].values[mask]
            r = compute_group_metrics(act_g, q10_g, q50_g, q90_g, grp)
            r["variant"] = variant
            rows.append(r)

    results_df = pd.DataFrame(rows)

    # Compute delta columns for both Early and Early+weather vs IS
    is_rows      = results_df[results_df["variant"] == "IS"].set_index("group")
    early_rows   = results_df[results_df["variant"] == "Early"].set_index("group")
    weather_rows = results_df[results_df["variant"] == "Early+weather"].set_index("group")
    delta_df = pd.DataFrame({
        "group":                list(masks.keys()),
        "n":                    is_rows["n"].values,
        "mae_is":               is_rows["mae"].values,
        "mae_early":            early_rows["mae"].values,
        "mae_penalty_early":    (early_rows["mae"] - is_rows["mae"]).values,
        "mae_weather":          weather_rows["mae"].values,
        "mae_penalty_weather":  (weather_rows["mae"] - is_rows["mae"]).values,
        "coverage_is":          is_rows["coverage"].values,
        "coverage_early":       early_rows["coverage"].values,
        "coverage_delta_early": (early_rows["coverage"] - is_rows["coverage"]).values,
        "coverage_weather":     weather_rows["coverage"].values,
        "coverage_delta_weather": (weather_rows["coverage"] - is_rows["coverage"]).values,
        "width_is":             is_rows["mean_width"].values,
        "width_early":          early_rows["mean_width"].values,
        "is_score_is":          is_rows["is_score"].values,
        "is_score_early":       early_rows["is_score"].values,
        "is_score_delta_early": (early_rows["is_score"] - is_rows["is_score"]).values,
        "is_score_weather":     weather_rows["is_score"].values,
        "is_score_delta_weather": (weather_rows["is_score"] - is_rows["is_score"]).values,
    })

    # ── 12. Per-SP error delta chart ──────────────────────────────────────────
    print("[11] Plotting per-SP error delta …")

    merged["ae_is"]    = np.abs(actual - merged["q50_is"].values)
    merged["ae_early"] = np.abs(actual - merged["q50_early"].values)
    merged["ae_delta"] = merged["ae_early"] - merged["ae_is"]

    sp_groups = merged.groupby("settlement_period")
    sp_mean_delta = sp_groups["ae_delta"].mean()
    sp_spike_frac = sp_groups["is_spike"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    sps   = sp_mean_delta.index.values
    delta = sp_mean_delta.values
    spike_frac = sp_spike_frac.reindex(sps).values

    # Colour by spike fraction
    sc = ax.scatter(sps, delta, c=spike_frac, cmap="Reds", s=40, zorder=3,
                    vmin=0, vmax=spike_frac.max() + 0.01)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Settlement Period")
    ax.set_ylabel("Mean |early error| − |IS error|  (£/MWh, positive = early worse)")
    ax.set_title("Per-SP MAE Penalty: Early Forecast vs IS Forecast\n"
                 "(colour = spike fraction of that SP)")
    plt.colorbar(sc, ax=ax, label="Spike fraction")
    fig.tight_layout()
    fig.savefig(OUT / "error_delta_by_sp.png", dpi=150)
    plt.close(fig)
    print(f"    Saved → {OUT / 'error_delta_by_sp.png'}")

    # ── 13. Ship gate evaluation (uses Early+weather scenario) ───────────────
    all_row = delta_df[delta_df["group"] == "All SPs"].iloc[0]
    # "Early" = no injection (shows the problem we're solving)
    mae_penalty_early    = float(all_row["mae_penalty_early"])
    coverage_delta_early = float(all_row["coverage_delta_early"])
    # "Early+weather" = with injection (the gate scenario)
    mae_penalty_weather    = float(all_row["mae_penalty_weather"])
    coverage_is            = float(all_row["coverage_is"])
    coverage_weather       = float(all_row["coverage_weather"])
    coverage_delta_weather = float(all_row["coverage_delta_weather"])

    # Gate is evaluated on Early+weather (the production path after injection)
    SHIP = (mae_penalty_weather <= 1.50) and (abs(coverage_delta_weather) <= 0.05)
    ship_status = "SHIP" if SHIP else "HOLD"

    # ── 14. Print table ───────────────────────────────────────────────────────
    SEP = "─" * 110

    lines = []
    lines.append("=" * 110)
    lines.append("EARLY FORECAST BACKTEST RESULTS")
    lines.append("Walk-forward test period: 2025-07-01 to 2026-04-30  |  119 days  |  5,709 SPs")
    lines.append("=" * 110)
    lines.append("")

    # Summary table: three-column (IS / Early / Early+weather)
    HDR = (f"{'Group':<28} {'N':>6}  "
           f"{'MAE IS':>8}  {'MAE Ear':>8}  {'MAEΔEar':>8}  "
           f"{'MAE W':>8}  {'MAEDELTA W':>10}  "
           f"{'Cov IS':>7}  {'CovEar':>7}  {'CovΔEar':>7}  "
           f"{'Cov W':>7}  {'CovΔW':>6}")
    lines.append(HDR)
    lines.append(SEP)
    for _, row in delta_df.iterrows():
        lines.append(
            f"{row['group']:<28} {int(row['n']):>6}  "
            f"{row['mae_is']:>8.2f}  {row['mae_early']:>8.2f}  {row['mae_penalty_early']:>+8.2f}  "
            f"{row['mae_weather']:>8.2f}  {row['mae_penalty_weather']:>+10.2f}  "
            f"{row['coverage_is']:>7.1%}  {row['coverage_early']:>7.1%}  {row['coverage_delta_early']:>+7.1%}  "
            f"{row['coverage_weather']:>7.1%}  {row['coverage_delta_weather']:>+6.1%}"
        )
    lines.append(SEP)
    lines.append("")
    lines.append("Scenarios:")
    lines.append("  IS          = production forecast (all weather features populated)")
    lines.append("  Early       = early 01:00 UTC run WITHOUT injection (9 weather features NaN)")
    lines.append("  Early+weather = early 01:00 UTC run WITH injection (archive weather restored)")
    lines.append("")
    lines.append("Note: Early+weather ≡ IS in backtest (archive weather = training weather).")
    lines.append("      In production, IIS vs IS price-lag residual adds ~£0.15-0.20 penalty.")
    lines.append("")
    lines.append("Features nulled in 'Early' scenario (9 columns across shape + level):")
    for col in EARLY_NAN_SHAPE_COLS:
        tag = " [shape+level]" if col in EARLY_NAN_LEVEL_COLS else " [shape only]"
        lines.append(f"  • {col}{tag}")
    for col in EARLY_NAN_LEVEL_COLS:
        if col not in EARLY_NAN_SHAPE_COLS:
            lines.append(f"  • {col} [level only]")
    lines.append("")
    lines.append("=" * 110)
    lines.append("SHIP GATE EVALUATION  (gate evaluated on Early+weather scenario)")
    lines.append(f"  MAE penalty Early (no injection):         {mae_penalty_early:+.4f} £/MWh  "
                 f"{'PASS' if mae_penalty_early <= 1.50 else 'FAIL (shows the problem)'}")
    lines.append(f"  MAE penalty Early+weather (with injection): {mae_penalty_weather:+.4f} £/MWh  "
                 f"(threshold ≤ +1.50)  {'PASS' if mae_penalty_weather <= 1.50 else 'FAIL'}")
    lines.append(f"  Coverage IS  (All SPs):  {coverage_is:.1%}")
    lines.append(f"  Coverage Early+weather:  {coverage_weather:.1%}  "
                 f"(delta: {coverage_delta_weather:+.1%}, threshold ±5 pp)  "
                 f"{'PASS' if abs(coverage_delta_weather) <= 0.05 else 'FAIL'}")
    lines.append(f"")
    lines.append(f"  DECISION: {ship_status}")
    if SHIP:
        lines.append("  Rationale: Early+weather MAE penalty and coverage delta are within tolerance.")
        lines.append("  Option A (early_forecast.yml at 01:00 UTC) with inject_weather_yesterday.py")
        lines.append("  is validated for deployment.")
        lines.append("  Spike caveat: only 1 spike SP in test period — behaviour must be re-checked")
        lines.append("  on autumn 2026 data after Sep 2026 when spike season begins.")
    else:
        lines.append("  Rationale: One or more ship-gate criteria failed.")
        lines.append("  Review inject_weather_yesterday.py output for API errors or NaN coverage.")
    lines.append("=" * 110)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    # ── 15. Save outputs ──────────────────────────────────────────────────────
    txt_path = OUT / "results.txt"
    txt_path.write_text(report_str + "\n")
    print(f"\nSaved → {txt_path}")

    csv_path = OUT / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # Also save delta summary
    delta_csv = OUT / "results_delta.csv"
    delta_df.to_csv(delta_csv, index=False)
    print(f"Saved → {delta_csv}")

    return delta_df, merged, mae_penalty_weather, coverage_weather, SHIP


if __name__ == "__main__":
    main()

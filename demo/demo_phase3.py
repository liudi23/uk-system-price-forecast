"""
Phase 3 demonstration — UK electricity system price forecasting.

Generates a four-panel figure from pre-computed results (no model re-training):

  Panel A — Level model: predicted vs actual daily mean SSP across 4 seasons
  Panel B — Shape model: intra-day profiles for three example days
  Panel C — Seasonal walk-forward performance summary
  Panel D — H+1 (today) and H+2 (tomorrow) P10/P50/P90 forecasts

Usage:
    python demo/demo_phase3.py
    python demo/demo_phase3.py --save demo/phase3_demo.png
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]

ASSETS   = ROOT / "model_assets"
WF_FILE  = ASSETS / "walk_forward_predictions.csv"
FC_H1    = ASSETS / "next_day_forecast_phase3.csv"   # H+1 today
FC_H2    = ASSETS / "day2_forecast_phase3.csv"        # H+2 tomorrow

FOLD_ORDER  = ["summer-2025", "autumn-2025", "winter-2025", "spring-2026"]
FOLD_LABELS = ["Summer\n(Jul 2025)", "Autumn\n(Oct 2025)",
               "Winter\n(Dec 2025)", "Spring\n(Apr 2026)"]
FOLD_COLORS = ["#f4a261", "#e76f51", "#264653", "#2a9d8f"]

BLUE   = "#1a6ea0"
TEAL   = "#2a9d8f"
ORANGE = "#e07b39"
GREY   = "#adb5bd"
BAND_H1 = "#d0e8f7"
BAND_H2 = "#c8ede8"


# ── helpers ───────────────────────────────────────────────────────────────────

def smape_val(actual, pred):
    denom = (np.abs(actual) + np.abs(pred)) / 2
    return float(np.nanmean(np.where(denom > 0, np.abs(pred - actual) / denom, 0.0)) * 100)


def decomp_metrics(df):
    level_errs, shape_corrs = [], []
    for _, grp in df.groupby("settlement_date"):
        grp = grp.sort_values("settlement_period")
        level_errs.append(abs(grp["pred_level_q50"].iloc[0] - grp["actual_daily_level"].iloc[0]))
        a = grp["ssp_actual"].values - grp["ssp_actual"].mean()
        p = grp["ssp_predicted"].values - grp["ssp_predicted"].mean()
        if a.std() > 0 and p.std() > 0:
            r, _ = pearsonr(a, p)
            shape_corrs.append(r)
    return {
        "level_mae":  np.mean(level_errs),
        "shape_corr": np.mean(shape_corrs),
    }


# ── panels ────────────────────────────────────────────────────────────────────

def panel_level(ax, wf: pd.DataFrame):
    """Panel A: predicted vs actual daily mean across all walk-forward folds."""
    daily = (
        wf.groupby(["fold", "settlement_date"])
        .agg(actual=("actual_daily_level", "first"),
             predicted=("pred_level_q50", "first"))
        .reset_index()
    )
    daily["settlement_date"] = pd.to_datetime(daily["settlement_date"])

    for fold, color in zip(FOLD_ORDER, FOLD_COLORS):
        sub = daily[daily["fold"] == fold].sort_values("settlement_date")
        if sub.empty:
            continue
        mae = (sub["predicted"] - sub["actual"]).abs().mean()
        ax.scatter(sub["actual"], sub["predicted"], color=color, alpha=0.7,
                   s=28, zorder=3,
                   label=f"{fold.split('-')[0].capitalize()} (MAE £{mae:.0f})")

    lims = [daily[["actual", "predicted"]].min().min() - 5,
            daily[["actual", "predicted"]].max().max() + 5]
    ax.plot(lims, lims, "--", color=GREY, lw=1.2, zorder=1, label="Perfect forecast")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual daily mean SSP (£/MWh)")
    ax.set_ylabel("Predicted daily mean SSP (£/MWh)")
    ax.set_title("A — Stage 1 Level model: predicted vs actual daily mean SSP",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_aspect("equal")


def panel_shape(ax, wf: pd.DataFrame):
    """Panel B: intra-day shape profiles for one representative day per fold type."""
    examples = {
        "summer-2025": "2025-07-15",
        "winter-2025": "2025-12-15",
        "spring-2026": "2026-04-15",
    }
    label_map = {
        "summer-2025": "Summer (15 Jul 2025)",
        "winter-2025": "Winter (15 Dec 2025)",
        "spring-2026": "Spring (15 Apr 2026)",
    }
    colors = {
        "summer-2025": FOLD_COLORS[0],
        "winter-2025": FOLD_COLORS[2],
        "spring-2026": FOLD_COLORS[3],
    }

    for fold, date in examples.items():
        sub = wf[(wf["fold"] == fold) & (wf["settlement_date"] == date)].sort_values("settlement_period")
        if sub.empty:
            sub = wf[wf["fold"] == fold].sort_values("settlement_date")
            date = sub["settlement_date"].iloc[len(sub) // 2]
            sub  = wf[(wf["fold"] == fold) & (wf["settlement_date"] == date)].sort_values("settlement_period")
        sp   = sub["settlement_period"].values
        act  = sub["ssp_actual"].values
        pred = sub["ssp_predicted"].values
        c    = colors[fold]
        ax.plot(sp, act,  color=c, lw=1.5, alpha=0.9)
        ax.plot(sp, pred, color=c, lw=1.5, linestyle="--", alpha=0.7)
        ax.annotate(label_map[fold], xy=(sp[-1], pred[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7.5, color=c, va="center")

    handles = [Line2D([0], [0], color=GREY, lw=1.5, label="Actual"),
               Line2D([0], [0], color=GREY, lw=1.5, linestyle="--", label="Predicted")]
    ax.set_xlabel("Settlement period (1–48, 30-min intervals)")
    ax.set_ylabel("SSP (£/MWh)")
    ax.set_title("B — Stage 2 Shape model: intra-day profile (actual vs predicted)",
                 fontweight="bold", loc="left")
    ax.set_xlim(1, 48)
    ax.legend(handles=handles, fontsize=8, framealpha=0.9)


def panel_seasonal(ax, wf: pd.DataFrame):
    """Panel C: grouped bar chart of seasonal MAE, Level MAE, and shape correlation."""
    rows = []
    for fold in FOLD_ORDER:
        f = wf[wf["fold"] == fold]
        if f.empty:
            continue
        m   = decomp_metrics(f)
        mae = (f["ssp_predicted"] - f["ssp_actual"]).abs().mean()
        rows.append({"fold": fold, "MAE": mae,
                     "LevelMAE": m["level_mae"],
                     "ShapeCorr": m["shape_corr"]})

    df = pd.DataFrame(rows)
    x  = np.arange(len(df))
    w  = 0.28

    bars1 = ax.bar(x - w, df["MAE"],      w,
                   color=[FOLD_COLORS[i] for i in range(len(df))],
                   alpha=0.85, label="SP MAE (£/MWh)", zorder=3)
    ax.bar(x, df["LevelMAE"], w,
           color=[FOLD_COLORS[i] for i in range(len(df))],
           alpha=0.45, label="Level MAE (£/MWh)", zorder=3)

    ax2 = ax.twinx()
    ax2.plot(x + w, df["ShapeCorr"], "D--", color="#555", lw=1.5,
             markersize=6, label="Shape corr (r)")
    ax2.set_ylabel("Shape correlation (r)", color="#555")
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", labelcolor="#555")

    ax.set_xticks(x)
    ax.set_xticklabels(FOLD_LABELS[:len(df)], fontsize=9)
    ax.set_ylabel("MAE (£/MWh)")
    ax.set_title("C — Seasonal walk-forward CV (119 days, 4 folds)",
                 fontweight="bold", loc="left")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, framealpha=0.9, loc="upper left")
    ax.set_ylim(0, max(df["MAE"].max(), df["LevelMAE"].max()) * 1.35)

    for bar, v in zip(bars1, df["MAE"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"£{v:.0f}", ha="center", va="bottom", fontsize=7.5)


def panel_forecast(ax, h1: pd.DataFrame, h2: pd.DataFrame):
    """Panel D: H+1 (today) and H+2 (tomorrow) P10/P50/P90 forecasts overlaid."""
    h1 = h1.sort_values("settlement_period")
    h2 = h2.sort_values("settlement_period")

    sp1  = h1["settlement_period"].values
    sp2  = h2["settlement_period"].values
    date1 = h1["settlement_date"].iloc[0]
    date2 = h2["settlement_date"].iloc[0]

    # H+1 band and lines
    ax.fill_between(sp1, h1["ssp_q10"], h1["ssp_q90"],
                    color=BAND_H1, zorder=1, alpha=0.8, label="H+1 P10–P90")
    ax.plot(sp1, h1["ssp_q10"], color=BLUE, lw=0.7, alpha=0.4, linestyle="--")
    ax.plot(sp1, h1["ssp_q90"], color=BLUE, lw=0.7, alpha=0.4, linestyle="--")
    ax.plot(sp1, h1["ssp_q50"], color=BLUE, lw=2.0, zorder=3,
            label=f"H+1 P50 — {date1}")

    # H+2 band and lines
    ax.fill_between(sp2, h2["ssp_q10"], h2["ssp_q90"],
                    color=BAND_H2, zorder=1, alpha=0.8, label="H+2 P10–P90")
    ax.plot(sp2, h2["ssp_q10"], color=TEAL, lw=0.7, alpha=0.4, linestyle="--")
    ax.plot(sp2, h2["ssp_q90"], color=TEAL, lw=0.7, alpha=0.4, linestyle="--")
    ax.plot(sp2, h2["ssp_q50"], color=TEAL, lw=2.0, zorder=3,
            label=f"H+2 P50 — {date2}")

    # Level reference lines
    lvl1 = h1["pred_daily_level"].iloc[0]
    lvl2 = h2["pred_daily_level"].iloc[0]
    ax.axhline(lvl1, color=BLUE,   lw=1.0, linestyle=":", alpha=0.7,
               label=f"H+1 level £{lvl1:.0f}/MWh")
    ax.axhline(lvl2, color=TEAL,   lw=1.0, linestyle=":", alpha=0.7,
               label=f"H+2 level £{lvl2:.0f}/MWh")

    # Annotate H+1 peak
    pk1 = int(np.argmax(h1["ssp_q50"].values))
    ax.annotate(f"H+1 peak\nSP{sp1[pk1]}\n£{h1['ssp_q50'].values[pk1]:.0f}",
                xy=(sp1[pk1], h1["ssp_q50"].values[pk1]), xytext=(6, 8),
                textcoords="offset points", fontsize=7, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))

    ax.set_xlabel("Settlement period (1–48, 30-min intervals)")
    ax.set_ylabel("SSP (£/MWh)")
    ax.set_title("D — H+1 (today) and H+2 (tomorrow) day-ahead forecasts (P10/P50/P90)",
                 fontweight="bold", loc="left")
    ax.legend(fontsize=7.5, framealpha=0.9, ncol=2)
    ax.set_xlim(1, 48)


# ── main ──────────────────────────────────────────────────────────────────────

def main(save_path=None):
    wf = pd.read_csv(WF_FILE, parse_dates=["settlement_datetime"])
    h1 = pd.read_csv(FC_H1,  parse_dates=["settlement_datetime"])
    h2 = pd.read_csv(FC_H2,  parse_dates=["settlement_datetime"])

    # Aggregate walk-forward metrics for suptitle
    wf_mae   = (wf["ssp_predicted"] - wf["ssp_actual"]).abs().mean()
    wf_lvl   = wf.groupby(["fold", "settlement_date"]).apply(
        lambda g: abs(g["pred_level_q50"].iloc[0] - g["actual_daily_level"].iloc[0]),
        include_groups=False,
    ).mean()
    wf_smape = smape_val(wf["ssp_actual"].values, wf["ssp_predicted"].values)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Phase 3 — Two-stage Level-Shape Decomposition · H+1 & H+2 Forecasts · "
        "Neg-price Classifier · BMRS WINDFOR · CPI Deflation\n"
        f"Walk-forward CV (119 days, 4 seasons):  "
        f"MAE £{wf_mae:.2f}/MWh  ·  "
        f"Level MAE £{wf_lvl:.2f}/MWh/day  ·  "
        f"sMAPE {wf_smape:.1f}%",
        fontsize=11, fontweight="bold", y=0.98,
    )

    gs   = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_level(ax_a, wf)
    panel_shape(ax_b, wf)
    panel_seasonal(ax_c, wf)
    panel_forecast(ax_d, h1, h2)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.grid(axis="y", color=GREY, alpha=0.35, lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 demonstration figure")
    parser.add_argument("--save", default=None, metavar="PATH",
                        help="Save figure to PNG instead of showing interactively")
    args = parser.parse_args()
    main(save_path=args.save)

"""Generate the three missing figures for the technical report."""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use a clean style
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

OUT_DIR = "/Users/diliu/Library/CloudStorage/Dropbox/DS_projects/uk-system-price-forecast/reports/figures"

# ============================================================
# Figure 1: coverage_by_bucket.png
# ============================================================

def fig1_coverage_by_bucket():
    buckets = ["<£85", "£85–120", "£120–150", "£150–250"]
    wf_cov   = [74.6, 88.5, 81.1, 49.1]
    live_cov = [66.6, 85.9, 47.6, 34.1]
    live_ci_lo = [62.9, 83.1, 43.4, 26.4]
    live_ci_hi = [70.1, 88.3, 51.7, 42.8]
    target = 79.8

    x = np.arange(len(buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars_wf = ax.bar(x - width/2, wf_cov, width, label="Walk-Forward (119d)",
                     color="#4878CF", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_live = ax.bar(x + width/2, live_cov, width, label="Live (30 days, May–Jun 2026)",
                       color="#6ACC65", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Error bars for live 90% CI
    err_lo = [live_cov[i] - live_ci_lo[i] for i in range(len(buckets))]
    err_hi = [live_ci_hi[i] - live_cov[i] for i in range(len(buckets))]
    ax.errorbar(x + width/2, live_cov, yerr=[err_lo, err_hi],
                fmt="none", color="black", capsize=5, capthick=1.5, linewidth=1.5,
                label="Live 90% CI")

    # Target line
    ax.axhline(target, color="#D65F5F", linestyle="--", linewidth=1.8,
               label=f"WF target {target}%")

    # Labels on bars
    for rect, val in zip(bars_wf, wf_cov):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#333333")
    for rect, val in zip(bars_live, live_cov):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(buckets, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("PI Coverage (%)", fontsize=12)
    ax.set_xlabel("Price Bucket (£/MWh)", fontsize=12)
    ax.set_title("PI Coverage by Price Bucket: Walk-Forward vs Live (30 days)", fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="upper right")

    ax.text(0.01, 0.01,
            "£250+ bucket omitted — N=0 live rows",
            transform=ax.transAxes, fontsize=8, color="#666666",
            va="bottom", ha="left")

    fig.tight_layout()
    out_path = f"{OUT_DIR}/coverage_by_bucket.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Figure 2: persistence_vs_da_crossover.png
# ============================================================

def fig2_persistence_vs_da():
    horizons = list(range(1, 13))
    pers_mae = [16.38, 22.14, 26.29, 29.21, 31.28, 33.18, 34.86, 36.13, 37.06, 37.83, 38.63, 39.40]
    da_mae   = [31.0] * 12  # flat provisional

    fig, ax = plt.subplots(figsize=(10, 6))

    # Shaded regions
    ax.axvspan(1, 4.5, alpha=0.12, color="#2ca02c", label="Persistence wins (h+1 to h+4)")
    ax.axvspan(4.5, 12, alpha=0.10, color="#ff7f0e", label="DA+Kalman wins (h+5+)")

    # Lines
    ax.plot(horizons, pers_mae, "o-", color="#1f77b4", linewidth=2.2, markersize=6,
            label="Persistence MAE", zorder=3)
    ax.plot(horizons, da_mae, "s--", color="#d62728", linewidth=2.2, markersize=6,
            label="DA+Kalman MAE (provisional, spring/summer 2026)", zorder=3)

    # Crossover line
    ax.axvline(4.5, color="#555555", linestyle=":", linewidth=1.8)
    ax.text(4.5 + 0.1, 17, "crossover h+4.5", rotation=90,
            va="bottom", ha="left", fontsize=9.5, color="#555555")

    # Annotation
    ax.annotate("DA+Kalman wins by h+5 (−5.7%)",
                xy=(5, 31.28), xytext=(6.2, 25.5),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3),
                fontsize=9.5, color="#555555")

    ax.set_xticks(horizons)
    ax.set_xticklabels([f"h+{h}" for h in horizons], fontsize=10)
    ax.set_ylim(10, 45)
    ax.set_ylabel("MAE (£/MWh)", fontsize=12)
    ax.set_xlabel("Horizon", fontsize=12)
    ax.set_title("Persistence vs DA+Kalman MAE by Horizon", fontsize=13, pad=12)
    ax.legend(fontsize=10, loc="lower right")

    # Footnote
    ax.text(0.01, 0.01,
            "DA+Kalman MAE provisional (30 days, May–Jun 2026). Persistence MAE from 2024+ data (n≈43,000 SP pairs).",
            transform=ax.transAxes, fontsize=7.5, color="#666666",
            va="bottom", ha="left")

    fig.tight_layout()
    out_path = f"{OUT_DIR}/persistence_vs_da_crossover.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Figure 3: nowcast_bands.png
# ============================================================

def fig3_nowcast_bands():
    # Read nowcast_bands.json
    json_path = "/Users/diliu/Library/CloudStorage/Dropbox/DS_projects/uk-system-price-forecast/model_assets/nowcast_bands.json"
    with open(json_path) as f:
        data = json.load(f)

    horizons = ["h1", "h2", "h3"]
    horizon_labels = ["h+1 (≈30 min ahead)", "h+2 (≈60 min ahead)", "h+3 (≈90 min ahead)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

    for ax, h, hlabel in zip(axes, horizons, horizon_labels):
        np_p10 = data["horizons"][h]["NP"]["p10"]
        np_p90 = data["horizons"][h]["NP"]["p90"]
        en_p10 = data["horizons"][h]["EN"]["p10"]
        en_p90 = data["horizons"][h]["EN"]["p90"]

        # Draw horizontal interval bars for each regime
        # NP regime (blue, positive-skewed)
        ax.barh(1, np_p90 - np_p10, left=np_p10, height=0.4, color="#4878CF",
                alpha=0.8, label="NP regime (N-code)", zorder=3)
        # EN regime (orange, negative-skewed)
        ax.barh(0, en_p90 - en_p10, left=en_p10, height=0.4, color="#E87722",
                alpha=0.8, label="EN regime (P/K-code)", zorder=3)

        # Mark P10 and P90 with text
        ax.text(np_p10 - 1, 1, f"P10\n{np_p10:.1f}£", ha="right", va="center",
                fontsize=7.5, color="#4878CF")
        ax.text(np_p90 + 1, 1, f"P90\n+{np_p90:.1f}£", ha="left", va="center",
                fontsize=7.5, color="#4878CF")
        ax.text(en_p10 - 1, 0, f"P10\n{en_p10:.1f}£", ha="right", va="center",
                fontsize=7.5, color="#E87722")
        ax.text(en_p90 + 1, 0, f"P90\n+{en_p90:.1f}£", ha="left", va="center",
                fontsize=7.5, color="#E87722")

        # Asymmetry annotation
        np_asym = np_p90 + abs(np_p10)
        np_mid = (np_p90 + np_p10) / 2
        en_asym = abs(en_p10) + en_p90
        en_mid = (en_p10 + en_p90) / 2
        ax.text(np_mid, 1.28,
                f"width £{np_asym:.0f}\nright-skewed (spike risk)",
                ha="center", va="bottom", fontsize=7, color="#4878CF",
                style="italic")
        ax.text(en_mid, -0.28,
                f"width £{en_asym:.0f}\nleft-skewed (formula floor)",
                ha="center", va="top", fontsize=7, color="#E87722",
                style="italic")

        # Vertical line at 0 (persistence point)
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.5, zorder=4)
        ax.text(0, 1.62, "0\n(persistence)", ha="center", va="bottom",
                fontsize=7.5, color="#333333")

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["EN\n(P/K-code)", "NP\n(N-code)"], fontsize=9)
        ax.set_ylim(-0.6, 1.9)
        ax.set_xlabel("Price change from persistence (£/MWh)", fontsize=9)
        ax.set_title(hlabel, fontsize=10, pad=6)
        ax.grid(axis="x", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Regime", fontsize=10)

    # Legend
    np_patch = mpatches.Patch(color="#4878CF", alpha=0.8, label="NP regime (N-code, upside spike risk)")
    en_patch = mpatches.Patch(color="#E87722", alpha=0.8, label="EN regime (P/K-code, formula, downside risk)")
    fig.legend(handles=[np_patch, en_patch], fontsize=9, loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, 0.0), framealpha=0.95)

    fig.suptitle("Intraday Nowcast — 80% Empirical Prediction Bands by Horizon and Regime",
                 fontsize=12, y=1.01)

    fig.text(0.5, -0.05,
             "NP = N-code (normal auction, upside spike risk); EN = P/K-code (formula price, downside risk).\n"
             "Bands fit on 18-month residuals (2024-12-17 → 2026-06-17, 26,298 SP pairs). "
             "Live coverage: h+1=79.5%, h+2=79.8%, h+3=79.4%.",
             ha="center", va="top", fontsize=7.5, color="#555555")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out_path = f"{OUT_DIR}/nowcast_bands.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Generating Figure 1: coverage_by_bucket.png ...")
    fig1_coverage_by_bucket()

    print("Generating Figure 2: persistence_vs_da_crossover.png ...")
    fig2_persistence_vs_da()

    print("Generating Figure 3: nowcast_bands.png ...")
    fig3_nowcast_bands()

    print("All figures generated successfully.")

"""
UK Electricity System Price Dashboard
Data source: Elexon BMRS — data/raw/system_prices.csv
Run: streamlit run src/dashboard/streamlit_app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
# Use full processed history for analytics; system_prices.csv is only a ~50-day rolling window
DATA_PATH     = ROOT / "data" / "processed" / "dataset_5yr.csv"
PRED_PATH_P3      = ROOT / "model_assets" / "test_predictions_phase3.csv"
FORECAST_PATH_P3  = ROOT / "model_assets" / "next_day_forecast_phase3.csv"   # H+1 today
FORECAST_PATH_H2  = ROOT / "model_assets" / "day2_forecast_phase3.csv"       # H+2 tomorrow
METRICS_P3       = ROOT / "model_assets" / "phase3_metrics.json"
LEVEL_FEAT_JSON  = ROOT / "model_assets" / "level_feature_cols.json"
SHAPE_FEAT_JSON  = ROOT / "model_assets" / "shape_feature_cols.json"
LEVEL_IMP_CSV    = ROOT / "model_assets" / "phase3_level_importance.csv"
SHAPE_IMP_CSV    = ROOT / "model_assets" / "phase3_shape_importance.csv"

FETCH_WEATHER     = ROOT / "src" / "data"    / "fetch_weather.py"
FETCH_GENERATION  = ROOT / "src" / "data"    / "fetch_generation.py"
FETCH_CPI         = ROOT / "src" / "data"    / "fetch_cpi.py"
EXTEND_DATASET    = ROOT / "src" / "data"    / "extend_dataset.py"
BUILD_FEATURES    = ROOT / "src" / "features"/ "build_features.py"
TRAIN_PHASE3      = ROOT / "src" / "models"  / "train_phase3.py"
FORECAST_SCRIPT_P3 = ROOT / "src" / "models" / "forecast_phase3.py"
FORECASTS_DIR     = ROOT / "model_assets"    / "forecasts"
DATASET_5YR       = ROOT / "data" / "processed" / "dataset_5yr.csv"
FEATURES_5YR      = ROOT / "data" / "processed" / "features_5yr.csv"

st.set_page_config(
    page_title="UK System Price Dashboard",
    page_icon="⚡",
    layout="wide",
)

# Updated by CI pipeline on each daily run — forces Streamlit Cloud to redeploy
_LAST_PIPELINE_RUN = "2026-06-14T15:23"


@st.cache_data(ttl=7200)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["settlement_date"])
    df["settlement_period"] = df["settlement_period"].astype(int)
    # Convert settlement period (1–48) to a clock time label
    df["time_label"] = pd.to_datetime(
        (df["settlement_period"] - 1) * 30 * 60, unit="s"
    ).dt.strftime("%H:%M")
    df["datetime"] = df["settlement_date"] + pd.to_timedelta(
        (df["settlement_period"] - 1) * 30, unit="min"
    )
    return df.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)


df = load_data()


@st.cache_data(ttl=7200)
def load_p3_metrics() -> dict:
    import json
    if METRICS_P3.exists():
        with open(METRICS_P3) as f:
            return json.load(f)
    return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Filters")

st.sidebar.divider()
_latest_date = df["settlement_date"].max().strftime("%Y-%m-%d")
st.sidebar.caption(f"📅 Latest data: **{_latest_date}**\n\nPipeline runs automatically at 12:30 UTC daily after Elexon publishes settlement prices.")
st.sidebar.divider()

min_date = df["settlement_date"].min().date()
max_date = df["settlement_date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

spike_threshold = st.sidebar.number_input(
    "Spike threshold (£/MWh)",
    min_value=0,
    max_value=5000,
    value=200,
    step=10,
    help="Highlight periods where SSP or SBP exceeds this value.",
)

mask = (df["settlement_date"].dt.date >= start_date) & (df["settlement_date"].dt.date <= end_date)
dff = df[mask].copy()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("UK Electricity System Price Dashboard")
st.markdown("**Di Liu** · [github.com/liudi23/uk-system-price-forecast](https://github.com/liudi23/uk-system-price-forecast)")
st.caption(
    f"Data: Elexon BMRS · {start_date} → {end_date} · "
    f"{len(dff):,} settlement periods"
)

# ── Production model banner ───────────────────────────────────────────────────
_m = load_p3_metrics()
_p3 = _m.get("Phase3_P50_two_stage", {})
_dc = _m.get("decomposition", {})
_mae_str   = f"£{_p3['MAE']:.2f}/MWh"        if "MAE"           in _p3 else "—"
_lvl_str   = f"£{_dc['level_mae']:.2f}/MWh/day" if "level_mae"  in _dc else "—"
_corr_str  = f"{_dc['shape_corr_mean']:.3f}"  if "shape_corr_mean" in _dc else "—"
_peak_str  = f"±{_dc['peak_timing_mae']:.1f} SPs" if "peak_timing_mae" in _dc else "—"
st.info(
    "**Phase 3 — Level-Shape Decomposition · CPI-adjusted · 3-year rolling window** · "
    "Stage 1: daily level HGBR (P10/P50/P90) · Stage 2: intra-day shape HGBR · "
    "All shape features lag ≥ 48 SPs — zero leakage · "
    f"**Honest MAE: {_mae_str}** · Level MAE: {_lvl_str} · "
    f"Shape corr: {_corr_str} · Peak timing: {_peak_str}"
)

# ── Next-day forecast panel ───────────────────────────────────────────────────
st.subheader("Day-Ahead Forecast (Phase 3 Level-Shape · P10 / P50 / P90 · 48 Settlement Periods)")

_fc_path = FORECAST_PATH_P3

if _fc_path.exists():
    fc = pd.read_csv(_fc_path, parse_dates=["settlement_datetime"])
    fc_date  = fc["settlement_date"].iloc[0]
    fc_label = f"{fc_date}  ·  SP 1–48 (midnight → 23:30)"

    has_quantiles = "ssp_q50" in fc.columns
    p50_col = "ssp_q50" if has_quantiles else "ssp_predicted"

    fm1, fm2, fm3, fm4, fm5 = st.columns(5)
    fm1.metric("Forecast date", fc_date)
    fm2.metric("Min P50", f"£{fc[p50_col].min():.1f}")
    fm3.metric("Avg P50 (daily level)", f"£{fc[p50_col].mean():.1f}")
    fm4.metric("Max P50", f"£{fc[p50_col].max():.1f}")
    if "pred_daily_level" in fc.columns:
        fm5.metric("Predicted daily level", f"£{fc['pred_daily_level'].iloc[0]:.1f}/MWh",
                   help="Stage 1 level model prediction — expected daily average price")
    elif has_quantiles and "spike_prob" in fc.columns:
        peak_sp  = int(fc.loc[fc["ssp_q90"].idxmax(), "settlement_period"])
        peak_q90 = fc["ssp_q90"].max()
        fm5.metric("Peak P90 risk", f"£{peak_q90:.0f}  SP {peak_sp}")
    else:
        fm5.metric("Max P50", f"£{fc[p50_col].max():.1f}")

    # Split by is_actual flag (set by forecast_phase3.py intraday post-processing)
    _has_actual = "is_actual" in fc.columns and fc["is_actual"].any()
    fc_actual    = fc[fc["is_actual"]]   if _has_actual else pd.DataFrame()
    fc_remaining = fc[~fc["is_actual"]]  if _has_actual else fc

    # Orange/yellow boundary on remaining (forecast) SPs: now + 2 h
    _now_utc   = pd.Timestamp.utcnow().tz_localize(None)
    _cutoff_sp = min(int((_now_utc.hour * 60 + _now_utc.minute) / 30) + 1 + 4, 48)
    fc_orange  = fc_remaining[fc_remaining["settlement_period"] <= _cutoff_sp]
    fc_yellow  = fc_remaining[fc_remaining["settlement_period"] >  _cutoff_sp]
    _boundary  = fc_remaining[fc_remaining["settlement_period"] == _cutoff_sp]

    fig_fc = go.Figure()

    if has_quantiles and not fc_remaining.empty:
        # P10–P90 band — only over the forecast (non-actual) SPs
        _rem = fc_remaining
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([_rem["settlement_datetime"], _rem["settlement_datetime"].iloc[::-1]]),
            y=pd.concat([_rem["ssp_q90"], _rem["ssp_q10"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.12)",
            line=dict(color="rgba(255,127,14,0)"),
            hoverinfo="skip", name="P10–P90 band", showlegend=True,
        ))
        fig_fc.add_trace(go.Scatter(
            x=_rem["settlement_datetime"], y=_rem["ssp_q90"],
            name="P90 (spike risk)", line=dict(color="#ff7f0e", width=1, dash="dot"),
            hovertemplate="SP %{customdata}<br>P90 £%{y:.2f}<extra></extra>",
            customdata=_rem["settlement_period"],
        ))
        fig_fc.add_trace(go.Scatter(
            x=_rem["settlement_datetime"], y=_rem["ssp_q10"],
            name="P10 (downside)", line=dict(color="#ff7f0e", width=1, dash="dot"),
            hovertemplate="SP %{customdata}<br>P10 £%{y:.2f}<extra></extra>",
            customdata=_rem["settlement_period"],
        ))

    # Actual settled prices — dark solid line
    if not fc_actual.empty:
        fig_fc.add_trace(go.Scatter(
            x=fc_actual["settlement_datetime"], y=fc_actual[p50_col],
            name="Actual SSP", line=dict(color="#c0392b", width=2.5),
            hovertemplate="SP %{customdata}<br>Actual £%{y:.2f}<extra></extra>",
            customdata=fc_actual["settlement_period"],
        ))

    # P50 — orange segment (near-term forecast ≤ now+2h, shape-corrected)
    if not fc_orange.empty:
        # Connect from last actual if available
        _x_join = pd.concat([fc_actual["settlement_datetime"].iloc[[-1]], fc_orange["settlement_datetime"]]) if not fc_actual.empty else fc_orange["settlement_datetime"]
        _y_join = pd.concat([fc_actual[p50_col].iloc[[-1]], fc_orange[p50_col]]) if not fc_actual.empty else fc_orange[p50_col]
        _sp_join = pd.concat([fc_actual["settlement_period"].iloc[[-1]], fc_orange["settlement_period"]]) if not fc_actual.empty else fc_orange["settlement_period"]
        fig_fc.add_trace(go.Scatter(
            x=_x_join, y=_y_join,
            name="P50 (near-term)", line=dict(color="#e05c00", width=2.5),
            hovertemplate="SP %{customdata}<br>P50 £%{y:.2f}<extra></extra>",
            customdata=_sp_join,
        ))

    # P50 — yellow segment (forecast horizon > now+2h, shape-corrected)
    if not fc_yellow.empty:
        _x_y = pd.concat([_boundary["settlement_datetime"], fc_yellow["settlement_datetime"]])
        _y_y = pd.concat([_boundary[p50_col],              fc_yellow[p50_col]])
        _sp_y = pd.concat([_boundary["settlement_period"],  fc_yellow["settlement_period"]])
        fig_fc.add_trace(go.Scatter(
            x=_x_y, y=_y_y,
            name="P50 (forecast)", line=dict(color="#f5a623", width=2.5),
            hovertemplate="SP %{customdata}<br>P50 £%{y:.2f}<extra></extra>",
            customdata=_sp_y,
        ))

    # Vertical marker at now+2h boundary
    if not fc_orange.empty and not fc_yellow.empty:
        _bdt = _boundary["settlement_datetime"].iloc[0]
        fig_fc.add_vline(
            x=_bdt.value / 1e6, line_dash="dot", line_color="grey", line_width=1,
            annotation_text="now + 2h", annotation_position="top",
            annotation_font=dict(size=10, color="grey"),
        )

    _legend_note = "red = actual · orange = near-term forecast · yellow = forecast horizon"
    fig_fc.update_layout(
        xaxis_title="Datetime", yaxis_title="£/MWh",
        height=340, margin=dict(t=10, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[dict(
            text=f"Forecast: {fc_label}  ·  {_legend_note}",
            xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
            font=dict(size=11, color="grey"),
        )],
    )
    st.plotly_chart(fig_fc)


else:
    st.info(
        "No forecast found. Run `python src/models/forecast_phase3.py` to generate one."
    )

# ── H+2 forecast panel (tomorrow) ────────────────────────────────────────────
if FORECAST_PATH_H2.exists():
    fc_h2  = pd.read_csv(FORECAST_PATH_H2, parse_dates=["settlement_datetime"])
    h2_date = fc_h2["settlement_date"].iloc[0]
    h2_p50  = "ssp_q50" if "ssp_q50" in fc_h2.columns else "ssp_predicted"
    h2_lvl  = fc_h2["pred_daily_level"].iloc[0] if "pred_daily_level" in fc_h2.columns else fc_h2[h2_p50].mean()

    st.subheader(f"Tomorrow Forecast · H+2 · {h2_date}  (daily level P50 = £{h2_lvl:.1f}/MWh)")
    st.caption("Two-day-ahead forecast using lag-96+ features only — lag-48 (today's prices) not yet settled.")

    h2m1, h2m2, h2m3 = st.columns(3)
    h2m1.metric("H+2 Date", h2_date)
    h2m2.metric("Level P50", f"£{h2_lvl:.1f}/MWh")
    h2m3.metric("Peak P50", f"£{fc_h2[h2_p50].max():.1f}  SP{int(fc_h2.loc[fc_h2[h2_p50].idxmax(),'settlement_period'])}")

    fig_h2 = go.Figure()
    if "ssp_q10" in fc_h2.columns:
        fig_h2.add_trace(go.Scatter(
            x=pd.concat([fc_h2["settlement_datetime"], fc_h2["settlement_datetime"].iloc[::-1]]),
            y=pd.concat([fc_h2["ssp_q90"], fc_h2["ssp_q10"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(42,157,143,0.15)",
            line=dict(color="rgba(42,157,143,0)"), hoverinfo="skip", name="P10–P90",
        ))
    fig_h2.add_trace(go.Scatter(
        x=fc_h2["settlement_datetime"], y=fc_h2[h2_p50],
        name="H+2 P50", line=dict(color="#2a9d8f", width=2.5),
        hovertemplate="SP %{customdata}<br>£%{y:.2f}<extra></extra>",
        customdata=fc_h2["settlement_period"],
    ))
    fig_h2.add_hline(y=h2_lvl, line_dash="dot", line_color="#e07b39", line_width=1.2,
                     annotation_text=f"Level P50 £{h2_lvl:.0f}", annotation_position="top left")
    fig_h2.update_layout(xaxis_title="Datetime", yaxis_title="£/MWh",
                          height=300, margin=dict(t=10, b=40), hovermode="x unified",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_h2)

st.divider()

# ── Live Forecast Verification ────────────────────────────────────────────────
st.subheader("Live Forecast Verification")
st.caption(
    "Compares each archived day-ahead forecast against the Elexon actuals once they are available. "
    "Forecast is generated automatically each day at 12:30 UTC."
)

archived = sorted(FORECASTS_DIR.glob("forecast_*.csv")) if FORECASTS_DIR.exists() else []

# Separate Phase 2 and Phase 3 archives; prefer Phase 3 for each date
def _archive_date(f):
    stem = f.stem  # "forecast_2026-05-18" or "forecast_phase3_2026-05-18"
    return stem.replace("forecast_phase3_", "").replace("forecast_", "")

if not archived:
    st.info("No archived forecasts yet. Run the forecast once to start tracking.")
else:
    actual_dates = set(df["settlement_date"].dt.strftime("%Y-%m-%d").unique())
    # Build date → file mapping, preferring phase3 archives
    date_to_file: dict = {}
    for f in archived:
        d = _archive_date(f)
        if d not in date_to_file or "phase3" in f.stem:
            date_to_file[d] = f

    verified_dates = [d for d in date_to_file if d in actual_dates]
    pending_dates  = [d for d in date_to_file if d not in actual_dates]

    if pending_dates:
        st.info(
            f"Forecast for **{', '.join(pending_dates)}** is waiting for actuals — "
            "refresh tomorrow once Elexon publishes the settlement prices."
        )

    if not verified_dates:
        st.warning("No forecast dates have actuals available yet for comparison.")
    else:
        sel_date = st.selectbox(
            "Select date to verify",
            options=sorted(verified_dates, reverse=True),
            format_func=lambda d: f"{d}  (actuals available)",
        )

        fc_v = pd.read_csv(
            date_to_file[sel_date],
            parse_dates=["settlement_datetime"],
        )
        act_v = (
            df[df["settlement_date"].dt.strftime("%Y-%m-%d") == sel_date]
            [["settlement_period", "ssp"]]
            .rename(columns={"ssp": "ssp_actual"})
        )
        merged = fc_v.merge(act_v, on="settlement_period", how="inner")
        merged["error"]     = merged["ssp_predicted"] - merged["ssp_actual"]
        merged["abs_error"] = merged["error"].abs()

        # Metrics
        v_mae   = merged["abs_error"].mean()
        v_rmse  = (merged["error"] ** 2).mean() ** 0.5
        v_max   = merged["abs_error"].max()
        worst_sp = int(merged.loc[merged["abs_error"].idxmax(), "settlement_period"])
        denom   = (merged["ssp_actual"].abs() + merged["ssp_predicted"].abs()) / 2
        v_smape = (merged["abs_error"] / denom.replace(0, float("nan"))).mean() * 100

        vc1, vc2, vc3, vc4, vc5 = st.columns(5)
        vc1.metric("Date verified", sel_date)
        vc2.metric("MAE", f"£{v_mae:.2f}/MWh")
        vc3.metric("RMSE", f"£{v_rmse:.2f}/MWh")
        vc4.metric("sMAPE", f"{v_smape:.1f}%")
        vc5.metric("Worst SP", f"SP {worst_sp}  (£{v_max:.1f} off)")

        # Actual vs predicted time series
        fig_v = go.Figure()
        _has_quantiles = "ssp_q10" in merged.columns and "ssp_q90" in merged.columns
        if _has_quantiles:
            fig_v.add_trace(go.Scatter(
                x=merged["settlement_datetime"], y=merged["ssp_q90"],
                name="P90", line=dict(color="rgba(0,0,0,0)"),
                hovertemplate="SP %{customdata}<br>P90 £%{y:.2f}<extra></extra>",
                customdata=merged["settlement_period"],
                showlegend=False,
            ))
            fig_v.add_trace(go.Scatter(
                x=merged["settlement_datetime"], y=merged["ssp_q10"],
                name="P10–P90 band",
                fill="tonexty",
                fillcolor="rgba(255,127,14,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hovertemplate="SP %{customdata}<br>P10 £%{y:.2f}<extra></extra>",
                customdata=merged["settlement_period"],
            ))
        fig_v.add_trace(go.Scatter(
            x=merged["settlement_datetime"], y=merged["ssp_actual"],
            name="Actual SSP", line=dict(color="#1f77b4", width=2),
            hovertemplate="SP %{customdata}<br>Actual £%{y:.2f}<extra></extra>",
            customdata=merged["settlement_period"],
        ))
        fig_v.add_trace(go.Scatter(
            x=merged["settlement_datetime"], y=merged["ssp_predicted"],
            name="Forecast P50", line=dict(color="#ff7f0e", width=2, dash="dot"),
            hovertemplate="SP %{customdata}<br>Forecast £%{y:.2f}<extra></extra>",
            customdata=merged["settlement_period"],
        ))
        fig_v.update_layout(
            xaxis_title="Datetime", yaxis_title="£/MWh",
            height=340, margin=dict(t=10, b=40), hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_v)

        # Error by settlement period + error histogram side by side
        col_ep, col_eh = st.columns(2)

        with col_ep:
            fig_ep = go.Figure(go.Bar(
                x=merged["settlement_period"],
                y=merged["abs_error"],
                marker_color=merged["abs_error"].apply(
                    lambda e: "#d62728" if e > v_mae * 2 else "#1f77b4"
                ),
                hovertemplate="SP %{x}<br>Error £%{y:.2f}<extra></extra>",
            ))
            fig_ep.add_hline(
                y=v_mae, line_dash="dash", line_color="orange",
                annotation_text=f"MAE £{v_mae:.1f}",
                annotation_position="top left",
            )
            fig_ep.update_layout(
                xaxis_title="Settlement Period", yaxis_title="Absolute Error (£/MWh)",
                height=300, margin=dict(t=30, b=40),
                title_text="Error by Settlement Period", title_x=0,
                showlegend=False,
            )
            st.plotly_chart(fig_ep)

        with col_eh:
            fig_eh = px.histogram(
                merged, x="error", nbins=24,
                color_discrete_sequence=["#ff7f0e"],
                labels={"error": "Forecast Error (£/MWh)", "count": "Periods"},
            )
            fig_eh.add_vline(x=0, line_dash="dash", line_color="black")
            fig_eh.update_layout(
                height=300, margin=dict(t=30, b=40),
                title_text="Error Distribution (Predicted − Actual)", title_x=0,
            )
            st.plotly_chart(fig_eh)

st.divider()

# ── KPI row ───────────────────────────────────────────────────────────────────
latest = dff.sort_values("datetime").iloc[-1]
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Latest SSP (£/MWh)", f"£{latest['ssp']:.2f}")
col2.metric("Avg SSP", f"£{dff['ssp'].mean():.2f}")
col3.metric("Min SSP", f"£{dff['ssp'].min():.2f}")
col4.metric("Max SSP", f"£{dff['ssp'].max():.2f}")
spike_count = (dff["ssp"] > spike_threshold).sum()
col5.metric(f"Spikes (>{spike_threshold})", int(spike_count))

st.divider()

# ── SSP / SBP time series ─────────────────────────────────────────────────────
st.subheader("System Sell Price (SSP)")

daily = (
    dff.groupby("settlement_date")["ssp"]
    .mean()
    .reset_index()
    .rename(columns={"ssp": "Avg SSP"})
)

fig_ts = go.Figure()
fig_ts.add_trace(
    go.Scatter(
        x=daily["settlement_date"], y=daily["Avg SSP"],
        name="Avg SSP", line=dict(color="#1f77b4", width=2),
    )
)
fig_ts.add_hline(
    y=spike_threshold, line_dash="dash", line_color="red",
    annotation_text=f"Spike threshold £{spike_threshold}",
    annotation_position="top left",
)
fig_ts.update_layout(
    xaxis_title="Date",
    yaxis_title="£/MWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=380,
    margin=dict(t=10, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_ts)

# ── Heatmap + Imbalance side by side ─────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    # Heatmap: cap at 90 days — column names must include year so Plotly
    # doesn't misinterpret bare "MM-DD" strings as year 2003/2004 dates.
    # Fill remaining NaN (DST days with ≠48 SPs) via forward-fill so no gaps.
    _hm_days  = 90
    _hm_end   = dff["settlement_date"].max()
    _hm_start = _hm_end - pd.Timedelta(days=_hm_days - 1)
    dff_hm = dff[dff["settlement_date"] >= _hm_start]
    st.subheader(f"Daily SSP Heatmap by Settlement Period (last {_hm_days} days)")

    pivot = dff_hm.pivot_table(
        index="settlement_period", columns="settlement_date", values="ssp", aggfunc="mean"
    )
    # Keep full YYYY-MM-DD label → unique, correctly ordered, no year ambiguity
    pivot.columns = pivot.columns.strftime("%Y-%m-%d")
    # Forward-fill rare NaN cells (DST 46/50-SP days) to remove visual gaps
    pivot = pivot.ffill(axis=0).bfill(axis=0)

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels=dict(x="Date", y="Settlement Period", color="SSP £/MWh"),
    )
    fig_heat.update_layout(height=420, margin=dict(t=10, b=40))
    fig_heat.update_xaxes(tickangle=45, nticks=20)
    st.plotly_chart(fig_heat)

with col_right:
    st.subheader("Net Imbalance Volume")
    daily_niv = (
        dff.groupby("settlement_date")["net_imbalance_volume"]
        .mean()
        .reset_index()
    )
    colors = daily_niv["net_imbalance_volume"].apply(
        lambda v: "#d62728" if v < 0 else "#2ca02c"
    )
    fig_niv = go.Figure(
        go.Bar(
            x=daily_niv["settlement_date"],
            y=daily_niv["net_imbalance_volume"],
            marker_color=colors,
            name="Avg NIV",
        )
    )
    fig_niv.update_layout(
        xaxis_title="Date",
        yaxis_title="MWh",
        height=420,
        margin=dict(t=10, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_niv)

# ── Average daily profile ────────────────────────────────────────────────────
st.subheader("Average Settlement Period Profile (SSP)")

col_sp, col_wk = st.columns(2)

with col_sp:
    _sp_dff = dff[dff["settlement_period"] <= 48]
    profile = (
        _sp_dff
        .groupby("settlement_period")[["ssp", "time_label"]]
        .agg({"ssp": "mean", "time_label": "first"})
        .reset_index()
        .sort_values("settlement_period")
    )
    pn_profile = (
        _sp_dff.groupby("settlement_period")["price_derivation_code"]
        .apply(lambda x: (x == "P").mean() * 100)
        .reset_index()
        .rename(columns={"price_derivation_code": "pct_P"})
    )
    profile = profile.merge(pn_profile, on="settlement_period")

    # Use integer settlement_period on x-axis (avoids Plotly misinterpreting
    # HH:MM strings as dates, which caused a diagonal fill artifact).
    _sp_ticks = profile["settlement_period"].tolist()[::4]
    _lb_ticks  = profile["time_label"].tolist()[::4]

    fig_profile = go.Figure()
    # SSP fill + line (left y-axis)
    fig_profile.add_trace(go.Scatter(
        x=profile["settlement_period"].tolist() + profile["settlement_period"].tolist()[::-1],
        y=profile["ssp"].tolist() + [0] * len(profile),
        fill="toself",
        fillcolor="rgba(31,119,180,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
        yaxis="y1",
    ))
    fig_profile.add_trace(go.Scatter(
        x=profile["settlement_period"], y=profile["ssp"],
        name="Avg SSP",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="SP %{x}  %{customdata}<br>SSP £%{y:.2f}/MWh<extra></extra>",
        customdata=profile["time_label"],
        yaxis="y1",
    ))
    # % P code (right y-axis)
    fig_profile.add_trace(go.Scatter(
        x=profile["settlement_period"], y=profile["pct_P"],
        name="% P code",
        line=dict(color="#d62728", width=1.5, dash="dot"),
        hovertemplate="SP %{x}<br>P code: %{y:.0f}%<extra></extra>",
        yaxis="y2",
    ))
    fig_profile.add_trace(go.Scatter(
        x=profile["settlement_period"], y=100 - profile["pct_P"],
        name="% N code",
        line=dict(color="#2ca02c", width=1.5, dash="dot"),
        hovertemplate="SP %{x}<br>N code: %{y:.0f}%<extra></extra>",
        yaxis="y2",
    ))
    fig_profile.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=_sp_ticks, ticktext=_lb_ticks,
            title="Time of Day (HH:MM)",
        ),
        yaxis=dict(title="£/MWh", rangemode="tozero"),
        yaxis2=dict(
            title="% of periods",
            overlaying="y", side="right",
            range=[0, 100], ticksuffix="%",
            showgrid=False,
        ),
        height=320,
        margin=dict(t=30, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        title=dict(text="Intra-day profile (selected range)", font=dict(size=13)),
    )
    st.plotly_chart(fig_profile)

with col_wk:
    # Week-of-year profile: last 3 years, ISO weeks 1–52, averaged across years
    _wk_end   = df["settlement_date"].max()
    _wk_start = _wk_end - pd.Timedelta(days=3 * 365)
    df_3yr = df[df["settlement_date"] >= _wk_start].copy()
    df_3yr["week"] = df_3yr["settlement_date"].dt.isocalendar().week.astype(int)
    df_3yr = df_3yr[df_3yr["week"] <= 52]   # drop rare week-53 partial ISO weeks

    week_profile = (
        df_3yr.groupby("week")["ssp"]
        .mean()
        .reset_index()
        .sort_values("week")
    )
    pn_week_profile = (
        df_3yr[df_3yr["settlement_period"] <= 48]
        .groupby("week")["price_derivation_code"]
        .apply(lambda x: (x == "P").mean() * 100)
        .reset_index()
        .rename(columns={"price_derivation_code": "pct_P"})
    )
    week_profile = week_profile.merge(pn_week_profile, on="week")

    fig_week = go.Figure()
    fig_week.add_trace(go.Scatter(
        x=week_profile["week"].tolist() + week_profile["week"].tolist()[::-1],
        y=week_profile["ssp"].tolist() + [0] * len(week_profile),
        fill="toself",
        fillcolor="rgba(42,157,143,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig_week.add_trace(go.Scatter(
        x=week_profile["week"], y=week_profile["ssp"],
        name="Avg SSP",
        line=dict(color="#2a9d8f", width=2),
        hovertemplate="Week %{x}<br>£%{y:.2f}/MWh<extra></extra>",
    ))
    fig_week.add_trace(go.Scatter(
        x=week_profile["week"], y=week_profile["pct_P"],
        name="% P code",
        line=dict(color="#d62728", width=1.5, dash="dot"),
        hovertemplate="Week %{x}<br>P code: %{y:.0f}%<extra></extra>",
        yaxis="y2",
    ))
    fig_week.add_trace(go.Scatter(
        x=week_profile["week"], y=100 - week_profile["pct_P"],
        name="% N code",
        line=dict(color="#2ca02c", width=1.5, dash="dot"),
        hovertemplate="Week %{x}<br>N code: %{y:.0f}%<extra></extra>",
        yaxis="y2",
    ))
    # Season band annotations
    for x0, x1, label in [(1, 13, "Winter"), (14, 26, "Spring"),
                           (27, 39, "Summer"), (40, 52, "Autumn")]:
        fig_week.add_vrect(x0=x0, x1=x1, fillcolor="rgba(0,0,0,0.03)",
                           layer="below", line_width=0,
                           annotation_text=label, annotation_position="top left",
                           annotation_font_size=9, annotation_font_color="#888")
    fig_week.update_layout(
        xaxis=dict(title="ISO Week of Year", tickmode="array",
                   tickvals=list(range(1, 53, 4)),
                   ticktext=[str(w) for w in range(1, 53, 4)]),
        yaxis=dict(title="£/MWh", rangemode="tozero"),
        yaxis2=dict(
            title="% of periods",
            overlaying="y", side="right",
            range=[0, 100], ticksuffix="%",
            showgrid=False,
        ),
        height=320,
        margin=dict(t=30, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        title=dict(text="Seasonal profile (last 3 years, weeks averaged)", font=dict(size=13)),
    )
    st.plotly_chart(fig_week)

# ── Price Derivation Code ────────────────────────────────────────────────────
st.subheader("Price Derivation Code (P vs N)")
st.caption(
    "**N** — genuine market auction price; **P** — formula fallback when the auction is unreliable (~50% each). "
    "N is more common overnight (simple dispatch); P dominates evening peaks (complex multi-generator dispatch). "
    "Bar height = count of periods per code per day (max 48). K is a rare special case (9 times in 5 years)."
)

# Full-width: daily P/N period counts over time (capped at SP 1-48; DST days have 50 SPs)
daily_pdc = (
    dff[dff["settlement_period"] <= 48]
    .groupby(["settlement_date", "price_derivation_code"])
    .size()
    .reset_index(name="count")
)
fig_pdc = px.bar(
    daily_pdc,
    x="settlement_date",
    y="count",
    color="price_derivation_code",
    color_discrete_map={"N": "#17becf", "P": "#e377c2", "K": "#d62728"},
    labels={"settlement_date": "Date", "count": "Periods", "price_derivation_code": "Code"},
    barmode="stack",
)
fig_pdc.update_layout(
    height=300,
    margin=dict(t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_pdc)

# Box plot + stats table side by side at equal width
col_pdc_box, col_pdc_tbl = st.columns(2)

with col_pdc_box:
    st.markdown("**SSP Distribution by Code**")
    fig_box = px.box(
        dff,
        x="price_derivation_code",
        y="ssp",
        color="price_derivation_code",
        color_discrete_map={"P": "#e377c2", "N": "#17becf"},
        labels={"price_derivation_code": "Code", "ssp": "SSP £/MWh"},
        points=False,
    )
    fig_box.update_layout(
        height=340,
        margin=dict(t=10, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_box)

with col_pdc_tbl:
    st.markdown("**Summary statistics by code**")
    stats = (
        dff.groupby("price_derivation_code")["ssp"]
        .agg(Count="count", Mean="mean", Median="median", Std="std", Min="min", Max="max")
        .round(2)
        .reset_index()
        .rename(columns={"price_derivation_code": "Code"})
    )
    st.dataframe(stats, width="stretch", hide_index=True)
    st.caption(
        "P-code prices cluster tighter around the replacement reference value. "
        "N-code periods show wider spread — driven by the actual market stack."
    )

# ── Actual vs Predicted ───────────────────────────────────────────────────────
st.divider()
_pred_path = PRED_PATH_P3
_is_p3     = True

if _pred_path.exists():
    _test_pred_tmp = pd.read_csv(_pred_path)
    _test_dates    = sorted(_test_pred_tmp["settlement_date"].unique())
    _test_label    = f"{_test_dates[0]} → {_test_dates[-1]}" if _test_dates else "—"
else:
    _test_label = "—"
st.subheader(f"Model Forecast vs Actual (Phase 3 Level-Shape · Test: {_test_label})")

if _pred_path.exists():
    pred = pd.read_csv(_pred_path, parse_dates=["settlement_datetime"])
    has_q_pred = "ssp_q50" in pred.columns

    # ── Metrics row ───────────────────────────────────────────────────────────
    mae_val   = pred["abs_error"].mean() if "abs_error" in pred.columns else (pred["ssp_predicted"] - pred["ssp_actual"]).abs().mean()
    pred["_err"] = pred["ssp_predicted"] - pred["ssp_actual"]
    pred["_abs"]  = pred["_err"].abs()
    rmse_val  = (pred["_err"] ** 2).mean() ** 0.5
    denom     = (pred["ssp_actual"].abs() + pred["ssp_predicted"].abs()) / 2
    smape_val = (pred["_abs"] / denom.replace(0, float("nan"))).mean() * 100

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("MAE (P50, all periods)", f"£{mae_val:.2f}/MWh")
    mc2.metric("RMSE", f"£{rmse_val:.2f}/MWh")
    mc3.metric("sMAPE", f"{smape_val:.1f}%")
    mc4.metric("Test periods", len(pred))

    # Phase 3 decomposition metrics
    if _is_p3 and "actual_daily_level" in pred.columns and "pred_level_q50" in pred.columns:
        level_errs = (
            pred.groupby("settlement_date")
            .apply(lambda g: abs(g["pred_level_q50"].iloc[0] - g["actual_daily_level"].iloc[0]))
        )
        mc5.metric("Level MAE", f"£{level_errs.mean():.2f}/MWh/day",
                   help="Error in predicting the day's average price level (Stage 1)")
    else:
        elevated = pred[pred["ssp_actual"] > 200]
        elev_mae = elevated["_abs"].mean() if len(elevated) > 0 else float("nan")
        mc5.metric("MAE (SSP > £200)", f"£{elev_mae:.1f}" if not pd.isna(elev_mae) else "n/a")

    # Phase 3 shape decomposition row
    if _is_p3 and "actual_daily_level" in pred.columns:
        import numpy as np
        from scipy.stats import pearsonr
        shape_corrs, peak_gaps = [], []
        for _, day in pred.groupby("settlement_date"):
            act  = day["ssp_actual"].values;  am = act.mean()
            prd  = day["ssp_predicted"].values; pm = prd.mean()
            if (act - am).std() > 0 and (prd - pm).std() > 0:
                r, _ = pearsonr(act - am, prd - pm)
                shape_corrs.append(r)
            peak_gaps.append(abs(int(np.argmax(act)) - int(np.argmax(prd))))
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Shape correlation", f"{float(pd.Series(shape_corrs).mean()):.3f}",
                   help="Mean Pearson r between predicted and actual intra-day profiles per day")
        dc2.metric("Peak timing error", f"{float(pd.Series(peak_gaps).mean()):.1f} SPs",
                   help="Mean absolute SP offset between predicted and actual daily peak")
        _p3_mae = f"£{_p3.get('MAE', 0):.2f}" if _p3 else "—"
        dc3.metric("Phase 3 MAE (this test)", f"{_p3_mae}/MWh",
                   help="Phase 3 two-stage non-recursive · CPI-adjusted · 3-year rolling train window")

    # ── Time series: actual vs predicted with quantile band ───────────────────
    fig_pred = go.Figure()

    if has_q_pred:
        fig_pred.add_trace(go.Scatter(
            x=pd.concat([pred["settlement_datetime"], pred["settlement_datetime"].iloc[::-1]]),
            y=pd.concat([pred["ssp_q90"], pred["ssp_q10"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.12)",
            line=dict(color="rgba(255,127,14,0)"),
            hoverinfo="skip", name="P10–P90",
        ))

    fig_pred.add_trace(go.Scatter(
        x=pred["settlement_datetime"], y=pred["ssp_actual"],
        name="Actual SSP", line=dict(color="#1f77b4", width=1.5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=pred["settlement_datetime"], y=pred["ssp_predicted"],
        name="P50 Forecast", line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))
    fig_pred.update_layout(
        xaxis_title="Datetime", yaxis_title="£/MWh",
        height=380, margin=dict(t=10, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_pred)

    # ── Scatter + error distribution side by side ─────────────────────────────
    col_sc, col_err = st.columns(2)

    with col_sc:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=pred["ssp_actual"], y=pred["ssp_predicted"],
            mode="markers", marker=dict(color="#1f77b4", opacity=0.5, size=5),
            name="Predicted vs Actual",
        ))
        ax_min = min(pred["ssp_actual"].min(), pred["ssp_predicted"].min()) - 5
        ax_max = max(pred["ssp_actual"].max(), pred["ssp_predicted"].max()) + 5
        fig_sc.add_trace(go.Scatter(
            x=[ax_min, ax_max], y=[ax_min, ax_max],
            mode="lines", line=dict(color="red", dash="dash"), name="Perfect forecast",
        ))
        fig_sc.update_layout(
            xaxis_title="Actual SSP (£/MWh)", yaxis_title="Predicted SSP (£/MWh)",
            height=360, margin=dict(t=30, b=40),
            title_text="Predicted vs Actual", title_x=0,
        )
        st.plotly_chart(fig_sc)

    with col_err:
        fig_err = px.histogram(
            pred, x="_err", nbins=40,
            color_discrete_sequence=["#ff7f0e"],
            labels={"_err": "Forecast Error (£/MWh)", "count": "Periods"},
        )
        fig_err.add_vline(x=0, line_dash="dash", line_color="black")
        fig_err.update_layout(
            height=360, margin=dict(t=30, b=40),
            title_text="Error Distribution (Predicted − Actual)", title_x=0,
        )
        st.plotly_chart(fig_err)

    # ── Daily error summary ───────────────────────────────────────────────────
    daily_err = (
        pred.groupby("settlement_date")["_abs"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "Mean AE", "max": "Max AE"})
    )
    fig_daily_err = go.Figure()
    # Draw Max AE first (taller bar, behind) then Mean AE on top (shorter bar, in front).
    # This ensures Mean AE (blue) is cleanly visible at the bottom and Max AE (orange)
    # shows only for the portion above Mean AE — no colour blending.
    fig_daily_err.add_trace(go.Bar(
        x=daily_err["settlement_date"], y=daily_err["Max AE"],
        name="Max AE", marker_color="#ff7f0e",
    ))
    fig_daily_err.add_trace(go.Bar(
        x=daily_err["settlement_date"], y=daily_err["Mean AE"],
        name="Mean AE", marker_color="#1f77b4",
    ))
    fig_daily_err.update_layout(
        xaxis_title="Date", yaxis_title="£/MWh",
        barmode="overlay", height=300, margin=dict(t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_text="Daily Forecast Error", title_x=0,
    )
    st.plotly_chart(fig_daily_err)

else:
    st.info("No predictions found. Run `python src/models/train_phase3.py` to generate them.")

# ── Feature importance ────────────────────────────────────────────────────────
st.divider()
st.subheader("Feature Importance (Phase 3 · Top 20)")
st.caption(
    "Level model (Stage 1): absolute Spearman rank correlation with the CPI-deflated daily mean target "
    "over the full training set — stable with O(1000) daily rows. "
    "Shape model (Stage 2): permutation importance (n_repeats=5) on the validation set."
)


@st.cache_data(ttl=7200)
def load_phase3_importances():
    import json
    if not (LEVEL_IMP_CSV.exists() and SHAPE_IMP_CSV.exists()):
        return None, None, None, None
    with open(LEVEL_FEAT_JSON) as f:
        n_level = len(json.load(f))
    with open(SHAPE_FEAT_JSON) as f:
        n_shape = len(json.load(f))
    level_imp = (
        pd.read_csv(LEVEL_IMP_CSV)
        .nlargest(20, "importance_mean")
        .sort_values("importance_mean")
        .rename(columns={"importance_mean": "importance"})
    )
    shape_imp = (
        pd.read_csv(SHAPE_IMP_CSV)
        .nlargest(20, "importance_mean")
        .sort_values("importance_mean")
        .rename(columns={"importance_mean": "importance"})
    )
    return level_imp, shape_imp, n_level, n_shape


_level_imp, _shape_imp, _n_level, _n_shape = load_phase3_importances()

if _level_imp is not None:
    fi_tab1, fi_tab2 = st.tabs([
        f"Level model (Stage 1 · {_n_level} features)",
        f"Shape model (Stage 2 · {_n_shape} features)",
    ])

    def _fi_chart(imp_df, color):
        fig = go.Figure()
        _err = imp_df["importance_std"].tolist() if "importance_std" in imp_df.columns else None
        fig.add_trace(go.Bar(
            x=imp_df["importance"],
            y=imp_df["feature"],
            orientation="h",
            marker_color=color,
            error_x=dict(type="data", array=_err, visible=_err is not None),
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis_title="Mean Decrease in Impurity (relative)",
            yaxis_title=None,
            height=520,
            margin=dict(t=10, b=40, l=220),
            showlegend=False,
        )
        return fig

    with fi_tab1:
        st.caption(
            "Daily-level features: SSP/NIV lags, rolling stats, weather, wind/gas generation lags, "
            "CPI index/YoY, calendar harmonics."
        )
        st.plotly_chart(_fi_chart(_level_imp, "#1a6ea0"))

    with fi_tab2:
        st.caption(
            "SP-level fixed-point features: lag-48/96/336 for SSP, NIV, weather, wind/gas; "
            "daily-level lags merged from Stage 1; calendar (SP position, day-of-week, harmonics)."
        )
        st.plotly_chart(_fi_chart(_shape_imp, "#e07b39"))
else:
    st.info("No Phase 3 models found. Run `python src/models/train_phase3.py` to generate them.")

# ── Raw data table ─────────────────────────────────────────────────────────────
with st.expander("Raw data"):
    st.dataframe(
        dff[["settlement_date", "settlement_period", "time_label", "ssp",
             "net_imbalance_volume", "price_derivation_code", "replacement_price"]]
        .sort_values(["settlement_date", "settlement_period"], ascending=False)
        .reset_index(drop=True),
        width="stretch",
        height=300,
    )
    st.download_button(
        "Download CSV",
        data=dff.to_csv(index=False),
        file_name="system_prices_filtered.csv",
        mime="text/csv",
    )

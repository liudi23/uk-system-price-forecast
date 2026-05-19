"""
UK Electricity System Price Dashboard
Data source: Elexon BMRS — data/raw/system_prices.csv
Run: streamlit run src/dashboard/streamlit_app.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH     = ROOT / "data" / "raw" / "system_prices.csv"
PRED_PATH     = ROOT / "model_assets" / "test_predictions.csv"
FORECAST_PATH = ROOT / "model_assets" / "next_day_forecast.csv"
FI_PATH       = ROOT / "model_assets" / "hgbr_feature_importance.csv"

FETCH_ELEXON    = ROOT / "src" / "data"   / "fetch_elexon.py"
FETCH_WEATHER   = ROOT / "src" / "data"   / "fetch_weather.py"
FORECAST_SCRIPT = ROOT / "src" / "models" / "forecast.py"
FORECASTS_DIR   = ROOT / "model_assets"   / "forecasts"

st.set_page_config(
    page_title="UK System Price Dashboard",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Filters")

st.sidebar.divider()
st.sidebar.subheader("Data & Forecast")
if st.sidebar.button("Refresh Data & Run Forecast", use_container_width=True):
    with st.spinner("Fetching latest Elexon data…"):
        subprocess.run(
            [sys.executable, str(FETCH_ELEXON), "--append"],
            capture_output=True,
        )
    with st.spinner("Fetching latest weather…"):
        subprocess.run(
            [sys.executable, str(FETCH_WEATHER)],
            capture_output=True,
        )
    with st.spinner("Running day-ahead forecast…"):
        subprocess.run(
            [sys.executable, str(FORECAST_SCRIPT)],
            capture_output=True,
        )
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Updates system prices, weather, and tomorrow's forecast.")
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
st.caption(
    f"Data: Elexon BMRS · {start_date} → {end_date} · "
    f"{len(dff):,} settlement periods"
)

# ── Production model banner ───────────────────────────────────────────────────
st.info(
    "**Production model:** HistGradientBoosting (HGBR) · "
    "Trained on 5 years of Elexon BMRS data (May 2021 – May 2026) · "
    "Features: price lags, rolling statistics, calendar/annual harmonics, "
    "UK weather (temperature, wind speed, solar irradiance, precipitation) · "
    "Test MAE: £15.01/MWh · sMAPE: 17.9%"
)

# ── Next-day forecast panel ───────────────────────────────────────────────────
st.subheader("Day-Ahead Forecast (HGBR · 48 Settlement Periods)")

if FORECAST_PATH.exists():
    fc = pd.read_csv(FORECAST_PATH, parse_dates=["settlement_datetime"])
    fc_date = fc["settlement_date"].iloc[0]
    fc_label = f"{fc_date}  ·  SP 1–48 (midnight → 23:30)"

    fm1, fm2, fm3, fm4 = st.columns(4)
    fm1.metric("Forecast date", fc_date)
    fm2.metric("Min predicted SSP", f"£{fc['ssp_predicted'].min():.2f}")
    fm3.metric("Avg predicted SSP", f"£{fc['ssp_predicted'].mean():.2f}")
    fm4.metric("Max predicted SSP", f"£{fc['ssp_predicted'].max():.2f}")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc["settlement_datetime"], y=fc["ssp_predicted"],
        name="Predicted SSP", fill="tozeroy",
        line=dict(color="#ff7f0e", width=2),
        hovertemplate="SP %{customdata}<br>£%{y:.2f}/MWh<extra></extra>",
        customdata=fc["settlement_period"],
    ))
    fig_fc.update_layout(
        xaxis_title="Datetime",
        yaxis_title="£/MWh",
        height=320,
        margin=dict(t=10, b=40),
        hovermode="x unified",
        showlegend=False,
        annotations=[dict(
            text=f"Forecast generated: {fc_label}",
            xref="paper", yref="paper",
            x=0, y=1.02, showarrow=False,
            font=dict(size=11, color="grey"),
        )],
    )
    st.plotly_chart(fig_fc, use_container_width=True)
else:
    st.info(
        "No forecast found. Click **Refresh Data & Run Forecast** in the sidebar "
        "or run `python src/models/forecast.py`."
    )

st.divider()

# ── Live: Today's Forecast vs Actual SSP ─────────────────────────────────────
st.subheader("Live: Today's Forecast vs Actual SSP")
st.caption(
    "Settled periods update automatically every 30 min. "
    "Click **↻ Fetch now** to pull the latest Elexon prices immediately."
)


@st.fragment(run_every="30m")
def live_comparison_panel():
    from datetime import datetime as dt_cls

    today_str = str(pd.Timestamp.now().date())
    today_fc_path = FORECASTS_DIR / f"forecast_{today_str}.csv"

    col_ts, col_btn = st.columns([5, 1])
    col_ts.caption(f"Last check: {dt_cls.now().strftime('%H:%M:%S')}")
    with col_btn:
        if st.button("↻ Fetch now", use_container_width=True):
            with st.spinner("Fetching…"):
                subprocess.run(
                    [sys.executable, str(FETCH_ELEXON), "--append"], capture_output=True
                )
            st.cache_data.clear()

    if not today_fc_path.exists():
        st.info(
            f"No forecast for today ({today_str}). "
            "Click **Refresh Data & Run Forecast** in the sidebar to generate one."
        )
        return

    fc_today = pd.read_csv(today_fc_path, parse_dates=["settlement_datetime"])

    # Read actuals directly (bypass cache so we always get the freshest file)
    raw_today = pd.read_csv(DATA_PATH, parse_dates=["settlement_date"])
    actuals_today = raw_today[
        raw_today["settlement_date"].dt.strftime("%Y-%m-%d") == today_str
    ][["settlement_period", "ssp"]].rename(columns={"ssp": "ssp_actual"})
    n_settled = len(actuals_today)

    # Current settlement period based on local clock
    now_local = pd.Timestamp.now()
    current_sp = min(int(now_local.hour * 2 + now_local.minute // 30 + 1), 48)
    now_dt = pd.Timestamp(today_str) + pd.Timedelta(minutes=30 * (current_sp - 1))

    # Metrics row (only shown when actuals are available)
    if n_settled > 0:
        merged_live = fc_today.merge(actuals_today, on="settlement_period", how="inner")
        merged_live["abs_error"] = (
            merged_live["ssp_predicted"] - merged_live["ssp_actual"]
        ).abs()
        running_mae = merged_live["abs_error"].mean()
        max_err     = merged_live["abs_error"].max()
        worst_sp    = int(merged_live.loc[merged_live["abs_error"].idxmax(), "settlement_period"])

        lm1, lm2, lm3, lm4 = st.columns(4)
        lm1.metric("Settled periods", f"{n_settled} / 48")
        lm2.metric("Running MAE", f"£{running_mae:.2f}/MWh")
        lm3.metric("Max error so far", f"£{max_err:.2f}/MWh")
        lm4.metric("Worst SP", f"SP {worst_sp}")

    # Chart
    fig_live = go.Figure()

    # Full forecast curve (dotted orange)
    fig_live.add_trace(go.Scatter(
        x=fc_today["settlement_datetime"], y=fc_today["ssp_predicted"],
        name="Forecast", line=dict(color="#ff7f0e", width=2, dash="dot"),
        hovertemplate="SP %{customdata}<br>Forecast £%{y:.2f}/MWh<extra></extra>",
        customdata=fc_today["settlement_period"],
    ))

    # Actual SSP for settled periods (solid blue)
    if n_settled > 0:
        fig_live.add_trace(go.Scatter(
            x=merged_live["settlement_datetime"], y=merged_live["ssp_actual"],
            name="Actual SSP", line=dict(color="#1f77b4", width=2.5),
            hovertemplate="SP %{customdata}<br>Actual £%{y:.2f}/MWh<extra></extra>",
            customdata=merged_live["settlement_period"],
        ))
    else:
        st.caption(
            "No actuals yet for today — Elexon publishes initial settlement prices "
            "after each period. Click **↻ Fetch now** or check back later."
        )

    # Vertical "Now" line
    fig_live.add_vline(
        x=now_dt.isoformat(), line_dash="solid", line_color="grey", line_width=1.5,
        annotation_text=f"Now · SP {current_sp}", annotation_position="top right",
    )

    fig_live.update_layout(
        xaxis_title="Settlement Period", yaxis_title="£/MWh",
        height=360, margin=dict(t=30, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_live, use_container_width=True)


live_comparison_panel()

st.divider()

# ── Live Forecast Verification ────────────────────────────────────────────────
st.subheader("Live Forecast Verification")
st.caption(
    "Compares each archived day-ahead forecast against the Elexon actuals once they are available. "
    "Click **Refresh Data & Run Forecast** in the sidebar to pull the latest prices."
)

archived = sorted(FORECASTS_DIR.glob("forecast_*.csv")) if FORECASTS_DIR.exists() else []

if not archived:
    st.info("No archived forecasts yet. Run the forecast once to start tracking.")
else:
    # Find which archived dates also have actuals in the loaded data
    actual_dates = set(df["settlement_date"].dt.strftime("%Y-%m-%d").unique())
    verified_dates = [
        f.stem.replace("forecast_", "")
        for f in archived
        if f.stem.replace("forecast_", "") in actual_dates
    ]
    pending_dates = [
        f.stem.replace("forecast_", "")
        for f in archived
        if f.stem.replace("forecast_", "") not in actual_dates
    ]

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
            FORECASTS_DIR / f"forecast_{sel_date}.csv",
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
        fig_v.add_trace(go.Scatter(
            x=merged["settlement_datetime"], y=merged["ssp_actual"],
            name="Actual SSP", line=dict(color="#1f77b4", width=2),
            hovertemplate="SP %{customdata}<br>Actual £%{y:.2f}<extra></extra>",
            customdata=merged["settlement_period"],
        ))
        fig_v.add_trace(go.Scatter(
            x=merged["settlement_datetime"], y=merged["ssp_predicted"],
            name="Forecast SSP", line=dict(color="#ff7f0e", width=2, dash="dot"),
            hovertemplate="SP %{customdata}<br>Forecast £%{y:.2f}<extra></extra>",
            customdata=merged["settlement_period"],
        ))
        fig_v.update_layout(
            xaxis_title="Datetime", yaxis_title="£/MWh",
            height=340, margin=dict(t=10, b=40), hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_v, use_container_width=True)

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
            st.plotly_chart(fig_ep, use_container_width=True)

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
            st.plotly_chart(fig_eh, use_container_width=True)

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
st.plotly_chart(fig_ts, use_container_width=True)

# ── Heatmap + Imbalance side by side ─────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Daily SSP Heatmap by Settlement Period")
    pivot = dff.pivot_table(
        index="settlement_period", columns="settlement_date", values="ssp", aggfunc="mean"
    )
    pivot.columns = pivot.columns.strftime("%m-%d")

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels=dict(x="Date", y="Settlement Period", color="SSP £/MWh"),
    )
    fig_heat.update_layout(height=420, margin=dict(t=10, b=40))
    fig_heat.update_xaxes(tickangle=45, nticks=15)
    st.plotly_chart(fig_heat, use_container_width=True)

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
    st.plotly_chart(fig_niv, use_container_width=True)

# ── Average daily profile ────────────────────────────────────────────────────
st.subheader("Average Settlement Period Profile (SSP)")

profile = (
    dff.groupby("settlement_period")[["ssp", "time_label"]]
    .agg({"ssp": "mean", "time_label": "first"})
    .reset_index()
)

fig_profile = go.Figure()
fig_profile.add_trace(
    go.Scatter(
        x=profile["time_label"], y=profile["ssp"],
        name="Avg SSP", fill="tozeroy",
        line=dict(color="#1f77b4"),
    )
)
fig_profile.update_layout(
    xaxis_title="Time of Day (HH:MM)",
    yaxis_title="£/MWh",
    height=320,
    margin=dict(t=10, b=40),
    hovermode="x unified",
    showlegend=False,
)
st.plotly_chart(fig_profile, use_container_width=True)

# ── Price Derivation Code ────────────────────────────────────────────────────
st.subheader("Price Derivation Code (P vs N)")
st.caption(
    "**N (Normal)** — SSP/SBP set from the accepted bid-offer stack: the actual marginal cost of balancing.  "
    "**P (Replacement Price)** — bid-offer stack too thin to price fairly; ESO substitutes a reference price, "
    "making SSP = SBP by definition. ~50% of periods trigger P-code."
)

# Full-width: daily P/N period counts over time
daily_pdc = (
    dff.groupby(["settlement_date", "price_derivation_code"])
    .size()
    .reset_index(name="count")
)
fig_pdc = px.bar(
    daily_pdc,
    x="settlement_date",
    y="count",
    color="price_derivation_code",
    color_discrete_map={"P": "#e377c2", "N": "#17becf"},
    labels={"settlement_date": "Date", "count": "Periods", "price_derivation_code": "Code"},
    barmode="stack",
)
fig_pdc.update_layout(
    height=300,
    margin=dict(t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_pdc, use_container_width=True)

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
    st.plotly_chart(fig_box, use_container_width=True)

with col_pdc_tbl:
    st.markdown("**Summary statistics by code**")
    stats = (
        dff.groupby("price_derivation_code")["ssp"]
        .agg(Count="count", Mean="mean", Median="median", Std="std", Min="min", Max="max")
        .round(2)
        .reset_index()
        .rename(columns={"price_derivation_code": "Code"})
    )
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.caption(
        "P-code prices cluster tighter around the replacement reference value. "
        "N-code periods show wider spread — driven by the actual market stack."
    )

# ── Actual vs Predicted ───────────────────────────────────────────────────────
st.divider()
st.subheader("Model Forecast vs Actual (HGBR — 5-year + Weather · Test: May 11–17 2026)")

if PRED_PATH.exists():
    pred = pd.read_csv(PRED_PATH, parse_dates=["settlement_datetime"])

    # ── Metrics row ───────────────────────────────────────────────────────────
    mae_val  = pred["abs_error"].mean()
    rmse_val = (pred["error"] ** 2).mean() ** 0.5
    denom    = (pred["ssp_actual"].abs() + pred["ssp_predicted"].abs()) / 2
    smape_val = (pred["abs_error"] / denom.replace(0, float("nan"))).mean() * 100

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("MAE", f"£{mae_val:.2f}/MWh")
    mc2.metric("RMSE", f"£{rmse_val:.2f}/MWh")
    mc3.metric("sMAPE", f"{smape_val:.1f}%")
    mc4.metric("Test periods", len(pred))

    # ── Time series: actual vs predicted ─────────────────────────────────────
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=pred["settlement_datetime"], y=pred["ssp_actual"],
        name="Actual SSP", line=dict(color="#1f77b4", width=1.5),
    ))
    fig_pred.add_trace(go.Scatter(
        x=pred["settlement_datetime"], y=pred["ssp_predicted"],
        name="Predicted SSP", line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))
    fig_pred.update_layout(
        xaxis_title="Datetime", yaxis_title="£/MWh",
        height=380, margin=dict(t=10, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_pred, use_container_width=True)

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
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_err:
        fig_err = px.histogram(
            pred, x="error", nbins=40,
            color_discrete_sequence=["#ff7f0e"],
            labels={"error": "Forecast Error (£/MWh)", "count": "Periods"},
        )
        fig_err.add_vline(x=0, line_dash="dash", line_color="black")
        fig_err.update_layout(
            height=360, margin=dict(t=30, b=40),
            title_text="Error Distribution (Predicted − Actual)", title_x=0,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # ── Daily error summary ───────────────────────────────────────────────────
    daily_err = (
        pred.groupby("settlement_date")["abs_error"]
        .agg(["mean", "max"])
        .reset_index()
        .rename(columns={"mean": "Mean AE", "max": "Max AE"})
    )
    fig_daily_err = go.Figure()
    fig_daily_err.add_trace(go.Bar(
        x=daily_err["settlement_date"], y=daily_err["Mean AE"],
        name="Mean AE", marker_color="#1f77b4",
    ))
    fig_daily_err.add_trace(go.Bar(
        x=daily_err["settlement_date"], y=daily_err["Max AE"],
        name="Max AE", marker_color="#ff7f0e", opacity=0.6,
    ))
    fig_daily_err.update_layout(
        xaxis_title="Date", yaxis_title="£/MWh",
        barmode="overlay", height=300, margin=dict(t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title_text="Daily Forecast Error", title_x=0,
    )
    st.plotly_chart(fig_daily_err, use_container_width=True)

else:
    st.info("No predictions found. Run `python src/models/train_lgbm.py` to generate them.")

# ── Feature importance ────────────────────────────────────────────────────────
st.divider()
st.subheader("Feature Importance (Top 20 · Permutation MAE Reduction)")
st.caption("Computed on the validation set via permutation importance — shows how much each feature reduces MAE when present.")

if FI_PATH.exists():
    fi = pd.read_csv(FI_PATH).drop_duplicates(subset="feature").nlargest(20, "importance_mean")
    fi = fi.sort_values("importance_mean")  # ascending for horizontal bar

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=fi["importance_mean"],
        y=fi["feature"],
        orientation="h",
        error_x=dict(type="data", array=fi["importance_std"].tolist(), visible=True),
        marker_color="#1f77b4",
        hovertemplate="%{y}<br>MAE reduction: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(
        xaxis_title="Mean MAE Reduction (£/MWh)",
        yaxis_title=None,
        height=520,
        margin=dict(t=10, b=40, l=200),
        showlegend=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)
else:
    st.info("No feature importance file found. Run `python src/models/train_lgbm.py`.")

# ── Raw data table ─────────────────────────────────────────────────────────────
with st.expander("Raw data"):
    st.dataframe(
        dff[["settlement_date", "settlement_period", "time_label", "ssp",
             "net_imbalance_volume", "price_derivation_code", "replacement_price"]]
        .sort_values(["settlement_date", "settlement_period"], ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=300,
    )
    st.download_button(
        "Download CSV",
        data=dff.to_csv(index=False),
        file_name="system_prices_filtered.csv",
        mime="text/csv",
    )

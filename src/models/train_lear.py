"""
Phase 5c: LEAR baseline — Lasso-Estimated AutoRegressive model for UK SSP.

The standard interpretable benchmark in electricity-price forecasting
(Weron et al.), adapted to GB half-hourly settlement:

  One Lasso per settlement period s. Target: asinh-scaled raw SSP of (D, s).
  All regressors are known by 12:00 UTC on D−1 (same information set and
  vintage cutoff as the Phase-3 HGBR after Phase 5b):

    - full price profiles of D−1, D−2, D−7  (3 × 48 columns)
    - daily aggregates of D−1 / D−7 (mean, min, max)
    - MID market-index daily mean of D−1 and the SSP−MID spread of D−1
    - day-ahead vintages for (D, s): WINDFOR, TSDF, margin, wind penetration,
      OCNMFD surplus (from data/vintages/, Phase 5b)
    - day-of-week one-hot + England bank-holiday flag

  Lasso α per SP via 5-fold LassoCV on the (robust-scaled, asinh) target.
  P10/P90 from per-SP training-residual quantiles (pre-conformal, like the
  HGBR raw bands; production calibration would re-fit δ as usual).

Usage:
    python src/models/train_lear.py --backtest                  # 5-fold WF, window 1092d
    python src/models/train_lear.py --backtest --window-days 365
    python src/models/train_lear.py --fit                       # production fit + coefficient export

Outputs (backtest): model_assets/wf_lear_predictions.csv + console report
         (fit)     : model_assets/lear_models.pkl, model_assets/lear_coefficients.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "features"))
from train_phase3 import WALK_FOLDS                      # same folds as HGBR
from calendar_features import _uk_holidays               # has offline fallback

DATASET      = ROOT / "data" / "processed" / "dataset_5yr.csv"
VINTAGES     = ROOT / "data" / "vintages" / "da_vintages.csv"
MID          = ROOT / "data" / "vintages" / "mid_prices.csv"
ASSETS       = ROOT / "model_assets"
WF_HGBR      = ASSETS / "walk_forward_predictions.csv"

N_SP = 48
PRICE_LAG_DAYS = (1, 2, 7)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Panel construction ────────────────────────────────────────────────────────

def load_panel():
    """Return (price_pivot days×48, exog dict of days×48 pivots, daily frame)."""
    ds = pd.read_csv(DATASET, usecols=["settlement_date", "settlement_period",
                                       "ssp", "ssp_raw"],
                     parse_dates=["settlement_date"])
    ds["price"] = ds["ssp_raw"].fillna(ds["ssp"])
    piv = ds.pivot_table(index="settlement_date", columns="settlement_period",
                         values="price", aggfunc="first")
    piv = piv.reindex(columns=range(1, N_SP + 1))
    piv = piv[piv.notna().sum(axis=1) == N_SP]           # complete days only

    exog = {}
    if VINTAGES.exists():
        v = pd.read_csv(VINTAGES, parse_dates=["settlement_date"])
        v["margin_fc_mw"] = v["tsdf_fc_mw"] - v["windfor_fc_mw"]
        v["wind_pen_fc"]  = 100.0 * v["windfor_fc_mw"] / v["tsdf_fc_mw"]
        for c in ["windfor_fc_mw", "tsdf_fc_mw", "margin_fc_mw",
                  "wind_pen_fc", "surplus_fc_mw"]:
            exog[c] = (v.pivot_table(index="settlement_date",
                                     columns="settlement_period",
                                     values=c, aggfunc="first")
                        .reindex(index=piv.index, columns=range(1, N_SP + 1)))
    else:
        log.warning("da_vintages.csv missing — LEAR runs without vintage exogenous")

    daily = pd.DataFrame(index=piv.index)
    daily["mean"] = piv.mean(axis=1)
    daily["min"]  = piv.min(axis=1)
    daily["max"]  = piv.max(axis=1)
    if MID.exists():
        m = pd.read_csv(MID, parse_dates=["settlement_date"])
        daily["mid_mean"] = (m.groupby("settlement_date")["mid_price"].mean()
                              .reindex(piv.index))
    else:
        daily["mid_mean"] = np.nan
    return piv, exog, daily


def build_design(piv, exog, daily):
    """Return (dates, X_shared, exog_by_sp, feature names).

    X_shared: day-level block identical for every SP (price profiles, daily
    aggregates, MID lags, calendar). exog_by_sp[s]: per-SP vintage columns.
    """
    dates = piv.index
    pos   = {d: i for i, d in enumerate(dates)}
    n     = len(dates)
    P     = piv.values

    blocks, names = [], []
    for lag in PRICE_LAG_DAYS:
        B = np.full((n, N_SP), np.nan)
        for i, d in enumerate(dates):
            j = pos.get(d - pd.Timedelta(days=lag))
            if j is not None:
                B[i] = P[j]
        blocks.append(B)
        names += [f"p_lag{lag}d_sp{s}" for s in range(1, N_SP + 1)]

    def _daily_lag(col, lag):
        return daily[col].reindex(dates - pd.Timedelta(days=lag)).values

    agg = np.column_stack([
        _daily_lag("mean", 1), _daily_lag("mean", 7),
        _daily_lag("min", 1),  _daily_lag("max", 1),
        _daily_lag("mid_mean", 1),
        _daily_lag("mean", 1) - _daily_lag("mid_mean", 1),
    ])
    names += ["dmean_lag1", "dmean_lag7", "dmin_lag1", "dmax_lag1",
              "mid_lag1", "ssp_mid_spread_lag1"]

    dow = pd.get_dummies(pd.Series(dates.dayofweek, index=dates),
                         prefix="dow").reindex(columns=[f"dow_{i}" for i in range(7)],
                                               fill_value=0).values.astype(float)
    names += [f"dow_{i}" for i in range(7)]
    hol = _uk_holidays(sorted({d.year for d in dates}))
    is_hol = np.array([1.0 if d.date() in hol else 0.0 for d in dates]).reshape(-1, 1)
    names += ["is_holiday"]

    X_shared = np.column_stack(blocks + [agg, dow, is_hol])

    exog_by_sp, exog_names = {}, list(exog.keys())
    for s in range(1, N_SP + 1):
        if exog:
            exog_by_sp[s] = np.column_stack([exog[c][s].values for c in exog_names])
        else:
            exog_by_sp[s] = np.zeros((n, 0))
    return dates, X_shared, exog_by_sp, names + exog_names


# ── Fitting ───────────────────────────────────────────────────────────────────

def _make_pipe():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("lasso",  LassoCV(cv=5, alphas=40, max_iter=20000, n_jobs=-1,
                           random_state=42)),
    ])


def fit_sp(X, y):
    """Fit one SP model on asinh-robust-scaled target. Returns (pipe, a, b, q10, q90)."""
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    a = float(np.median(y))
    b = float(1.4826 * np.median(np.abs(y - a))) or 1.0
    z = np.arcsinh((y - a) / b)
    pipe = _make_pipe()
    pipe.fit(X, z)
    resid = y - (a + b * np.sinh(pipe.predict(X)))
    return pipe, a, b, float(np.quantile(resid, 0.10)), float(np.quantile(resid, 0.90))


def predict_sp(model, X):
    pipe, a, b, q10, q90 = model
    p = a + b * np.sinh(pipe.predict(X))
    return p, p + q10, p + q90


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(window_days: int, out_path: Path) -> pd.DataFrame:
    piv, exog, daily = load_panel()
    dates, X_shared, exog_by_sp, feat_names = build_design(piv, exog, daily)
    P = piv.values
    rows = []
    for fold_start_str, fold_end_str, label in WALK_FOLDS:
        fs, fe = pd.Timestamp(fold_start_str), pd.Timestamp(fold_end_str)
        if fe > dates.max():
            log.info("skip fold %s (beyond data)", label)
            continue
        tr_mask = (dates < fs - pd.Timedelta(days=1))
        if window_days:
            tr_mask &= (dates >= fs - pd.Timedelta(days=window_days))
        te_mask = (dates >= fs) & (dates <= fe)
        # price-profile warm-up: need D−7 present
        tr_mask &= np.isfinite(X_shared[:, :N_SP * len(PRICE_LAG_DAYS)]).all(axis=1)
        te_ok    = te_mask & np.isfinite(X_shared[:, :N_SP * len(PRICE_LAG_DAYS)]).all(axis=1)
        log.info("fold %s: train %d days, test %d days", label, tr_mask.sum(), te_ok.sum())
        for s in range(1, N_SP + 1):
            Xs = np.column_stack([X_shared, exog_by_sp[s]])
            model = fit_sp(Xs[tr_mask], P[tr_mask, s - 1])
            pred, lo, hi = predict_sp(model, Xs[te_ok])
            for d, y, p, l, h in zip(dates[te_ok], P[te_ok, s - 1], pred, lo, hi):
                rows.append((str(d.date()), s, y, p, l, h, label))
    df = pd.DataFrame(rows, columns=["settlement_date", "settlement_period",
                                     "ssp_actual", "lear_pred", "lear_q10",
                                     "lear_q90", "fold"])
    df.to_csv(out_path, index=False)
    log.info("predictions → %s", out_path)
    return df


def report(df: pd.DataFrame, window_days: int) -> None:
    def pinball(y, lo, hi):
        d10, d90 = y - lo, y - hi
        return float((np.maximum(0.10 * d10, -0.90 * d10).mean()
                      + np.maximum(0.90 * d90, -0.10 * d90).mean()) / 2)

    hgbr = None
    if WF_HGBR.exists():
        h = pd.read_csv(WF_HGBR)
        h["settlement_date"] = h["settlement_date"].astype(str).str[:10]
        hgbr = df.merge(h[["settlement_date", "settlement_period",
                           "ssp_predicted"]].rename(columns={"ssp_predicted": "hgbr"}),
                        on=["settlement_date", "settlement_period"], how="left")

    print(f"\n=== LEAR walk-forward (window {window_days or 'full'} days) ===")
    print(f"{'fold':12s} {'n':>5s} {'LEAR MAE':>9s} {'HGBR MAE':>9s} {'LEAR PB':>8s} "
          f"{'cov80':>6s}  {'LEAR/HGBR':>9s}")
    for label, g in list(df.groupby("fold")) + [("ALL", df)]:
        mae = (g.ssp_actual - g.lear_pred).abs().mean()
        pb  = pinball(g.ssp_actual.values, g.lear_q10.values, g.lear_q90.values)
        cov = ((g.ssp_actual >= g.lear_q10) & (g.ssp_actual <= g.lear_q90)).mean()
        hm = np.nan
        if hgbr is not None:
            gm = hgbr[(hgbr.fold == label)] if label != "ALL" else hgbr
            gm = gm.dropna(subset=["hgbr"])
            if len(gm):
                hm = (gm.ssp_actual - gm.hgbr).abs().mean()
        ratio = mae / hm if hm == hm else np.nan
        print(f"{label:12s} {len(g):5d} {mae:9.2f} {hm:9.2f} {pb:8.2f} {cov:6.2f}  {ratio:9.3f}")


# ── Production fit ────────────────────────────────────────────────────────────

def fit_production(window_days: int) -> None:
    import joblib
    piv, exog, daily = load_panel()
    dates, X_shared, exog_by_sp, feat_names = build_design(piv, exog, daily)
    P = piv.values
    cutoff = dates.max()
    mask = np.isfinite(X_shared[:, :N_SP * len(PRICE_LAG_DAYS)]).all(axis=1)
    if window_days:
        mask &= (dates >= cutoff - pd.Timedelta(days=window_days))
    models, coefs = {}, {}
    for s in range(1, N_SP + 1):
        Xs = np.column_stack([X_shared, exog_by_sp[s]])
        models[s] = fit_sp(Xs[mask], P[mask, s - 1])
        pipe = models[s][0]
        beta = pipe.named_steps["lasso"].coef_
        nz = [(feat_names[i], round(float(beta[i]), 4))
              for i in np.argsort(-np.abs(beta))[:15] if beta[i] != 0]
        coefs[s] = {"alpha": float(pipe.named_steps["lasso"].alpha_),
                    "n_nonzero": int((beta != 0).sum()), "top": nz}
    joblib.dump({"models": models, "features": feat_names,
                 "fit_through": str(cutoff.date()), "window_days": window_days},
                ASSETS / "lear_models.pkl")
    with open(ASSETS / "lear_coefficients.json", "w") as f:
        json.dump({"fit_through": str(cutoff.date()), "window_days": window_days,
                   "by_sp": coefs}, f, indent=1)
    log.info("saved lear_models.pkl + lear_coefficients.json (fit through %s)", cutoff.date())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--window-days", type=int, default=1092,
                    help="training window in days (0 = full history)")
    ap.add_argument("--out", default=str(ASSETS / "wf_lear_predictions.csv"))
    args = ap.parse_args()
    if args.backtest:
        df = backtest(args.window_days, Path(args.out))
        report(df, args.window_days)
    if args.fit:
        fit_production(args.window_days)
    if not (args.backtest or args.fit):
        ap.error("one of --backtest / --fit is required")


if __name__ == "__main__":
    main()

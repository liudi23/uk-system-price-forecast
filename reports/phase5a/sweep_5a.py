"""Phase 5a sweep: training window x sample weighting x target transform for the LEVEL model.

Walk-forward protocol: fold starts every 14 days over the recent period; for each fold,
train P10/P50/P90 level models on data strictly before (fold_start - VAL_GAP), filtered
to the config's window and weighted by the config's half-life; evaluate daily-level
q50 MAE, pinball(0.1/0.9), and 80% band coverage over the 14-day fold.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = Path("/tmp/claude-0/-home-claude/8964bb41-24bf-56f8-b485-9c93a9e15997/scratchpad/repo")
sys.path.insert(0, str(REPO / "src" / "models"))
sys.path.insert(0, str(REPO / "src" / "features"))
from level_features import LEVEL_TARGET, build_level_dataset, get_level_feature_cols
from train_phase3 import load_sp_data, BASE_PARAMS

OUT = Path("/tmp/claude-0/-home-claude/8964bb41-24bf-56f8-b485-9c93a9e15997/scratchpad/sweep_results.csv")

sp_df = load_sp_data(REPO / "data" / "processed" / "features_5yr.csv")
daily = build_level_dataset(sp_df)
feat_cols = get_level_feature_cols(daily)
# keep only columns with any data locally (reduced feature set: no weather/CPI/genmix)
feat_cols = [c for c in feat_cols if daily[c].notna().any()]
daily = daily.dropna(subset=[LEVEL_TARGET]).sort_values("date").reset_index(drop=True)
print(f"daily rows {len(daily)}  {daily['date'].min().date()} -> {daily['date'].max().date()}  feats {len(feat_cols)}")

FOLD_STARTS = pd.date_range("2026-05-01", "2026-10-19", freq="14D")
FOLD_LEN = pd.Timedelta(days=14)
VAL_GAP = pd.Timedelta(days=2)
QUANTILES = [0.10, 0.50, 0.90]

def pinball(y, q_pred, q):
    d = y - q_pred
    return np.mean(np.maximum(q * d, (q - 1) * d))

def run_config(window_months, halflife_days, transform):
    recs = []
    for fs in FOLD_STARTS:
        fe = fs + FOLD_LEN
        tr = daily[daily["date"] < fs - VAL_GAP]
        if window_months is not None:
            tr = tr[tr["date"] >= fs - pd.DateOffset(months=window_months)]
        te = daily[(daily["date"] >= fs) & (daily["date"] < fe)]
        if len(tr) < 120 or len(te) == 0:
            continue
        X, Xte = tr[feat_cols].values, te[feat_cols].values
        y = tr[LEVEL_TARGET].values
        if transform == "asinh":
            S = 50.0
            yt = np.arcsinh(y / S)
        else:
            yt = y
        if halflife_days:
            age = (fs - tr["date"]).dt.days.values
            w = 0.5 ** (age / halflife_days)
        else:
            w = None
        preds = {}
        for q in QUANTILES:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                              validation_fraction=0.1, **BASE_PARAMS)
            m.fit(X, yt, sample_weight=w)
            p = m.predict(Xte)
            preds[q] = np.sinh(p) * S if transform == "asinh" else p
        ya = te[LEVEL_TARGET].values
        recs.append(dict(
            fold=str(fs.date()), n=len(te),
            mae=float(np.mean(np.abs(ya - preds[0.50]))),
            bias=float(np.mean(preds[0.50] - ya)),
            pb10=pinball(ya, preds[0.10], 0.10), pb90=pinball(ya, preds[0.90], 0.90),
            cov80=float(np.mean((ya >= preds[0.10]) & (ya <= preds[0.90]))),
        ))
    r = pd.DataFrame(recs)
    return dict(window=window_months or 99, halflife=halflife_days or 0, transform=transform,
                n_folds=len(r), n_days=int(r.n.sum()),
                mae=float((r.mae * r.n).sum() / r.n.sum()),
                bias=float((r.bias * r.n).sum() / r.n.sum()),
                pinball=float(((r.pb10 + r.pb90) * r.n).sum() / r.n.sum() / 2),
                cov=float((r.cov80 * r.n).sum() / r.n.sum()))

configs = []
for wm in [12, 18, 24, 36, None]:
    for hl in [None, 365, 182, 91]:
        configs.append((wm, hl, "linear"))
# transform variants on a subset (baseline window + short list; extended after seeing winners)
for wm in [36, 18, None]:
    for hl in [None, 182]:
        configs.append((wm, hl, "asinh"))

import os
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT)
    done = {(int(r.window), int(r.halflife), r.transform) for r in prev.itertuples()}
t0 = time.time()
for i, (wm, hl, tf) in enumerate(configs):
    key = (wm or 99, hl or 0, tf)
    if key in done:
        continue
    res = run_config(wm, hl, tf)
    pd.DataFrame([res]).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"[{i+1}/{len(configs)}  {time.time()-t0:5.0f}s] win={str(wm or 'all'):>4} hl={str(hl or 'unif'):>5} {tf:6s} "
          f"MAE £{res['mae']:6.2f}  bias £{res['bias']:+6.2f}  pinball £{res['pinball']:5.2f}  cov {res['cov']:.2f}", flush=True)
print("saved", OUT)

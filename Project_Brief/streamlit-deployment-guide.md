# Deploying to Streamlit Community Cloud

A step-by-step guide to publishing the UK SSP Forecasting dashboard as a live public website using Streamlit Community Cloud (free), while keeping the GitHub repository private.

**Architecture:** Run the daily pipeline locally → push updated forecast CSVs to GitHub → Streamlit Cloud auto-reloads. The dashboard is read-only on the hosted version; retraining happens on your machine.

---

## Step 1 — Prepare the Repository

### 1a. Pin package versions in `requirements.txt`

Streamlit Cloud needs exact versions to build reliably. Run locally:

```bash
source .venv/bin/activate
pip freeze | grep -E "pandas|numpy|scikit-learn|streamlit|plotly|requests|holidays|reportlab" > requirements_pinned.txt
```

Update `requirements.txt` with pinned versions, e.g.:

```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
streamlit==1.45.1
plotly==5.24.1
requests==2.32.3
holidays==0.60
reportlab==4.2.5
```

### 1b. Add a Streamlit config file

Create `.streamlit/config.toml`:

```toml
[server]
headless = true
port = 8501

[theme]
base = "light"
```

### 1c. Ensure dashboard data files are tracked in git

The dashboard only reads CSVs/JSONs — it does **not** need `.pkl` files or `features_5yr.csv` at runtime. Verify these are not gitignored:

```bash
git status model_assets/next_day_forecast_phase3.csv
git status model_assets/day2_forecast_phase3.csv
git status model_assets/phase3_metrics.json
git status model_assets/test_predictions_phase3.csv
git status data/processed/dataset_5yr.csv
```

If any are untracked, add them:

```bash
git add model_assets/next_day_forecast_phase3.csv \
        model_assets/day2_forecast_phase3.csv \
        model_assets/phase3_metrics.json \
        model_assets/test_predictions_phase3.csv \
        model_assets/phase3_level_importance.csv \
        model_assets/phase3_shape_importance.csv \
        model_assets/walk_forward_predictions.csv \
        model_assets/forecasts/ \
        data/processed/dataset_5yr.csv \
        .streamlit/config.toml
git commit -m "Add dashboard data files and Streamlit config"
git push origin phase-3
```

**File sizes for reference** — all well within GitHub's 100 MB per-file limit:

| File | Size |
|---|---|
| `data/processed/dataset_5yr.csv` | 7.9 MB |
| `model_assets/shape_q50.pkl` | 3.6 MB |
| `model_assets/level_q90.pkl` | 2.2 MB |
| `data/processed/features_5yr.csv` | 116 MB — **gitignored**, not needed by dashboard |

---

## Step 2 — Merge to Main

Streamlit Community Cloud deploys from a branch — `main` is the standard choice:

```bash
git checkout main
git merge phase-3
git push origin main
```

---

## Step 3 — Create a Streamlit Community Cloud Account

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **Sign up** → **Continue with GitHub**
3. Authorise Streamlit to access your GitHub account

---

## Step 4 — Deploy the App

1. On the Streamlit Cloud dashboard, click **New app**
2. Fill in:
   - **Repository:** `liudi23/uk-system-price-forecast`
   - **Branch:** `main`
   - **Main file path:** `src/dashboard/streamlit_app.py`
3. Click **Deploy**

Streamlit installs `requirements.txt`, starts the app, and provides a public URL such as:

```
https://liudi23-uk-system-price-forecast-streamlit-app.streamlit.app
```

---

## Step 5 — Disable the Refresh Button on the Hosted Version

The Refresh button (which retrains models) will not work on Community Cloud — there is no compute for the full pipeline. Add this guard in `src/dashboard/streamlit_app.py`:

```python
import os
IS_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit_sharing"

if not IS_CLOUD:
    if st.button("Refresh Data & Retrain"):
        # ... existing pipeline code ...
else:
    st.info("Live deployment — forecasts updated daily via local pipeline push.")
```

---

## Step 6 — Daily Update Workflow

Create `run_daily_push.sh` in the project root:

```bash
#!/bin/bash
set -e
cd /Users/diliu/Library/CloudStorage/Dropbox/DS_projects/uk-system-price-forecast
source .venv/bin/activate

echo "=== Fetching latest data ==="
python src/data/fetch_elexon.py --append
python src/data/fetch_weather.py
python src/data/fetch_generation.py --append
python src/data/fetch_cpi.py
python src/data/extend_dataset.py

echo "=== Rebuilding features ==="
python src/features/build_features.py \
    --input  data/processed/dataset_5yr.csv \
    --output data/processed/features_5yr.csv

echo "=== Retraining & forecasting ==="
python src/models/train_phase3.py
python src/models/forecast_phase3.py

echo "=== Pushing to GitHub ==="
git add data/processed/dataset_5yr.csv \
        model_assets/next_day_forecast_phase3.csv \
        model_assets/day2_forecast_phase3.csv \
        model_assets/phase3_metrics.json \
        model_assets/test_predictions_phase3.csv \
        model_assets/forecasts/
git commit -m "Daily forecast update $(date +%Y-%m-%d)"
git push origin main

echo "=== Done — Streamlit Cloud will auto-reload ==="
```

Make it executable and run each morning:

```bash
chmod +x run_daily_push.sh
./run_daily_push.sh
```

Streamlit Community Cloud detects the push and reloads within ~1 minute. Visitors always see today's forecast.

---

## Summary

| What | Where |
|---|---|
| Source code | Private GitHub repository |
| Live dashboard | Public Streamlit Community Cloud URL |
| Daily update | Run `run_daily_push.sh` locally → push CSVs → app reloads |
| Cost | Free |

The code remains private. Only the pre-computed forecast outputs (CSV/JSON) are visible via the public app.

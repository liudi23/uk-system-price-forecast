# UK Electricity System Price Forecasting Platform

Keywords: FastAPI, Streamlit, deployment, GitHub workflow, end-to-end (e2e), time series forecasting, electricity markets, Elexon BMRS, NESO, LightGBM, XGBoost, feature engineering, energy analytics

We provide an end-to-end example using public UK electricity market data from Elexon BMRS and NESO. The project focuses on forecasting UK electricity system prices at the settlement-period level (30-minute intervals), with applications in electricity trading, flexibility optimisation, and market analytics.

The goal of this repository is not only to train a forecasting model, but also to demonstrate a realistic data science workflow including:

- automated data ingestion
- feature engineering
- forecasting pipelines
- model serving
- interactive analytics dashboards
- deployment-oriented project structure

In this repo, we implement:

- a data ingestion and training pipeline
- an inference FastAPI backend
- an example Streamlit app for interactive analytics and forecasting visualisation

---

## A training/data pipeline in `./training_pipeline`

The pipeline automatically downloads and processes historical electricity market data from Elexon BMRS.

Initial modelling focuses on computationally efficient approaches such as:

- lag-based forecasting
- rolling statistics
- LightGBM / XGBoost baselines

Potential future extensions include:

- probabilistic forecasting
- spike classification
- weather-aware forecasting
- regime analysis

### Folder structure

```text
data/
    raw/                    # Raw downloaded Elexon/NESO datasets
    processed/              # Processed modelling datasets

training_pipeline/
    src/
        data/               # Data ingestion and preprocessing
        features/           # Feature engineering
        models/             # Model training and evaluation

    run_train_pipeline.py   # Training pipeline entrypoint
    train_config.yml        # Training configuration

model_assets/
    # Trained models, fitted objects, evaluation metrics

requirements.txt
```

### To run the training pipeline

```bash
pip install -r requirements.txt
python src/data/fetch_elexon.py
python src/data/build_dataset.py
python src/models/train_baseline.py
```

---

## A corresponding inference FastAPI backend in `./inference_backend`

The backend serves trained forecasting models through REST API endpoints.

Potential endpoints include:

- latest market price
- next settlement-period forecast
- day-ahead forecast
- probabilistic forecast intervals
- spike probability

### Folder structure

```text
inference_backend/
    model_assets/
        # Trained models and fitted preprocessing objects

    src/
        # Inference and preprocessing logic

    main.py
        # FastAPI app

    Dockerfile
    docker_compose.yml
```

### To test the backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then test the API using Swagger UI:

```text
http://localhost:8000/docs
```

---

## An example Streamlit analytics dashboard in `./streamlit`

The Streamlit app functions as an interactive analytics and forecasting interface rather than a static notebook output.

Planned dashboard functionality includes:

- actual vs predicted prices
- next-day forecast curves
- confidence intervals
- feature importance visualisation
- anomaly/spike highlighting
- configurable forecasting horizons
- historical market analysis
- scenario analysis

### Folder structure

```text
streamlit/
    app.py
        # Main Streamlit dashboard

    backend_client.py
        # Client connecting to FastAPI backend

    app_config.yml
        # Dashboard configuration

requirements.txt
```

### To test the Streamlit app

```bash
streamlit run app.py
```

Open in browser:

```text
http://localhost:8501
```

---

## Project roadmap

### Current status

Implemented:

- GitHub repository structure
- automated Elexon data ingestion
- raw/processed data pipeline
- feature engineering prototype
- baseline forecasting pipeline
- initial Streamlit dashboard

### Planned improvements

- multi-month historical ingestion
- probabilistic forecasting
- spike classification
- renewable generation forecasting
- weather API integration
- FastAPI deployment
- Docker support
- scenario analysis tools
- real-time dashboard updates

---

## Motivation

Electricity markets are highly dynamic systems characterised by:

- strong seasonality
- renewable intermittency
- extreme price spikes
- non-stationary behaviour

This project aims to combine:

- data engineering
- forecasting
- deployment
- interactive analytics

into a realistic end-to-end data science solution relevant to:

- energy trading
- flexibility optimisation
- quantitative research
- market analytics
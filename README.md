# Trendpilot AI 📈

[![CI](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](.github/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

**Trendpilot AI** is a containerized, end-to-end machine learning forecasting platform for entity-level panel data — any dataset with one row per (entity, date). It covers the full ML lifecycle: data ingestion and validation, leakage-safe feature engineering, hyperparameter tuning, model training and evaluation, a REST inference API, an interactive dashboard, and optional prediction-history logging to PostgreSQL.

Out of the box it forecasts **housing prices per metro area** from real, public market data updated monthly (Redfin Data Center + Freddie Mac mortgage rates via FRED — no API keys required; ZIP, county, city, and state levels also supported). Point it at your own dataset by editing one config file.

## 🚀 Features

- **Config-Driven, Dataset-Agnostic Pipeline**: `config/dataset.yaml` defines the date, target, entity, and feature columns. Feature engineering, preprocessing, training, the API, and the dashboard all read from it — swap datasets without touching code.
- **Real Public Data Ingestion**: `make fetch-data` downloads housing market data published by Redfin (metro level by default — a ~110 MB download, cached; ZIP/county/city/state via `LEVEL=`) and 30-year mortgage rates from FRED, normalizes them to the canonical schema, and validates the result. Filter by state or location; no API keys needed.
- **Schema Validation**: Every dataset load is validated against the active config with actionable errors (missing columns, non-numeric targets, duplicate entity-period rows).
- **Leakage-Safe Temporal Features**: Lags and rolling means grouped by entity (`shift(1)` before rolling, so a value never enters its own window), calendar features, and cyclical `sin/cos` month encodings. Warm-up rows are dropped, never imputed.
- **Time-Aware Splitting**: Chronological train/validation/test splits — no shuffling, no future data in training windows.
- **Deep Learning + Baseline**: A configurable Keras dense network benchmarked against a scikit-learn Ridge baseline on MAE, RMSE, MAPE, and R².
- **Optional KerasTuner Tuning**: A small, CPU-friendly RandomSearch over layer count, units, dropout, learning rate, and L2 (`make tune`). Best hyperparameters are saved and picked up automatically by the next `make train`.
- **FastAPI Inference Service**: `/predict`, `/batch-predict` (single vectorized model call), `/model-metadata`, `/dataset-schema`, `/entities`, `/health`, and `/prediction-history`. Optional API-key auth via the `API_KEY` environment variable.
- **Optional PostgreSQL Prediction Logging**: Set `DATABASE_URL` and every prediction is logged (timestamp, model version, entity, horizon, request/adjustments/prediction/warnings JSON, latency). Logging is fail-soft — inference keeps working if the database is down or unconfigured.
- **Multi-Page Streamlit Dashboard**: Data explorer, training trigger, forecast viewer with pre-filled lag features, model evaluation, what-if scenario simulator, and prediction history — all rendered from the dataset config.
- **Docker-First Reproducibility**: Every command runs through `docker-compose` via the Makefile; native Apple Silicon support via `tensorflow-macos`.

## 📂 Architecture

```text
trendpilot-ai/
├── api/                  # FastAPI inference service (routes, schemas, model service)
├── app/                  # Streamlit multi-page dashboard
├── config/               # Dataset config (schema, features, windows)
├── data/                 # raw / processed / sample datasets (raw+processed gitignored)
├── models/               # Saved Keras artifacts, preprocessor, tuning metadata (gitignored)
├── src/
│   ├── config.py         # Dataset config loading + dataframe validation
│   ├── data/             # Ingestion (Redfin + FRED), loaders, feature engineering
│   ├── modeling/         # Keras model factory, training, tuning, baseline, evaluation
│   ├── db/               # Optional SQLAlchemy prediction-history logging
│   └── utils/            # Path management
├── tests/                # Pytest suite (unit + integration + offline fixtures)
├── Dockerfile.api        # Backend image (installs dev + tuning + db extras)
├── Dockerfile.dashboard  # Frontend image
├── docker-compose.yml    # api + dashboard + optional postgres (profile "db")
└── Makefile              # Command orchestration
```

Core logic lives entirely in `src/` and is consumed by both services — the API and dashboard contain no ML code.

## 🛠️ Quickstart (Docker)

```bash
make setup     # build images, fetch real housing data (Redfin + FRED), train the model
make launch    # start API + dashboard and open the app in your browser
```

- Dashboard: http://localhost:9174
- API docs (Swagger): http://localhost:9173/docs

The individual steps remain available (`make build`, `make fetch-data`, `make train`, `make up`), plus `make open` to reopen the dashboard without restarting anything.

Host ports default to uncommon values (9173/9174, and 9175 for the optional Postgres) to avoid clashing with services commonly bound to 3000/8000/8501/5432. Override with `API_PORT`, `DASHBOARD_PORT`, and `POSTGRES_PORT` (environment or `.env`).

`make up` starts the dashboard only after the API passes its health check; the TensorFlow model loads in a background thread, so `/health` responds immediately and reports `model_loading` until warm-up completes.

`make fetch-data` defaults to **metro-level** data (a ~110 MB download, cached for re-runs; keeps the 50 metros with the densest history). Adjust the scope:

```bash
make fetch-data STATES="CA TX"                 # only California and Texas metros
make fetch-data LEVEL=state MAX_ENTITIES=0     # all 50 states (~5 MB download)
make fetch-data LEVEL=zip STATES="WA"          # ZIP-code level (~1 GB download)
# or run the ingester directly for full control:
docker-compose run --rm api python src/data/ingest.py --locations "Seattle, WA" "Portland, OR"
```

All levels produce the same canonical schema (`region`, `location`, drivers, target), so no config change is needed when switching. Downloads are **resumable and cached** under `data/raw/.cache/` — if a transfer is interrupted, re-running `make fetch-data` picks up where it left off, and subsequent runs reuse the cached file. Pass `--refresh` to the ingester to force a fresh copy. Column matching is case-insensitive, so header-casing changes in the export don't break parsing.

Housing data © [Redfin](https://www.redfin.com/news/data-center/); mortgage rates from Freddie Mac via [FRED](https://fred.stlouisfed.org/series/MORTGAGE30US). Cite the sources when publishing results.

### Example prediction

```bash
curl -X POST http://localhost:9173/predict -H "Content-Type: application/json" -d '{
  "features": {
    "region": "WA", "location": "Seattle, WA",
    "inventory": 1500, "mortgage_rate": 6.5, "days_on_market": 30,
    "sale_to_list_ratio": 1.0, "new_listings": 300,
    "month": 6, "quarter": 2, "year": 2026,
    "median_sale_price_lag_1": 800000, "median_sale_price_lag_2": 795000,
    "median_sale_price_lag_3": 790000, "median_sale_price_rolling_mean_3": 795000,
    "median_sale_price_rolling_mean_6": 780000
  },
  "horizon": 3,
  "scenario_adjustments": {"mortgage_rate": 7.5}
}'
```

The required feature columns are dataset-dependent — `GET /dataset-schema` and `GET /model-metadata` report exactly what the active config and trained model expect.

## 🔧 Bring Your Own Dataset

Any panel dataset works: one row per (entity, date), a numeric target, and optional numeric/categorical driver features.

1. Place your CSV at `data/raw/<your_file>.csv`.
2. Describe its schema in `config/dataset.yaml` (or a copy referenced by the `TRENDPILOT_CONFIG` environment variable):

```yaml
name: store_sales
data_file: store_sales.csv
date_col: date
target_col: weekly_sales
entity_col: store_id
entity_hierarchy: [region, store_id]
categorical_features: [region, store_id]
numeric_features: [promotions, footfall, avg_basket_size]
lags: [1, 2, 4]
rolling_windows: [4, 8]
```

3. Run `make train` — validation, feature engineering, training, the API, and the dashboard adapt automatically.

## 🎛️ Optional: Hyperparameter Tuning (KerasTuner)

```bash
make tune     # small RandomSearch (default: 8 trials, CPU-friendly)
make train    # automatically trains with the tuned hyperparameters
```

Results are written to `models/metadata/tuning/` (`best_hyperparameters.json`, `tuning_summary.json`). Control the budget with `TUNER_MAX_TRIALS` and `TUNER_EPOCHS`.

## 🗄️ Optional: PostgreSQL Prediction Logging

1. Copy `.env.example` to `.env` (never commit `.env`).
2. Set `DATABASE_URL` — either a managed PostgreSQL instance, or the bundled local one:

```bash
docker-compose --profile db up -d    # local postgres on host port 9175
export DATABASE_URL=postgresql+psycopg2://trendpilot:trendpilot@db:5432/trendpilot
make api
```

(The URL above uses `db:5432` — the in-network address. From the host, connect via `localhost:9175`.)

Predictions then appear at `GET /prediction-history` and on the dashboard's **Prediction History** page. Without `DATABASE_URL`, logging is silently disabled and inference is unaffected.

## 🔐 Optional: API-Key Auth

Set `API_KEY` in the environment and every endpoint except `/health` and the docs requires a matching `X-API-Key` header. Leave it unset for open local development.

## 🧪 Testing and Code Quality

```bash
make test              # pytest: ingestion, schema validation, leakage guards, API, DB logging
make lint              # ruff + black --check
make format            # ruff --fix + black
make clean             # remove Python caches (safe — never touches data or models)
make clean-artifacts   # remove fetched data + trained models (download cache is kept)
```

Notable tests: features at time *t* are proven identical with and without future rows present; perturbing future targets cannot change past feature rows; ingestion normalization runs offline against fixtures; DB logging round-trips through SQLite and fails soft on a bad URL. CI runs lint, tests, and Docker image builds.

## 📝 Design Decisions

1. **Leakage prevention as a first-class concern**: grouped `shift(1)` rolling windows, warm-up rows dropped instead of imputed, chronological splits, and regression tests that assert temporal isolation.
2. **One schema definition**: the dataset config is the single source of truth consumed by ingestion, training, serving, and UI — changing datasets is a config edit, not a refactor.
3. **Separation of concerns**: `src/` holds all domain logic; FastAPI only serves inference/metadata; Streamlit only renders UI.
4. **Graceful degradation**: the API boots without a trained model (503 on predict), tuning is optional, auth is optional, and DB logging never blocks inference.
5. **Reproducibility**: pinned Python version, Docker-first workflow, deterministic tuning seed, and operational metadata — every trained model records its hyperparameters, feature columns, split sizes, and metrics; every logged prediction records model version and latency.

## ⚠️ Limitations

- The model forecasts one step given engineered features; `horizon` is recorded metadata rather than a true multi-step recursive forecast.
- Lag features are supplied by the client (pre-filled by the dashboard); computing them server-side from stored history (a lightweight feature store) is the natural next step.
- Redfin metrics are rolling windows reported monthly (90-day windows at finer geographic levels); treat month-over-month changes accordingly.

## 📄 License

See [LICENSE](LICENSE).

# Trendpilot AI

Trendpilot AI is a forecasting project built with TensorFlow/Keras, FastAPI,
and Streamlit. It ingests panel data, builds leakage-safe features, compares a
neural network with a Ridge baseline, and serves predictions through an API
and dashboard. The default dataset combines Redfin housing metrics with
Freddie Mac mortgage rates from FRED. A YAML configuration defines the schema
for other datasets with one row per entity and date.

## Quickstart

Install Docker with Compose, Make, and Python 3.12 for the local key helper.
Application containers, CI, and native development use Python 3.12.16.
From the project directory:

```bash
make setup     # build images, fetch data, and train the first model
make launch    # start the API/dashboard and open the browser
```

Data ingestion needs internet access and downloads about 110 MB for the default
metro dataset. Downloads are cached and resumable. For a smaller first run,
use `make setup LEVEL=state MAX_ENTITIES=0`.

- Dashboard: [localhost:9174](http://localhost:9174)
- API documentation: [localhost:9173/docs](http://localhost:9173/docs)

The dashboard provides data exploration, training, forecasts, model evaluation,
scenario comparisons, and optional prediction history. It sends API credentials
automatically. Published local ports bind to `127.0.0.1`.

## Common commands

| Command | Purpose |
|---|---|
| `make build` | Build the API and dashboard images |
| `make up` / `make down` | Start or stop the local services |
| `make open` | Open the dashboard |
| `make fetch-data` | Fetch and validate the default dataset |
| `make train` | Train and publish a model bundle |
| `make tune` | Run the optional KerasTuner search |
| `make test` | Run pytest in the API container |
| `make lint` / `make format` | Check or format Python source |
| `make clean` | Remove Python caches |

`make clean-artifacts` deletes fetched datasets and trained models while keeping
the download cache. Use it only when you intend to rebuild those artifacts.

## Configuration

`config/dataset.yaml` defines the data file, date/entity/target columns, features,
lags, and rolling windows. For another dataset, place its CSV in `data/raw/`,
update the configuration, then run `make train`. An example schema is:

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

Set `TRENDPILOT_CONFIG` to select another configuration for native processes;
for containers, provide it through their environment and mount the file.
The existing `config/` directory is mounted by Compose.

For the default housing ingester:

```bash
make fetch-data STATES="CA TX"
make fetch-data LEVEL=state MAX_ENTITIES=0
make fetch-data LEVEL=zip STATES="WA"
# Additional filters and source overrides:
docker compose run --rm api python src/data/ingest.py --help
```

Metro is the default geographic level; state, county, city, and ZIP are also
supported. `MAX_ENTITIES=0` removes the entity limit. ZIP downloads are large.
Training accepts `EPOCHS`, `BATCH_SIZE`, and `LEARNING_RATE`; tuning accepts
`TUNER_MAX_TRIALS` and `TUNER_EPOCHS` in the training process environment.
The dashboard supplies training settings directly. Tuning results are stored in
`models/metadata/tuning/` and used by the next training run.

See `.env.example` for settings. `API_PORT`, `DASHBOARD_PORT`, and `POSTGRES_PORT`
override the host ports, which default to 9173, 9174, and 9175.

### API credentials

`make setup`, `make up`, and `make launch` initialize a random API key at
`~/.config/trendpilot-ai/api-key` if needed. The directory is mode `0700` and
the file is mode `0600`. Compose mounts the key read-only into both services;
only its path appears in environment configuration. Set
`TRENDPILOT_API_KEY_FILE` in your shell to use another absolute external path.

```bash
make api-key-init      # idempotent; preserves an existing key
make api-key-status    # checks storage without displaying the key
make api-key-rotate    # rotates and recreates both consumers
```

Keep real credentials outside the checkout, including `.env`. The local key is
private plaintext, accessible to its owner, root, and Docker; use disk encryption
and avoid shared backups. Atomic rotation requires container recreation, since
restarting a container can retain the old mounted file.

The API accepts `API_KEY_FILE` or externally injected `API_KEY`. Invalid configured
files fail closed; `API_KEY_REQUIRED=true` also rejects absent or empty keys.
Protected routes require `X-API-Key`. `/health`, `/ready`, and API documentation
remain public. For an authenticated request after starting Compose:

```bash
docker compose exec -T api python - <<'PYTHON'
import requests
from app.api_client import api_headers

response = requests.get(
    "http://localhost:8000/dataset-schema", headers=api_headers(), timeout=10
)
response.raise_for_status()
print(response.json())
PYTHON
```

For native development, install uv 0.13.8 and run:

```bash
uv sync --locked --all-extras
uv run --no-sync python scripts/api_key.py run -- uvicorn api.main:app --reload
# In another terminal, for the dashboard:
uv run --no-sync python scripts/api_key.py run -- streamlit run app/main.py
```

uv selects Python from `.python-version` and creates a local `.venv`. The launcher
passes the external credential path and requires auth; initialize the key first
with `make api-key-init`. An unconfigured native API allows unauthenticated
development. Linux containers and CI use TensorFlow 2.21. Native Apple Silicon
uses TensorFlow 2.19 with PyArrow 17 to avoid a native library conflict; Intel
Macs use TensorFlow's final compatible 2.16 release. The lock records these
platform differences.

### Optional prediction history

Set `DATABASE_URL` to enable SQLAlchemy prediction logging. For the bundled local
PostgreSQL service, after initializing the API key:

```bash
export DATABASE_URL=postgresql+psycopg2://trendpilot:trendpilot@db:5432/trendpilot
docker compose --profile db up -d --build
```

These database credentials are disposable local defaults. Use a private external
secret store for real credentials. Containers connect to `db:5432`; host tools use
`localhost:9175`. Without a database URL, logging is disabled. Database failures
do not block inference.

## API and model behavior

The API exposes `/predict`, `/batch-predict`, `/model-metadata`, `/dataset-schema`,
`/entities`, and `/prediction-history`. The active dataset configuration determines
the required prediction columns. `/health` reports process liveness; `/ready`
returns 503 until a validated model is loaded. Compose waits for process health,
so the dashboard can start and train the first model on an empty installation.

Training validates a serialized prediction and publishes an immutable model,
preprocessor, metadata, and checksum bundle under `models/bundles/`. An atomic
`models/current.json` replacement selects the active bundle. Training and ingestion
CLI commands use a shared writer lock. The API loads complete snapshots in the
background and keeps its working model if a replacement fails. Legacy model files
remain readable until the first bundle is published. Model bundles are retained;
storage cleanup and rollback require deliberate coordination.

## Tests

Run `make build`, then `make lint` and `make test`. Tests use offline fixtures and
temporary storage, covering data validation, temporal leakage, preprocessing, API
authentication, model publication/recovery, writer exclusion, and database logging.
They do not require live data downloads. To run the same Python checks natively:

```bash
uv run --no-sync ruff check src api app tests scripts
uv run --no-sync black --check src api app tests scripts
uv run --no-sync pytest
```

`uv.lock` is the dependency lock for development, CI, and both Docker images,
including the optional tuning/database dependencies. After an intentional change
to `pyproject.toml`, run `uv lock`, sync again, and rerun the checks. Use
`uv lock --upgrade` only for an intentional dependency refresh. The build backend
is pinned separately in `pyproject.toml`.

The GitHub Actions workflow runs these checks, builds both images, renders/checks
both Kubernetes overlays, and runs the disposable ingress smoke test. Registry
publishing and cluster deployment are separate operations.

## Optional Kubernetes deployment

Kustomize bases/overlays and Argo CD Applications are in `deploy/`. See
[deploy/README.md](deploy/README.md) for prerequisites, storage, secrets, ingress,
rendering checks, migration, and first-model bootstrap.

## Limitations

- Forecasts are one-step predictions from supplied features. `horizon` is recorded
  metadata, not a recursive multi-step forecast.
- Clients supply lag/rolling features; the dashboard fills them from the dataset.
- Training uses the dashboard container's resources and runs one writer at a time.
- Redfin metrics use rolling windows, so month-over-month changes need care.

## Data attribution and license

Housing metrics are from [Redfin Data Center](https://www.redfin.com/news/data-center/).
Mortgage rates are Freddie Mac's MORTGAGE30US series, accessed through
[FRED](https://fred.stlouisfed.org/series/MORTGAGE30US). Preserve source attribution
when sharing results and check the providers' terms for your intended use.

Project code is licensed under the [MIT License](LICENSE), copyright 2026 Jonathan
Serrano. External datasets, dependencies, and third-party materials retain their
own licenses and usage terms; the software license does not relicense them.

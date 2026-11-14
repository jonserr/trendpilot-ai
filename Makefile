.DEFAULT_GOAL := install

.PHONY: install setup launch open lint format test fetch-data train tune api dashboard build up down clean clean-artifacts api-key-init api-key-status api-key-rotate

PYTHON ?= python3
COMPOSE ?= docker compose

# Idempotent; creates only this project's external credential, never a repo .env.
api-key-init:
	$(PYTHON) scripts/api_key.py init

api-key-status:
	$(PYTHON) scripts/api_key.py status

# Recreate both consumers after rotation; `restart` may retain the old file mount.
api-key-rotate:
	$(PYTHON) scripts/api_key.py rotate
	$(COMPOSE) up -d --build --force-recreate api dashboard

lint format test fetch-data train tune api dashboard up: api-key-init

# Scope for `make fetch-data` (Redfin market data + FRED rates).
# LEVEL: metro (default, ~110 MB) | state (~5 MB) | county | city | zip (~1 GB)
LEVEL ?= metro
STATES ?=
MAX_ENTITIES ?= 50

# Host ports (exported so Compose sees overrides).
API_PORT ?= 9173
DASHBOARD_PORT ?= 9174
export API_PORT DASHBOARD_PORT

# Cross-platform "open a URL": macOS `open`, Linux `xdg-open`.
OPEN_CMD := $(shell command -v open 2>/dev/null || command -v xdg-open 2>/dev/null || echo echo)

install: build

# One-shot first-time setup: build images, fetch real data, train the model.
setup: build fetch-data train
	@echo ""
	@echo "Setup complete. Run 'make launch' to start the app and open it in your browser."

# Start everything and open the dashboard in the default browser.
launch: up
	@echo "Waiting for the dashboard at http://localhost:$(DASHBOARD_PORT) ..."
	@for i in $$(seq 1 60); do \
		curl -fsS -o /dev/null http://localhost:$(DASHBOARD_PORT) 2>/dev/null && break; \
		sleep 1; \
	done
	$(OPEN_CMD) http://localhost:$(DASHBOARD_PORT)

# Open the dashboard without (re)starting containers.
open:
	$(OPEN_CMD) http://localhost:$(DASHBOARD_PORT)

lint:
	$(COMPOSE) run --rm api ruff check src api app tests scripts
	$(COMPOSE) run --rm api black --check src api app tests scripts

format:
	$(COMPOSE) run --rm api ruff check --fix src api app tests scripts
	$(COMPOSE) run --rm api black src api app tests scripts

test:
	$(COMPOSE) run --rm api pytest

fetch-data:
	$(COMPOSE) run --rm api python src/data/ingest.py --level $(LEVEL) $(if $(STATES),--states $(STATES),) --max-entities $(MAX_ENTITIES)

train:
	$(COMPOSE) run --rm api python src/modeling/train.py

tune:
	$(COMPOSE) run --rm api python src/modeling/tuning.py

api:
	$(COMPOSE) up -d --build api

dashboard:
	$(COMPOSE) up -d --build dashboard

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

# Safe: removes Python caches only. Never touches datasets or trained models.
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# DESTRUCTIVE: removes the fetched dataset and all trained model artifacts.
# The Redfin download cache (data/raw/.cache) is preserved, so a subsequent
# `make fetch-data` rebuilds the dataset without re-downloading.
clean-artifacts:
	@echo "Removing datasets and trained model artifacts (download cache is kept)..."
	rm -rf data/raw/* data/processed/* models/saved/* models/metadata/* models/bundles models/current.json
	touch data/raw/.gitkeep data/processed/.gitkeep models/saved/.gitkeep models/metadata/.gitkeep
	@echo "Done. Run 'make fetch-data && make train' to rebuild."

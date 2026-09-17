# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
UV ?= $(shell which uv 2>/dev/null || echo ~/.local/bin/uv)

# Parameters
CAMPAIGN ?= 1458
START_DATE ?=
END_DATE ?=
PROGRAMS ?= 200
TASK_ID ?=

.DEFAULT_GOAL := help

.PHONY: help setup auth download-data prepare-data run report oracle test mask mask-dry unmask clean

help: ## Show available Makefile targets
	@echo "=================================================================="
	@echo "🚀 AlphaEvolve Advertiser-Side Auto-Bidding Project"
	@echo "=================================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage examples:"
	@echo "  make setup                                   # Set up virtual environment and install dependencies"
	@echo "  make auth                                    # Login to GCP Application Default Credentials"
	@echo "  make download-data                           # Acquire or verify raw iPinYou dataset archive"
	@echo "  make prepare-data [CAMPAIGN=1458]            # Clean, split, and calibrate dataset for campaign"
	@echo "  make prepare-data START_DATE=20130606 END_DATE=20130612"
	@echo "  make run programs=200                        # Run AlphaEvolve evolution loop"
	@echo "  make oracle                                  # Run Day 7 Held-out Oracle benchmark on 447k auctions"
	@echo "  make report                                  # Generate interactive bilingual HTML reports"
	@echo "  make test                                    # Run full unit and integration test suite"
	@echo "  make mask                                    # Mask personal GCP credentials before sharing/pushing"
	@echo "  make unmask                                  # Restore personal GCP credentials for local development"

setup: ## Install project dependencies into virtual environment
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating Python virtual environment in $(VENV)..."; \
		python3 -m venv $(VENV); \
	fi
	@echo "Installing dependencies from requirements.txt..."
	@if [ -x "$(UV)" ]; then \
		$(UV) pip install --python $(PYTHON) -r requirements.txt; \
	else \
		$(PYTHON) -m pip install --upgrade pip && $(PYTHON) -m pip install -r requirements.txt; \
	fi
	@if [ ! -f config.yaml ]; then \
		echo "⚠️ config.yaml not found! Please check your configuration."; \
	else \
		echo "✅ config.yaml verified."; \
	fi
	@echo "Setup complete! Run 'make auth' then 'make prepare-data'."

auth: ## Authenticate with Google Cloud Application Default Credentials
	@echo "Authenticating with Google Cloud ADC..."
	gcloud auth application-default login

download-data: ## Download or acquire raw iPinYou RTB benchmark dataset (optional URL=...)
	@echo "Acquiring raw iPinYou dataset..."
	@$(PYTHON) -m src.data.download_data $(if $(URL),--url $(URL),) $(if $(SOURCE),--source $(SOURCE),) $(if $(FORCE),--force,)

prepare-data: ## Clean, split and calibrate data (supports CAMPAIGN=<id>, START_DATE=YYYYMMDD, END_DATE=YYYYMMDD)
	@echo "Preparing and calibrating RTB dataset for Campaign $(CAMPAIGN)..."
	@$(PYTHON) -m src.data.prepare_data --campaign-id $(CAMPAIGN) $(if $(START_DATE),--start-date $(START_DATE),) $(if $(END_DATE),--end-date $(END_DATE),)

pipeline: prepare-data ## Legacy alias for prepare-data

run: ## Run AlphaEvolve experiment (supports 'make run programs=N' or 'make run task_id=...')
	@echo "Starting AlphaEvolve evolution..."
	@PYTHONPATH=. $(PYTHON) -m src.run_evolution $(if $(programs),--programs $(programs),$(if $(PROGRAMS),--programs $(PROGRAMS),)) $(if $(task_id),--task-id $(task_id),$(if $(TASK_ID),--task-id $(TASK_ID),))

report: ## Generate HTML evolution report (supports 'make report' or 'make report task_id=...')
	@echo "Generating HTML evolution report..."
	@PYTHONPATH=. $(PYTHON) -m src.report $(if $(task_id),--task-id $(task_id),$(if $(TASK_ID),--task-id $(TASK_ID),))

oracle: ## Run full baseline benchmark and Day 7 Oracle test set verification (supports 'make oracle task_id=...')
	@echo "Running Oracle ground-truth benchmark on Day 7 held-out test data..."
	@PYTHONPATH=. $(PYTHON) -m src.bidding.benchmark $(if $(task_id),--task-id $(task_id),$(if $(TASK_ID),--task-id $(TASK_ID),))

test: ## Run unit and integration test suite
	@echo "Running tests..."
	@PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

mask: ## Mask personal GCP credentials across project files
	@python3 scripts/mask_credentials.py

mask-dry: ## Dry-run preview of credentials masking
	@python3 scripts/mask_credentials.py --dry-run

unmask: ## Restore personal GCP credentials for local development
	@python3 scripts/mask_credentials.py --restore $(if $(PROJECT_ID),--project-id $(PROJECT_ID),) $(if $(APP_ID),--app-id $(APP_ID),)

clean: ## Clean up Python cache and temporary files
	@echo "Cleaning up build artifacts and caches..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true

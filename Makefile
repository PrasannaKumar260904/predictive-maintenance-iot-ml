.PHONY: setup install format lint test train api app docker-build docker-run help

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

setup: ## Set up Python virtual environment and dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

format: ## Format code using black, isort, and ruff
	$(BIN)/ruff check --fix .
	$(BIN)/isort .
	$(BIN)/black .

lint: ## Run lint checks
	$(BIN)/ruff check .
	$(BIN)/black --check .

test: ## Run unit tests with coverage
	$(BIN)/pytest --cov=src --cov=api --cov-report=term-missing --cov-report=html tests/

train: ## Train all models and generate artifacts
	$(BIN)/python src/models/train.py

api: ## Run FastAPI REST API server
	$(BIN)/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app: ## Run Streamlit interactive dashboard
	$(BIN)/streamlit run app/streamlit_app.py --server.port 8501

docker-build: ## Build Docker container
	docker build -t predictive-maintenance-iot:latest .

docker-run: ## Run Docker containers using docker-compose
	docker-compose up --build

help: ## Show Makefile target options
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

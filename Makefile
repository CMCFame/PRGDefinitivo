# Makefile for Progol Engine
# Usage: make [target]

.PHONY: help setup install test clean run-etl run-model run-optimizer run-all docker-up docker-down dashboard airflow lint format

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
DOCKER_COMPOSE := docker-compose
STREAMLIT := streamlit
AIRFLOW := airflow

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "${BLUE}Progol Engine - Available Commands${NC}"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "${GREEN}%-20s${NC} %s\n", $$1, $$2}'

# Setup and Installation
setup: ## Run initial setup script
	@echo "${BLUE}Running initial setup...${NC}"
	$(PYTHON) setup.py

install: ## Install all dependencies
	@echo "${BLUE}Installing dependencies...${NC}"
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "${BLUE}Installing development dependencies...${NC}"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 isort mypy

venv: ## Create virtual environment
	@echo "${BLUE}Creating virtual environment...${NC}"
	$(PYTHON) -m venv venv
	@echo "${GREEN}Virtual environment created. Activate with: source venv/bin/activate${NC}"

# Data Pipeline
run-etl: ## Run ETL pipeline manually
	@echo "${BLUE}Running ETL pipeline...${NC}"
	$(PYTHON) -m src.etl.ingest_csv
	$(PYTHON) -m src.etl.scrape_odds
	$(PYTHON) -m src.etl.parse_previas
	$(PYTHON) -m src.etl.build_features

run-model: ## Run model training pipeline
	@echo "${BLUE}Running model training...${NC}"
	$(PYTHON) -m src.modeling.poisson_model
	$(PYTHON) -m src.modeling.stacking
	$(PYTHON) -m src.modeling.bayesian_adjustment
	$(PYTHON) -m src.modeling.draw_propensity

run-optimizer: ## Run portfolio optimization
	@echo "${BLUE}Running portfolio optimization...${NC}"
	$(PYTHON) -m src.optimizer.classify_matches
	$(PYTHON) -m src.optimizer.generate_core
	$(PYTHON) -m src.optimizer.generate_satellites
	$(PYTHON) -m src.optimizer.grasp
	$(PYTHON) -m src.optimizer.annealing
	$(PYTHON) -m src.optimizer.checklist

run-simulation: ## Run Monte Carlo simulation
	@echo "${BLUE}Running simulation...${NC}"
	$(PYTHON) -m src.simulation.montecarlo_sim

run-all: run-etl run-model run-optimizer run-simulation ## Run complete pipeline
	@echo "${GREEN}Complete pipeline executed successfully!${NC}"

# Testing
test: ## Run all tests
	@echo "${BLUE}Running tests...${NC}"
	$(PYTEST) tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "${BLUE}Running tests with coverage...${NC}"
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

test-etl: ## Run ETL tests only
	$(PYTEST) tests/test_etl.py -v

test-model: ## Run model tests only
	$(PYTEST) tests/test_modeling.py -v

test-optimizer: ## Run optimizer tests only
	$(PYTEST) tests/test_optimizer.py -v

# Code Quality
lint: ## Run linting checks
	@echo "${BLUE}Running linting checks...${NC}"
	flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__
	mypy src/ --ignore-missing-imports

format: ## Format code with black and isort
	@echo "${BLUE}Formatting code...${NC}"
	black src/ tests/
	isort src/ tests/

check-format: ## Check if code is formatted
	@echo "${BLUE}Checking code format...${NC}"
	black --check src/ tests/
	isort --check-only src/ tests/

# Docker
docker-build: ## Build Docker images
	@echo "${BLUE}Building Docker images...${NC}"
	$(DOCKER_COMPOSE) build

docker-up: ## Start all services with Docker Compose
	@echo "${BLUE}Starting Docker services...${NC}"
	$(DOCKER_COMPOSE) up -d
	@echo "${GREEN}Services started! Access:${NC}"
	@echo "  - Airflow: http://localhost:8080"
	@echo "  - Streamlit: http://localhost:8501"
	@echo "  - Jupyter: http://localhost:8888"

docker-down: ## Stop all Docker services
	@echo "${BLUE}Stopping Docker services...${NC}"
	$(DOCKER_COMPOSE) down

docker-logs: ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-clean: ## Clean Docker volumes and images
	@echo "${RED}Warning: This will delete all data!${NC}"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(DOCKER_COMPOSE) down -v --rmi all; \
	fi

# Services
dashboard: ## Run Streamlit dashboard
	@echo "${BLUE}Starting Streamlit dashboard...${NC}"
	$(STREAMLIT) run streamlit_app/dashboard.py

airflow-init: ## Initialize Airflow database
	@echo "${BLUE}Initializing Airflow...${NC}"
	$(AIRFLOW) db init
	$(AIRFLOW) users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com

airflow-webserver: ## Start Airflow webserver
	@echo "${BLUE}Starting Airflow webserver...${NC}"
	$(AIRFLOW) webserver --port 8080

airflow-scheduler: ## Start Airflow scheduler
	@echo "${BLUE}Starting Airflow scheduler...${NC}"
	$(AIRFLOW) scheduler

jupyter: ## Start Jupyter notebook
	@echo "${BLUE}Starting Jupyter notebook...${NC}"
	jupyter notebook notebooks/

# Data Management
clean-data: ## Clean all processed data (careful!)
	@echo "${YELLOW}Warning: This will delete all processed data!${NC}"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/processed/*; \
		rm -rf data/dashboard/*; \
		rm -rf data/reports/*; \
		echo "${GREEN}Data cleaned${NC}"; \
	fi

clean-models: ## Clean all trained models
	@echo "${YELLOW}Warning: This will delete all trained models!${NC}"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo ""; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/poisson/*; \
		rm -rf models/bayes/*; \
		echo "${GREEN}Models cleaned${NC}"; \
	fi

clean-logs: ## Clean log files
	@echo "${BLUE}Cleaning log files...${NC}"
	rm -rf logs/*
	find . -name "*.log" -type f -delete
	@echo "${GREEN}Logs cleaned${NC}"

clean-cache: ## Clean Python cache files
	@echo "${BLUE}Cleaning cache files...${NC}"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "${GREEN}Cache cleaned${NC}"

clean: clean-cache clean-logs ## Clean cache and logs (safe)

clean-all: clean clean-data clean-models ## Clean everything (careful!)

# Utilities
check-env: ## Check if .env file exists
	@if [ ! -f .env ]; then \
		echo "${RED}Error: .env file not found!${NC}"; \
		echo "${YELLOW}Run 'make setup' or copy .env.template to .env${NC}"; \
		exit 1; \
	else \
		echo "${GREEN}.env file found${NC}"; \
	fi

validate-data: ## Validate input data files
	@echo "${BLUE}Validating data files...${NC}"
	@$(PYTHON) -c "import os; \
		files = ['data/raw/Progol.csv', 'data/raw/odds.csv']; \
		missing = [f for f in files if not os.path.exists(f)]; \
		if missing: \
			print('${RED}Missing files: ' + ', '.join(missing) + '${NC}'); \
			exit(1); \
		else: \
			print('${GREEN}All required data files found${NC}')"

generate-sample-data: ## Generate sample data for testing
	@echo "${BLUE}Generating sample data...${NC}"
	$(PYTHON) -c "from setup import create_sample_data; create_sample_data()"
	@echo "${GREEN}Sample data generated in data/raw/${NC}"

# Development
dev-install: install-dev ## Alias for install-dev

dev-server: ## Run all services for development
	@echo "${BLUE}Starting development servers...${NC}"
	make dashboard &
	make airflow-webserver &
	make airflow-scheduler &
	@echo "${GREEN}Development servers started${NC}"

# Documentation
docs: ## Generate documentation
	@echo "${BLUE}Generating documentation...${NC}"
	sphinx-build -b html docs/ docs/_build/html

docs-serve: ## Serve documentation locally
	@echo "${BLUE}Serving documentation...${NC}"
	cd docs/_build/html && python -m http.server 8000

# Quick commands
quick-test: test-coverage lint ## Run tests and linting

quick-run: check-env validate-data run-all dashboard ## Quick run with dashboard

# Git hooks
install-hooks: ## Install git pre-commit hooks
	@echo "${BLUE}Installing git hooks...${NC}"
	pre-commit install
	@echo "${GREEN}Git hooks installed${NC}"

# Version and info
version: ## Show version information
	@echo "${BLUE}Progol Engine v1.0${NC}"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"

info: ## Show project information
	@echo "${BLUE}Progol Engine - Project Information${NC}"
	@echo ""
	@echo "Project structure:"
	@echo "  - ETL Pipeline: src/etl/"
	@echo "  - Models: src/modeling/"
	@echo "  - Optimizer: src/optimizer/"
	@echo "  - Dashboard: streamlit_app/"
	@echo ""
	@echo "Data locations:"
	@echo "  - Raw data: data/raw/"
	@echo "  - Processed: data/processed/"
	@echo "  - Models: models/"
	@echo ""
	@echo "Run 'make help' for available commands"
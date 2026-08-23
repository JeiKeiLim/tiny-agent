.DEFAULT_GOAL := help

UV := uv

.PHONY: help install sync format lint typecheck test coverage check clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-12s %s\n", $$1, $$2}'

install: ## Install all dependencies (runtime + dev)
	$(UV) sync --all-groups

sync: install

format: ## Auto-format and fix lint issues
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

lint: ## Lint (ruff check + format check)
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

typecheck: ## Static type check (mypy, strict)
	$(UV) run mypy src

test: ## Run unit tests
	$(UV) run pytest

coverage: ## Run unit tests with coverage
	$(UV) run pytest --cov=kestrel --cov-report=term-missing

check: lint typecheck test ## Run all checks (lint + format + typecheck + test)

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} +

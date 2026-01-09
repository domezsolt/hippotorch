.PHONY: help compare-cartpole sensitivity-cartpole install-dev install-extras install-extras-wheels lint format test test-cov typecheck

SEEDS ?= 3
STEPS ?= 20000
BATCH ?= 256
WARMUP ?= 50000

help:
	@echo "Targets:"
	@echo "  compare-cartpole       # Run head-to-head and multi-seed comparisons"
	@echo "  sensitivity-cartpole   # Grid sweep (mixture, temperature, momentum, budget)"
	@echo "  install-dev            # Install dev deps (pytest, ruff, black, etc.)"
	@echo "  install-extras         # Install optional env packages (gymnasium, minigrid, robotics)"
	@echo "  install-extras-wheels  # Install gymnasium wheels only (skip minigrid/robotics)"
	@echo "  lint                   # Run ruff, black (check), isort (check)"
	@echo "  format                 # Auto-format with ruff --fix, black, isort"
	@echo "  test                   # Run test suite"
	@echo "  test-cov               # Run tests with coverage"
	@echo "  typecheck              # Run mypy on hippotorch"
	@echo "Variables: SEEDS, STEPS, BATCH, WARMUP"

compare-cartpole:
	SEEDS=$(SEEDS) STEPS=$(STEPS) BATCH=$(BATCH) WARMUP=$(WARMUP) bash scripts/compare_cartpole.sh

sensitivity-cartpole:
	python -m examples.cartpole_sensitivity \
		--seeds $(SEEDS) --steps $(STEPS) --batch-size $(BATCH) \
		--mixture-ratios 0.0 0.2 0.4 0.6 0.8 \
		--temperatures 0.05 0.07 0.1 \
		--momentums 0.9 0.99 0.995 \
		--cons-intervals 500 1000 \
		--cons-steps-list 50 100 \
		--aggregate --out results/cartpole_sensitivity.csv
	@echo "Sensitivity results written to results/cartpole_sensitivity.csv"

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

install-extras:
	pip install -r requirements-extras.txt || true
	@echo "Note: If pygame build fails for minigrid, install system SDL2 (e.g., libsdl2-dev) or skip minigrid."

install-extras-wheels:
	pip install "gymnasium==0.29.*" --only-binary=:all:
	@echo "Installed Gym only. For minigrid/robotics, see requirements-extras.txt and system prerequisites."

lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	ruff check --fix . || true
	black .
	isort .

test:
	pytest -q

test-cov:
	pytest --cov=hippotorch --cov-report=term-missing

typecheck:
	mypy hippotorch

# Makefile for hostai — adapted from FlareSolverr's validation workflow.

.PHONY: all format reformat-ruff check fix-ruff fix test test-cov vulture \
        complexity xenon bandit pyright validate

# Default target: run validation and unit tests
all: validate test

# Format the code using ruff
format:
	ruff format --check --diff .

reformat-ruff:
	ruff format .

# Check the code using ruff
check:
	ruff check .

fix-ruff:
	ruff check . --fix

fix: reformat-ruff fix-ruff
	@echo "Updated code."

# Run unit tests (with coverage)
test:
	pytest tests

# Build the integration test image and run the local provider acceptance tests
test-local:
	docker build -f tests/integration/Dockerfile.test -t hostai-test:latest .
	uv run pytest -q -m "not slow" tests/test_local_integration.py

test-cov:
	pytest tests \
		--cov=hostai \
		--cov-report=xml \
		--cov-report=term-missing \
		--cov-fail-under=60

# Dead code detection
vulture:
	vulture src/ --exclude src/hostai/undetected_chromedriver 2>/dev/null || true

# Cyclomatic complexity
complexity:
	radon cc . -a -nc

# Average complexity (xenon)
xenon:
	xenon -b D -m B -a B src

# Security lint (high severity only to avoid known CLI false positives)
bandit:
	bandit -c pyproject.toml -r src -lll

# Full security lint (may be noisy for a CLI tool)
bandit-all:
	bandit -c pyproject.toml -r src

# Type checking (strict)
pyright:
	pyright src/

# Full validation suite (format, lint, complexity, security, typecheck)
validate: format check complexity bandit pyright
	@echo "Validation passed. Your code is ready to push."

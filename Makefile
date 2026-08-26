# Sorbed developer tasks. Assumes an activated virtualenv:
#   . .venv/bin/activate
# `make check` runs the exact gate CI enforces.

.PHONY: install lint format watchdog type test check serve clean

install:
	pip install -e ".[dev,formats,api,ml]"

lint:
	ruff check .

format:
	ruff format .

watchdog:
	python scripts/watchdog.py

type:
	mypy src/sorbed

test:
	pytest -q

check: lint watchdog test

serve:
	uvicorn sorbed.api.app:app --reload

clean:
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +

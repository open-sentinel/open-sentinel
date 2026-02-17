.PHONY: install install-dev test lint typecheck format serve clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=opensentinel --cov-report=term-missing

lint:
	ruff check opensentinel/ tests/

typecheck:
	mypy opensentinel/

format:
	ruff check --fix opensentinel/ tests/
	ruff format opensentinel/ tests/

serve:
	osentinel serve

validate:
	osentinel validate $(file)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

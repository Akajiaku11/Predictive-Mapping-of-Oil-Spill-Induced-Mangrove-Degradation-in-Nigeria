.PHONY: format lint test

format:
	python -m pip install ruff black || true
	ruff check --select I --fix . || true
	black . || true

lint:
	python -m pip install ruff || true
	ruff check . || true

test:
	python -m pytest -q

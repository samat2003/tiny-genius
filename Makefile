.PHONY: test lint smoke

test:
	python -m pytest

lint:
	python -m ruff check src tests scripts

smoke:
	python scripts/smoke.py

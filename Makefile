.PHONY: test test-fast lint format clean

test:
	pytest tests/

test-fast:
	pytest tests/ -x

lint:
	ruff check steerling/ scripts/

format:
	ruff format steerling/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +

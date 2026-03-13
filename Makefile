.PHONY: test test-fast lint format clean
test:
	pytest tests/
test-fast:
	pytest tests/ -x
lint:
	ruff check --respect-gitignore steerling/ scripts/
format:
	ruff format --respect-gitignore steerling/ scripts/
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
.PHONY: ingest install test lint clean

ingest:
	python3 -m ingestion.ingest_local

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	flake8 ingestion/ db/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

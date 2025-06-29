.PHONY: ingest install test lint clean retrieve

ingest:
	python3 -m ingestion.ingest_local

retrieve:
	python3 -m retrieval.retriever

install:
	pip install -r requirements.txt

test:
	pytest

lint:
	flake8 ingestion/ db/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

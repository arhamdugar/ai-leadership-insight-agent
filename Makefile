.PHONY: install ingest run test lint clean help

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies
	pip install -r requirements.txt

ingest:  ## Run the document ingestion pipeline
	python scripts/ingest.py

run:  ## Launch the Streamlit app
	streamlit run app.py

test:  ## Run unit tests
	pytest tests/ -v

lint:  ## Lint the source code with ruff
	ruff check src/ ui/ scripts/

clean:  ## Remove generated data (ChromaDB + BM25 index)
	rm -rf data/chroma_db data/bm25_index.pkl

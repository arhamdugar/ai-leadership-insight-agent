# AI Leadership Insight & Decision Agent

> **Adobe GenAI Engineering Assignment** · March 2025

An agentic RAG system that ingests company documents and returns grounded, structured leadership reports — powered by Gemini 1.5 Flash, LangGraph, and Hybrid Retrieval.

---

## Architecture

```
Raw Docs ──► Loader ──► Chunker ──► Gemini Embeddings ──► ChromaDB
                                          │
                                      BM25 Index
                                          │
Query ──► Decompose ──► Dense Retrieval (ChromaDB)
                      ──► Sparse Retrieval (BM25)
                      ──► RRF Fusion
                      ──► LLM Synthesis (Gemini)
                      ──► Structured Report + Chart
```

**Task 2 — Decision Agent (LangGraph):**
```
Strategic Question ──► Decompose ──► Retrieve ──► Synthesise ──► Reflect
                                        ▲                            │
                                        └─────── if not done ────────┘
                                                          │
                                                     if done ──► END
```

---

## Project Structure

```
ai-leadership-insight-agent/
├── app.py                        ← Streamlit entry point
├── config.yaml                   ← API key + all parameters
├── requirements.txt
├── pyproject.toml                ← Packaging + ruff + pytest config
├── Makefile                      ← make install / ingest / run / test
├── .env.example                  ← Environment variable template
│
├── scripts/
│   └── ingest.py                 ← Document ingestion CLI
│
├── src/
│   └── insight_agent/            ← Python package
│       ├── config.py             ← Config singleton
│       ├── exceptions.py         ← Custom exception hierarchy
│       ├── ingestion/
│       │   ├── loader.py         ← PDF / DOCX / MD loaders
│       │   ├── chunker.py        ← Text chunker
│       │   └── pipeline.py       ← load → chunk → embed → index
│       ├── retrieval/
│       │   ├── embedder.py       ← Gemini embeddings + ChromaDB
│       │   ├── bm25.py           ← BM25Index class
│       │   └── retriever.py      ← Hybrid RRF retrieval
│       ├── reasoning/
│       │   ├── generator.py      ← LLM synthesis
│       │   └── agent.py          ← LangGraph Decision Agent
│       └── output/
│           └── charts.py         ← Auto matplotlib charts
│
├── ui/
│   ├── app.py                    ← Streamlit page + CSS + header
│   └── components/
│       ├── insight_tab.py        ← Task 1 Q&A UI
│       └── decision_tab.py       ← Task 2 Agent UI
│
├── data/
│   └── docs/                     ← Drop documents here
│       ├── FY2024_Annual_Report.md
│       ├── Q3_FY2024_Quarterly_Report.md
│       └── APAC_Strategy_Note_Oct2024.md
│
├── tests/
│   ├── conftest.py               ← Shared fixtures
│   ├── test_ingestion.py         ← Loader + chunker unit tests
│   └── test_retrieval.py         ← BM25 + RRF unit tests
│
└── validation/
    ├── qa_pairs.json             ← 10 hand-crafted Q&A pairs
    └── evaluate.py               ← LLM-as-Judge eval script
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| LLM | `gemini-1.5-flash` |
| Embeddings | `models/text-embedding-004` |
| Vector DB | ChromaDB (local, persistent) |
| Keyword search | BM25 via `rank-bm25` |
| Retrieval fusion | Reciprocal Rank Fusion (RRF) |
| PDF parsing | `pdfplumber` |
| DOCX parsing | `python-docx` |
| Agent framework | `LangGraph` |
| UI | `Streamlit` |
| Charts | `matplotlib` |
| Config | `pyyaml` + `python-dotenv` |

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd ai-leadership-insight-agent
pip install -r requirements.txt
```

Or using Make:
```bash
make install
```

### 2. Configure your Gemini API key

**Option A** — Edit `config.yaml`:
```yaml
api_key: "YOUR_GEMINI_API_KEY_HERE"
```

**Option B** — Set environment variable (takes precedence):
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your_key_here"

# macOS/Linux
export GEMINI_API_KEY="your_key_here"
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### 3. Add documents

Drop PDF, DOCX, or Markdown files into `data/docs/`.

Three sample documents are already included for demo.

### 4. Ingest (run once)

```bash
python scripts/ingest.py
# or
make ingest
```

### 5. Launch the app

```bash
streamlit run app.py
# or
make run
```

---

## Running Tests

Tests do **not** require a Gemini API key — they test pure Python logic:

```bash
pytest tests/ -v
# or
make test
```

---

## Validation

Run LLM-as-Judge evaluation on the included 10 Q&A pairs:

```bash
python validation/evaluate.py
python validation/evaluate.py --output results.json
```

---

## Configuration Reference

All parameters live in `config.yaml`:

```yaml
api_key: "YOUR_GEMINI_API_KEY_HERE"   # Or set GEMINI_API_KEY env var

llm:
  model: gemini-1.5-flash
  temperature: 0.1
  max_tokens: 2048

embedding:
  model: models/text-embedding-004

retrieval:
  top_k_dense: 10     # ChromaDB results
  top_k_sparse: 10    # BM25 results
  top_k_final: 5      # After RRF fusion

chunking:
  chunk_size: 600     # Characters (approx)
  chunk_overlap: 80
```

---

## Sample Questions

**Task 1 — Insight:**
- *"What is our current revenue trend?"*
- *"Which regions are underperforming and why?"*
- *"What are the key strategic risks for FY2025?"*
- *"How has EBITDA margin changed over the past two years?"*

**Task 2 — Decision:**
- *"Should we expand into Southeast Asia in FY2025?"*
- *"How should we respond to the APAC underperformance?"*
- *"What strategic investments should we prioritise?"*

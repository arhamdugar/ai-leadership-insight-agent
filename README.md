# AI Leadership Insight & Decision Agent

> **Open Source Project**

An agentic RAG system that ingests company documents and returns grounded, structured leadership reports — powered by Gemini 2.5 Flash, LangGraph, and Hybrid Retrieval.

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

## Methodology

### 1. Hybrid Retrieval (Sparse + Dense)
Standard RAG often misses specific keywords or technical terms. This system implements **Hybrid Retrieval** to ensure zero-loss context:
- **Dense Retrieval:** Uses `gemini-embedding-001` to find chunks based on semantic meaning (vector similarity).
- **Sparse Retrieval:** Uses `BM25` to find chunks based on exact keyword matches (lexical similarity).
- **Reciprocal Rank Fusion (RRF):** Merges both result sets using the RRF algorithm, which mathematically boosts chunks that appear high in both lists, ensuring the most relevant context reaches the LLM.

### 2. Multi-Step Agentic Reasoning (LangGraph)
Strategic questions are rarely answered by a single search. The **Decision Agent** uses a directed acyclic graph (DAG) to:
- **Decompose:** Logic-gate to break 1 complex question into 3 targeted sub-queries.
- **Synthesize & Reflect:** After generating a draft, the agent crititques its own answer. If it identifies a data gap, it loops back to retrieval for more specific evidence before finalizing.

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
| LLM (Reasoning) | `gemini-2.5-flash` |
| LLM (Fast/Intermediate) | `gemini-2.5-flash` |
| Embeddings | `models/gemini-embedding-001` |
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

**Option B** — Use a `.env` file (Recommended):
Create a file named `.env` in the project root:
```env
GEMINI_API_KEY=your_key_here
```

**Option C** — Set environment variable:
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

## Interactive Notebook Tester

If you prefer to test the agent logic directly in Python without starting the Streamlit UI, a Jupyter Notebook is provided:

1. Install Jupyter (if not already installed via `requirements.txt`):
   ```bash
   pip install jupyter
   ```
2. Launch the notebook:
   ```bash
   jupyter notebook interactive_agent_tester.ipynb
   ```

This notebook will initialize the environment, connect to Chroma/BM25, and let you run the Insight Agent (Task 1) and Decision Agent (Task 2) interactively while observing the internal LangGraph thought process in the cell outputs.

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
  model: gemini-2.5-flash
  temperature: 0.1
  max_tokens: 4096

fast_llm:
  model: gemini-2.5-flash
  temperature: 0.1
  max_tokens: 2048

embedding:
  model: models/gemini-embedding-001

retrieval:
  top_k_dense: 10     # ChromaDB results
  top_k_sparse: 10    # BM25 results
  top_k_final: 5      # After RRF fusion

chunking:
  chunk_size: 512     # Characters (approx)
  chunk_overlap: 64
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

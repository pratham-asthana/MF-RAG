# SEBI Mutual Fund RAG System

A fully local, open-source RAG (Retrieval-Augmented Generation) system built on official SEBI Mutual Fund regulatory documents. Ask questions in plain English — get grounded, cited answers. No API keys. No internet required after setup.

---

## Architecture

```
Your Question
     │
     ▼
┌─────────────────────────────────┐
│         RETRIEVAL               │
│                                 │
│  Dense Search    Keyword Search │
│  (Qdrant +       (BM25)        │
│   BGE-Large)                    │
│       │               │         │
│       └──── RRF Fusion ────┘    │
│              (top 40)           │
│                 │               │
│        Cross-Encoder Rerank     │
│              (top 5)            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         GENERATION              │
│    phi3:mini via Ollama         │
│    (runs on your GPU locally)   │
│  Answer + [SOURCE N] citations  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│         STREAMLIT UI            │
│  Chat interface + Source cards  │
└─────────────────────────────────┘
```

---

## Documents

| # | Document | Date |
|---|---|---|
| 1 | Master Circular for Mutual Funds | Mar 20, 2026 |
| 2 | Categorization and Rationalization of MF Schemes | Feb 2026 |
| 3 | MF Lite Framework for Passive Schemes | Dec 31, 2024 |
| 4 | Disclosure of Expenses, Returns, Yield & Risk-o-meter | Nov 2024 |
| 5 | Transaction Charges to MF Distributors | Aug 2025 |
| 6 | Swing Pricing Framework | Sep 2021 |

All sourced from [sebi.gov.in](https://www.sebi.gov.in)

---

## Setup & Run

### 1. Install Ollama and pull the model
Download Ollama from https://ollama.com, then:
```bash
ollama pull phi3:mini
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Place documents
Copy all SEBI PDFs into:
```
data/documents/
```

### 4. Index documents (run once)
```bash
cd src
python indexer.py
```

### 5. Launch the app
```bash
streamlit run src/app.py
```
Open http://localhost:8501 in your browser.

---

## Design Decisions

**BGE-Large embeddings** — Ranked #1 on MTEB retrieval benchmark. Runs locally, no API cost.

**Hybrid retrieval (Dense + BM25)** — Dense search finds semantically similar chunks; BM25 finds exact keyword matches (e.g. regulation codes). Combining both with RRF gives better results than either alone.

**Cross-encoder reranking** — The embedding model ranks chunks approximately. A cross-encoder scores each (query, chunk) pair precisely. Reranking the top 40 down to 5 dramatically improves what the LLM sees.

**phi3:mini** — Microsoft's 3.8B model. Specifically trained for instruction-following and factual Q&A. Ideal for constrained RAG tasks where the LLM must stay grounded in retrieved context. Runs cleanly on a 4GB GPU.

**Qdrant (local mode)** — Production-grade vector store running as local files. No server setup needed.

---

## Project Structure

```
sebi-rag/
├── data/
│   ├── documents/       ← Place SEBI PDFs here
│   ├── qdrant_store/    ← Auto-created on first index run
│   └── bm25_index.pkl   ← Auto-created on first index run
├── src/
│   ├── ingest.py        ← Reads PDFs, splits into chunks
│   ├── indexer.py       ← Embeds chunks, saves to Qdrant + BM25
│   ├── retriever.py     ← Hybrid search + reranking
│   ├── rag_pipeline.py  ← Builds prompt, calls phi3:mini, returns answer
│   └── app.py           ← Streamlit chat UI
├── requirements.txt
└── README.md
```

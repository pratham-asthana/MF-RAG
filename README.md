# SEBI Mutual Fund RAG System

A local RAG (Retrieval-Augmented Generation) chatbot for SEBI Mutual Fund documents. Ask questions and get answers with citations from official regulatory documents.

---

## Prerequisites

- Python 3.8+
- Ollama (for running the LLM locally)

---

## Setup

### 1. Install Ollama
Download from https://ollama.com and install on your system.

### 2. Pull the model
```bash
ollama pull phi3:mini
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add documents
Place all SEBI PDF documents in the `data/documents/` folder.

### 5. Index the documents (one-time setup)
```bash
python src/indexer.py
```

---

## Execution Flow

```
1. User Input (Streamlit UI)
         ↓
2. Question Retrieval
   • Dense search (BGE embeddings + Qdrant)
   • Keyword search (BM25)
   • Rerank with cross-encoder
         ↓
3. LLM Generation (phi3:mini via Ollama)
   • Ground answer in retrieved chunks
   • Add source citations
         ↓
4. Display Result
   • Show answer with [SOURCE N] citations
   • Display relevant document excerpts
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `src/app.py` | Streamlit UI — Chat interface for asking questions |
| `src/indexer.py` | Builds vector embeddings (Qdrant) and BM25 index from documents (run once) |
| `src/ingest.py` | Loads and chunks PDF documents into text segments |
| `src/retriever.py` | Hybrid retrieval system — searches dense vectors + BM25, reranks with cross-encoder |
| `src/rag_pipeline.py` | LLM pipeline — generates answers using phi3:mini with retrieved context |
| `requirements.txt` | Python dependencies |

---

## How to Run

```bash
streamlit run src/app.py
```

Open your browser and go to: **http://localhost:8501**

Start asking questions about Mutual Funds and SEBI regulations!

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

"""
indexer.py — Step 2: Embed chunks and save to vector store

What this file does:
  1. Loads all chunks from ingest.py
  2. Converts each chunk's text into a vector (list of numbers) using BGE-Large
  3. Saves all vectors into Qdrant (a local vector database)
  4. Also builds a BM25 keyword index (for hybrid search later)

Run this ONCE before using the app:
  cd src
  python indexer.py

A "vector" or "embedding" is a list of ~1024 numbers that captures the
meaning of a piece of text. Similar texts have similar vectors.
Qdrant lets us search: "give me the 20 vectors most similar to this query vector."
"""

import pickle
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from ingest import load_documents

# ── Settings ──────────────────────────────────────────────────────────────────
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"   # Best free local embedding model
COLLECTION   = "sebi_mf"                   # Name of our Qdrant collection
VECTOR_DIM   = 1024                        # BGE-Large output size
BATCH_SIZE   = 32                          # Chunks to embed at once

QDRANT_PATH  = Path(__file__).parent.parent / "data" / "qdrant_store"
BM25_PATH    = Path(__file__).parent.parent / "data" / "bm25_index.pkl"
# ──────────────────────────────────────────────────────────────────────────────


def build_vector_index(chunks: List[Dict]):
    """Embed all chunks and store in local Qdrant."""

    # Load embedding model (downloads ~1.2GB on first run, then cached)
    print(f"Loading embedding model: {EMBED_MODEL}")
    print("(Downloads ~1.2GB on first run — cached after that)\n")
    embedder = SentenceTransformer(EMBED_MODEL)

    # Set up local Qdrant (stores files on disk, no server needed)
    print(f"Setting up Qdrant vector store at: {QDRANT_PATH}")
    client = QdrantClient(path=str(QDRANT_PATH))

    # Create a fresh collection (delete old one if exists)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    # Embed chunks in batches and upload to Qdrant
    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...\n")
    points = []

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # BGE models work best with this prefix for document embedding
        prefixed_texts = [f"Represent this document for retrieval: {t}" for t in texts]
        vectors = embedder.encode(prefixed_texts, normalize_embeddings=True)

        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            points.append(PointStruct(
                id      = i + j,
                vector  = vector.tolist(),
                payload = {          # metadata stored alongside the vector
                    "id":     chunk["id"],
                    "text":   chunk["text"],
                    "source": chunk["source"],
                    "page":   chunk["page"],
                }
            ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\n✓ Saved {len(points)} vectors to Qdrant")


def build_bm25_index(chunks: List[Dict]):
    """Build a BM25 keyword index and save it to disk."""

    print("\nBuilding BM25 keyword index...")
    # BM25 works on tokenized text (just split by spaces)
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25      = BM25Okapi(tokenized)

    # Save both the index and the original chunks (needed to fetch text later)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"✓ BM25 index saved to {BM25_PATH}")


if __name__ == "__main__":
    # Step 1: Load and chunk all documents
    chunks = load_documents()

    # Step 2: Build dense vector index
    build_vector_index(chunks)

    # Step 3: Build sparse keyword index
    build_bm25_index(chunks)

    print("\n✅ Indexing complete! You can now run the app.")
    print("   streamlit run src/app.py")

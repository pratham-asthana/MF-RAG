import pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from ingest import load_documents

EMBED_MODEL  = "BAAI/bge-large-en-v1.5"   
COLLECTION   = "sebi_mf"                   
VECTOR_DIM   = 1024                        
BATCH_SIZE   = 32                          

QDRANT_PATH  = Path(__file__).parent.parent / "data" / "qdrant_store"
BM25_PATH    = Path(__file__).parent.parent / "data" / "bm25_index.pkl"


def build_vector_index(chunks: List[Dict]):
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Setting up Qdrant vector store at: {QDRANT_PATH}")
    client = QdrantClient(path=str(QDRANT_PATH))

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    print(f"Embedding {len(chunks)} chunks in batches of {BATCH_SIZE}...\n")
    points = []

    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        prefixed_texts = [f"Represent this document for retrieval: {t}" for t in texts]
        vectors = embedder.encode(prefixed_texts, normalize_embeddings=True)

        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            points.append(PointStruct(
                id      = i + j,
                vector  = vector.tolist(),
                payload = {          
                    "id":     chunk["id"],
                    "text":   chunk["text"],
                    "source": chunk["source"],
                    "page":   chunk["page"],
                }
            ))

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"\n Saved {len(points)} vectors to Qdrant")


def build_bm25_index(chunks: List[Dict]):
    print("\nBuilding BM25 keyword index...")
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25      = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f" BM25 index saved to {BM25_PATH}")


if __name__ == "__main__":
    chunks = load_documents()

    build_vector_index(chunks)

    build_bm25_index(chunks)

    print("\n Indexing complete!")

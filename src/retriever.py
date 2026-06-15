import pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  
COLLECTION   = "sebi_mf"

DENSE_TOP_K  = 20    
BM25_TOP_K   = 20    
FINAL_TOP_K  = 2     
RRF_K        = 60    

QDRANT_PATH  = Path(__file__).parent.parent / "data" / "qdrant_store"
BM25_PATH    = Path(__file__).parent.parent / "data" / "bm25_index.pkl"

class Retriever:
    def __init__(self):
        print("Loading retrieval models...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = CrossEncoder(RERANK_MODEL)
        self.client   = QdrantClient(path=str(QDRANT_PATH))

        with open(BM25_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25   = data["bm25"]
        self.chunks = data["chunks"]   
        print("Retriever ready.\n")

    def dense_search(self, query: str) -> List[Dict]:
        prefixed = f"Represent this query for retrieving relevant documents: {query}"
        query_vector = self.embedder.encode(prefixed, normalize_embeddings=True).tolist()

        results = self.client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=DENSE_TOP_K,
            with_payload=True,
        )

        return [
            {"chunk": hit.payload, "rank": i}
            for i, hit in enumerate(results)
        ]

    def bm25_search(self, query: str) -> List[Dict]:
        """Search using BM25 keyword matching. Returns ranked chunks."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices    = ranked_indices[:BM25_TOP_K]

        return [
            {
                "chunk": {
                    "id":     self.chunks[i]["id"],
                    "text":   self.chunks[i]["text"],
                    "source": self.chunks[i]["source"],
                    "page":   self.chunks[i]["page"],
                },
                "rank": rank,
            }
            for rank, i in enumerate(top_indices)
        ]

    def fuse_with_rrf(self, dense: List[Dict], sparse: List[Dict]) -> List[Dict]:
        rrf_scores = {}
        chunk_map  = {}

        for result in dense:
            cid = result["chunk"]["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + result["rank"])
            chunk_map[cid]  = result["chunk"]

        for result in sparse:
            cid = result["chunk"]["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + result["rank"])
            chunk_map[cid]  = result["chunk"]

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        return [{"chunk": chunk_map[cid], "rrf_score": rrf_scores[cid]} for cid in sorted_ids]

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        pairs  = [(query, c["chunk"]["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:FINAL_TOP_K]

    def retrieve(self, query: str) -> List[Dict]:
        dense      = self.dense_search(query)
        sparse     = self.bm25_search(query)
        fused      = self.fuse_with_rrf(dense, sparse)
        final      = self.rerank(query, fused)
        return final


if __name__ == "__main__":
    retriever = Retriever()
    query     = "What is the maximum TER for a large-cap mutual fund?"
    results   = retriever.retrieve(query)

    print(f"Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"Result {i} | {r['chunk']['source']} | Page {r['chunk']['page']}")
        print(f"  {r['chunk']['text'][:200]}...\n")

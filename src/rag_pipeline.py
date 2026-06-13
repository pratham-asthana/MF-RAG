"""
rag_pipeline.py — Step 4: Generate a grounded answer using the LLM

What this file does:
  1. Takes a user query
  2. Retrieves the top 5 most relevant chunks (via retriever.py)
  3. Builds a prompt that shows those chunks to the LLM
  4. Calls phi3:mini (running locally via Ollama) to generate an answer
  5. Returns the answer + which sources it came from

The LLM is told: "Only use these chunks. Cite [SOURCE N] inline."
This is what makes it a RAG system — the LLM cannot hallucinate because
it is constrained to only use what we retrieved.
"""

from typing import List, Dict
import ollama

from retriever import Retriever

# ── Settings ──────────────────────────────────────────────────────────────────
LLM_MODEL = "phi3:mini"   # Pull with: ollama pull phi3:mini
# ──────────────────────────────────────────────────────────────────────────────

# This is the instruction we give the LLM before every conversation
SYSTEM_PROMPT = """You are a precise regulatory assistant specializing in SEBI Mutual Fund regulations.

Your rules:
- Answer ONLY using the document excerpts provided to you.
- Always cite your source inline using [SOURCE 1], [SOURCE 2], etc.
- If the excerpts don't contain enough information, say: "The provided documents do not contain enough information to answer this."
- Never guess or use outside knowledge.
- Be concise and factual.
"""


def build_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Builds the full prompt sent to the LLM.

    Format:
      QUESTION: <user question>

      DOCUMENT EXCERPTS:
      [SOURCE 1]
      Document: xyz.pdf | Page: 3
      <chunk text>
      ---
      [SOURCE 2]
      ...

      ANSWER:
    """
    excerpts = []
    for i, result in enumerate(chunks, 1):
        chunk = result["chunk"]
        excerpt = (
            f"[SOURCE {i}]\n"
            f"Document: {chunk['source']} | Page: {chunk['page']}\n\n"
            f"{chunk['text']}"
        )
        excerpts.append(excerpt)

    context = "\n---\n".join(excerpts)

    return (
        f"Answer the following question using ONLY the document excerpts below.\n"
        f"Cite sources inline as [SOURCE 1], [SOURCE 2], etc.\n\n"
        f"QUESTION: {query}\n\n"
        f"DOCUMENT EXCERPTS:\n{context}\n\n"
        f"ANSWER:"
    )


def get_sources(chunks: List[Dict]) -> List[Dict]:
    """Extract clean source metadata from retrieved chunks."""
    sources = []
    for i, result in enumerate(chunks, 1):
        chunk = result["chunk"]
        sources.append({
            "label":   f"[SOURCE {i}]",
            "file":    chunk["source"],
            "page":    chunk["page"],
            "snippet": chunk["text"][:250] + "...",   # short preview
        })
    return sources


def answer(query: str, retriever: Retriever) -> Dict:
    """
    Main RAG function. Takes a query, returns answer + sources.

    Returns:
    {
        "answer":  "The maximum TER is... [SOURCE 1]",
        "sources": [{"label": "[SOURCE 1]", "file": "...", "page": 3, "snippet": "..."}, ...]
    }
    """
    # Step 1: Retrieve top 5 relevant chunks
    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    # Step 2: Build the prompt with those chunks
    prompt = build_prompt(query, retrieved_chunks)

    # Step 3: Send to phi3:mini via Ollama
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": 0.1},   # low = more factual, less creative
    )

    # Step 4: Extract answer text
    answer_text = response["message"]["content"].strip()

    # Step 5: Package sources for display
    sources = get_sources(retrieved_chunks)

    return {
        "answer":  answer_text,
        "sources": sources,
    }


# Test the full pipeline directly
if __name__ == "__main__":
    retriever = Retriever()

    questions = [
        "What is the maximum Total Expense Ratio for a large-cap mutual fund?",
        "How is the Risk-o-meter of a scheme determined?",
        "What are the key features of the MF Lite framework?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("="*60)

        result = answer(q, retriever)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  {s['label']}  {s['file']}  (Page {s['page']})")

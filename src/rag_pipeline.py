from typing import List, Dict
import ollama
from retriever import Retriever

LLM_MODEL = "phi3:mini" 
  
SYSTEM_PROMPT = """You are a SEBI Mutual Fund regulatory assistant.
 
Answer using ONLY the document excerpts given to you.
Be short. Be direct. No explanations unless asked.
 
Rules:
1. Give the exact fact or number from the excerpt. Nothing more.
2. Cite inline: [SOURCE 1], [SOURCE 2], etc.
3. If the answer is not in the excerpts, say: "Not found in the provided documents."
4. Do not repeat the question.
5. Do not explain what you are about to do.
6. Do not add background context or definitions.
 
Format:
- One to three sentences maximum.
- If the answer is a list, use bullet points. Keep each point one line.
- Always end with the citation.
"""


def build_prompt(query: str, chunks: List[Dict]) -> str:
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
    sources = []
    for i, result in enumerate(chunks, 1):
        chunk = result["chunk"]
        sources.append({
            "label":   f"[SOURCE {i}]",
            "file":    chunk["source"],
            "page":    chunk["page"],
            "snippet": chunk["text"][:250] + "...",   
        })
    return sources


def answer(query: str, retriever: Retriever) -> Dict:
    
    retrieved_chunks = retriever.retrieve(query)

    if not retrieved_chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    prompt = build_prompt(query, retrieved_chunks)

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={"temperature": 0.1},   
    )
    
    answer_text = response["message"]["content"].strip()

    sources = get_sources(retrieved_chunks)

    return {
        "answer":  answer_text,
        "sources": sources,
    }


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

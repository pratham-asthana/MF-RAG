"""
ingest.py — Step 1: Read documents and split into chunks

What this file does:
  1. Reads every PDF/DOCX file from the data/documents/ folder
  2. Extracts raw text from each page
  3. Splits the text into overlapping chunks (like cutting a book into cards)
  4. Attaches metadata to each chunk: which file, which page

A "chunk" is just a small piece of text (~500 words) with info about where it came from.
Overlap means consecutive chunks share ~100 words so context isn't lost at boundaries.
"""

import uuid
from pathlib import Path
from typing import List, Dict

import fitz          # PyMuPDF — reads PDFs
from docx import Document   # reads DOCX files
from tqdm import tqdm

# ── Settings ──────────────────────────────────────────────────────────────────
DOCS_FOLDER  = Path(__file__).parent.parent / "data" / "documents"
CHUNK_SIZE   = 500    # words per chunk
OVERLAP      = 100    # words shared between consecutive chunks
# ──────────────────────────────────────────────────────────────────────────────


def read_pdf(filepath: Path) -> List[Dict]:
    """Read a PDF and return a list of {text, page, source} dicts — one per page."""
    pages = []
    pdf = fitz.open(str(filepath))

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text().strip()
        if text:  # skip blank pages
            pages.append({
                "text":   text,
                "page":   page_num,
                "source": filepath.name,
            })

    pdf.close()
    return pages


def read_docx(filepath: Path) -> List[Dict]:
    """Read a DOCX and return its full text as a single page."""
    doc = Document(str(filepath))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": full_text, "page": 1, "source": filepath.name}]


def split_into_chunks(pages: List[Dict]) -> List[Dict]:
    """
    Merge all pages into one word stream, then slide a window across it.

    Example with CHUNK_SIZE=5, OVERLAP=2:
      Words:   [A B C D E F G H I J]
      Chunk 1: [A B C D E]
      Chunk 2: [D E F G H]   ← shares D,E with chunk 1
      Chunk 3: [G H I J]
    """
    # Flatten all words, keeping track of which page each word came from
    all_words = []
    for page in pages:
        for word in page["text"].split():
            all_words.append({
                "word":   word,
                "page":   page["page"],
                "source": page["source"],
            })

    chunks = []
    start  = 0

    while start < len(all_words):
        end        = min(start + CHUNK_SIZE, len(all_words))
        word_slice = all_words[start:end]

        chunk_text = " ".join(w["word"] for w in word_slice)
        first_page = word_slice[0]["page"]
        source     = word_slice[0]["source"]

        chunks.append({
            "id":     str(uuid.uuid4()),   # unique ID for this chunk
            "text":   chunk_text,
            "source": source,
            "page":   first_page,
        })

        start += CHUNK_SIZE - OVERLAP   # slide forward with overlap

    return chunks


def load_documents() -> List[Dict]:
    """
    Main function: reads all documents from data/documents/ and
    returns a flat list of chunks ready for embedding.
    """
    files = [f for f in DOCS_FOLDER.iterdir()
             if f.suffix.lower() in (".pdf", ".docx", ".txt")]

    if not files:
        raise FileNotFoundError(
            f"No documents found in {DOCS_FOLDER}\n"
            "Please place your SEBI PDFs there."
        )

    print(f"Found {len(files)} document(s). Reading and chunking...\n")
    all_chunks = []

    for filepath in tqdm(files):
        try:
            if filepath.suffix.lower() == ".pdf":
                pages = read_pdf(filepath)
            elif filepath.suffix.lower() == ".docx":
                pages = read_docx(filepath)
            else:
                text  = filepath.read_text(encoding="utf-8", errors="ignore")
                pages = [{"text": text, "page": 1, "source": filepath.name}]

            chunks = split_into_chunks(pages)
            all_chunks.extend(chunks)
            print(f"  ✓  {filepath.name}  →  {len(chunks)} chunks")

        except Exception as e:
            print(f"  ✗  {filepath.name}  →  Error: {e}")

    print(f"\nTotal chunks ready: {len(all_chunks)}")
    return all_chunks


# Run this file directly to test it
if __name__ == "__main__":
    chunks = load_documents()
    print(f"\nExample chunk:\n")
    print(f"  Source : {chunks[0]['source']}")
    print(f"  Page   : {chunks[0]['page']}")
    print(f"  Text   : {chunks[0]['text'][:200]}...")

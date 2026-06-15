import uuid
from pathlib import Path
from typing import List, Dict
import fitz
from docx import Document  
from tqdm import tqdm

DOCS_FOLDER  = Path(__file__).parent.parent / "data" / "documents"
CHUNK_SIZE   = 500    
OVERLAP      = 100    

def read_pdf(filepath: Path) -> List[Dict]:
    pages = []
    pdf = fitz.open(str(filepath))

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text().strip()
        if text:  
            pages.append({
                "text":   text,
                "page":   page_num,
                "source": filepath.name,
            })

    pdf.close()
    return pages


def read_docx(filepath: Path) -> List[Dict]:
    doc = Document(str(filepath))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": full_text, "page": 1, "source": filepath.name}]


def split_into_chunks(pages: List[Dict]) -> List[Dict]:
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
            "id":     str(uuid.uuid4()), 
            "text":   chunk_text,
            "source": source,
            "page":   first_page,
        })

        start += CHUNK_SIZE - OVERLAP   
    return chunks


def load_documents() -> List[Dict]:
    files = [f for f in DOCS_FOLDER.iterdir()
             if f.suffix.lower() in (".pdf", ".docx", ".txt")]

    if not files:
        raise FileNotFoundError(
            f"No documents found in {DOCS_FOLDER}"
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
            print(f"{filepath.name}  →  {len(chunks)} chunks")

        except Exception as e:
            print(f"{filepath.name}  →  Error: {e}")

    print(f"\nTotal chunks ready: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = load_documents()
    print(f"\nExample chunk:\n")
    print(f"  Source : {chunks[0]['source']}")
    print(f"  Page   : {chunks[0]['page']}")
    print(f"  Text   : {chunks[0]['text'][:200]}...")

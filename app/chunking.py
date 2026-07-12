"""
Pure-Python document loading and chunking. Deliberately has zero
dependencies on chromadb/sentence-transformers so it can be unit-tested
in any environment.
"""
import glob
import os
from typing import List, Tuple


def load_documents(docs_dir: str) -> List[Tuple[str, str]]:
    """Return a list of (doc_id, text) for every .txt file in docs_dir."""
    paths = sorted(glob.glob(os.path.join(docs_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {docs_dir}")
    docs = []
    for path in paths:
        doc_id = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            docs.append((doc_id, f.read()))
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Paragraph-aware chunker: split on blank lines first, then greedily pack
    paragraphs into ~chunk_size character windows with a character overlap
    carried into the next chunk so context isn't lost at boundaries.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # carry the tail of the previous chunk forward for overlap
                current = current[-overlap:] + "\n\n" + para
            else:
                # nothing pending — the paragraph itself is the overflow
                current = para

            # if `current` is still oversized, hard-split it on fixed windows
            while len(current) > chunk_size:
                chunks.append(current[:chunk_size])
                current = current[chunk_size - overlap :]
    if current:
        chunks.append(current)
    return chunks

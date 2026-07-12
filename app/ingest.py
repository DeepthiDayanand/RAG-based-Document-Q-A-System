"""
Ingestion pipeline: load .txt documents from a directory, split them into
overlapping chunks, embed each chunk, and upsert into a persisted Chroma
collection.

Usage:
    python -m app.ingest --docs_dir data/sample_docs
"""
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_text, load_documents
from app.config import settings


def build_index(docs_dir: str = None, reset: bool = True) -> int:
    """
    Ingest all documents in docs_dir into the Chroma collection.
    Returns the number of chunks indexed.
    """
    docs_dir = docs_dir or settings.docs_dir
    docs = load_documents(docs_dir)

    print(f"Loaded {len(docs)} documents from {docs_dir}")
    embedder = SentenceTransformer(settings.embedding_model)

    client = chromadb.PersistentClient(path=settings.chroma_db_dir)
    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )

    all_ids, all_texts, all_metadatas = [], [], []
    for doc_id, text in docs:
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_ids.append(f"{doc_id}::{i}")
            all_texts.append(chunk)
            all_metadatas.append({"doc_id": doc_id, "chunk_index": i})

    if not all_texts:
        raise ValueError("No chunks produced from documents — check input files.")

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = embedder.encode(all_texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_texts,
        metadatas=all_metadatas,
    )
    print(f"Indexed {len(all_texts)} chunks into '{settings.chroma_collection}'")
    return len(all_texts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default=None, help="Directory of .txt files")
    parser.add_argument(
        "--no_reset",
        action="store_true",
        help="Append to existing collection instead of rebuilding it",
    )
    args = parser.parse_args()
    build_index(docs_dir=args.docs_dir, reset=not args.no_reset)

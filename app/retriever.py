"""
Thin wrapper around a Chroma collection that exposes a single `retrieve`
method. Kept intentionally small so a different vector DB (Pinecone, FAISS)
can be swapped in by re-implementing this one class with the same interface.
"""
from dataclasses import dataclass
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    text: str
    score: float  # similarity score, higher = more relevant


class Retriever:
    def __init__(self):
        self._embedder = SentenceTransformer(settings.embedding_model)
        client = chromadb.PersistentClient(path=settings.chroma_db_dir)
        self._collection = client.get_or_create_collection(
            name=settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedChunk]:
        top_k = top_k or settings.top_k
        if self.count() == 0:
            raise RuntimeError(
                "Vector store is empty. Run `python -m app.ingest` first."
            )

        query_embedding = self._embedder.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.count()),
        )

        chunks: List[RetrievedChunk] = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]  # cosine distance, lower = closer

        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            similarity = 1 - distance  # convert distance -> similarity score
            chunks.append(
                RetrievedChunk(
                    doc_id=meta["doc_id"],
                    chunk_id=chunk_id,
                    text=text,
                    score=round(float(similarity), 4),
                )
            )
        return chunks

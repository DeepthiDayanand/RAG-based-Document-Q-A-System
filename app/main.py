"""
FastAPI backend for the RAG Document Q&A system.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.ingest import build_index
from app.rag_pipeline import RAGPipeline

_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RAGPipeline()
    yield


app = FastAPI(title="RAG Document Q&A API", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float


class IngestRequest(BaseModel):
    docs_dir: Optional[str] = None
    reset: bool = True


class IngestResponse(BaseModel):
    chunks_indexed: int


@app.get("/health")
def health():
    return {"status": "ok", "collection_size": _pipeline._retriever.count()}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        n = build_index(docs_dir=req.docs_dir, reset=req.reset)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    # rebuild the pipeline so the retriever picks up the fresh collection
    global _pipeline
    _pipeline = RAGPipeline()
    return IngestResponse(chunks_indexed=n)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        result = _pipeline.answer(req.question, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                doc_id=s.doc_id, chunk_id=s.chunk_id, score=s.score, text=s.text
            )
            for s in result.sources
        ],
        latency_ms=result.latency_ms,
        retrieval_latency_ms=result.retrieval_latency_ms,
        generation_latency_ms=result.generation_latency_ms,
    )

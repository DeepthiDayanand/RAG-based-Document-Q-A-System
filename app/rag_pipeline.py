"""
Wires the retriever and generator together into a single callable pipeline,
and tracks end-to-end latency (used both by the API and the eval harness).
"""
import time
from dataclasses import dataclass, field
from typing import List

from app.generator import generate_answer
from app.retriever import Retriever, RetrievedChunk


@dataclass
class RAGResult:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0


class RAGPipeline:
    def __init__(self):
        self._retriever = Retriever()

    def answer(self, question: str, top_k: int = None) -> RAGResult:
        start = time.perf_counter()

        t0 = time.perf_counter()
        chunks = self._retriever.retrieve(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        answer_text = generate_answer(question, chunks)
        generation_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - start) * 1000
        return RAGResult(
            answer=answer_text,
            sources=chunks,
            latency_ms=round(total_ms, 2),
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(generation_ms, 2),
        )

"""
Calls the Anthropic API to generate an answer grounded strictly in the
retrieved chunks. Swap this module out to use a different LLM provider —
the rest of the pipeline only depends on `generate_answer`'s signature.
"""
from typing import List

from anthropic import Anthropic

from app.config import settings
from app.retriever import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer the user's question using ONLY "
    "the provided context excerpts. If the context does not contain enough "
    "information to answer, say so plainly instead of guessing. Cite which "
    "excerpt(s) you used by their [n] marker. Keep answers concise."
)


def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {chunk.doc_id})\n{chunk.text}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return "I couldn't find any relevant context to answer that question."

    context_block = _build_context_block(chunks)
    user_message = (
        f"Context excerpts:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )

    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

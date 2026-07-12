"""
Lightweight tests for the chunker that don't require the embedding model,
Chroma, or an API key — safe to run in any environment.

Run with: python -m pytest tests/test_chunking.py
"""
from app.chunking import chunk_text


def test_short_text_single_chunk():
    text = "This is a short paragraph."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_multi_paragraph_packing():
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    # all three paragraphs fit comfortably under 500 chars -> single chunk
    assert len(chunks) == 1
    assert "Para one." in chunks[0]
    assert "Para three." in chunks[0]


def test_long_text_splits_into_multiple_chunks():
    paragraph = "A" * 300
    text = "\n\n".join([paragraph] * 5)  # 5 paragraphs, way over chunk_size
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        # allow slack for the overlap carried into the next chunk
        assert len(c) <= 500 + 60


def test_oversized_single_paragraph_hard_splits():
    paragraph = "B" * 1200
    chunks = chunk_text(paragraph, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    rejoined = "".join(chunks)
    assert "B" * 1200 in rejoined.replace("", "")  # content preserved (with overlap dupes)


if __name__ == "__main__":
    test_short_text_single_chunk()
    test_multi_paragraph_packing()
    test_long_text_splits_into_multiple_chunks()
    test_oversized_single_paragraph_hard_splits()
    print("All chunking tests passed.")

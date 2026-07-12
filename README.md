# RAG-Based Document Q&A System

A retrieval-augmented generation (RAG) pipeline that answers questions over a
document collection. Documents are chunked, embedded with a local sentence-transformer
model, indexed in a Chroma vector store, and retrieved chunks are passed to
Claude via a FastAPI backend to generate grounded answers.

Resume bullet, filled in:

> Built a retrieval-augmented generation pipeline over **12 documents** using
> **all-MiniLM-L6-v2** and **Chroma** vector DB, serving answers via
> **Claude Sonnet** with a FastAPI backend; achieved **recall@3 = 0.91,
> p50 latency = 480ms, answer accuracy = 87%** on a held-out eval set of
> **20 Q&A pairs**.

(Swap the bold numbers for whatever your own `eval/evaluate.py` run produces —
see "Running the evaluation" below.)

## Architecture

```
                    ┌─────────────┐
   documents/  ───▶ │   ingest.py │ ──▶ chunk ──▶ embed ──▶ Chroma (persisted)
                    └─────────────┘

   question   ───▶ ┌───────────────┐      ┌───────────┐      ┌───────────┐
   (HTTP POST) ───▶ │  FastAPI      │ ───▶ │ retriever │ ───▶ │ generator │ ─▶ answer
                    │  /query       │      │ (top-k)   │      │ (Claude)  │      + sources
                    └───────────────┘      └───────────┘      └───────────┘
```

- **Chunking**: paragraph-aware splitter, ~500 chars/chunk with 50-char overlap
  (`app/ingest.py`).
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  runs locally/CPU, no API cost).
- **Vector DB**: Chroma, persisted to disk (`./chroma_db`). Swappable — see
  "Swapping components" below.
- **LLM**: Anthropic Claude (`app/generator.py`), called only with the
  retrieved chunks as context (grounded generation, not open-book).
- **API**: FastAPI with `/ingest`, `/query`, `/health` endpoints
  (`app/main.py`).
- **Eval harness**: `eval/evaluate.py` computes recall@k, latency
  percentiles, and answer accuracy against a labeled held-out set.

## Project layout

```
rag-qa-system/
├── app/
│   ├── config.py         # all tunables in one place
│   ├── ingest.py          # load, chunk, embed, upsert into Chroma
│   ├── retriever.py       # top-k vector search
│   ├── generator.py       # prompt construction + Claude call
│   ├── rag_pipeline.py    # wires retriever + generator together
│   └── main.py            # FastAPI app
├── data/sample_docs/       # 5 sample .txt docs to try it out immediately
├── eval/
│   ├── eval_dataset.jsonl  # 10 labeled Q&A pairs for the sample docs
│   └── evaluate.py         # recall@k / latency / accuracy report
├── scripts/run_ingest.sh
├── tests/test_chunking.py  # dependency-free unit tests for the chunker
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd rag-qa-system
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Run it

1. **Ingest the sample documents** (or point `DOCS_DIR` at your own folder):

```bash
python -m app.ingest --docs_dir data/sample_docs
```

2. **Start the API**:

```bash
uvicorn app.main:app --reload --port 8000
```

3. **Ask a question**:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI used for?", "top_k": 3}'
```

Response shape:

```json
{
  "answer": "FastAPI is a Python web framework used for building APIs...",
  "sources": [
    {"doc_id": "fastapi.txt", "chunk_id": "fastapi.txt::0", "score": 0.87, "text": "..."}
  ],
  "latency_ms": 512.4
}
```

To index your own documents instead: drop `.txt` files into a folder and set
`DOCS_DIR` (env var or `--docs_dir` flag) to that folder before running
`app.ingest`.

## Running the evaluation

```bash
python -m eval.evaluate --k 3
```

This runs every question in `eval/eval_dataset.jsonl` through the live
pipeline and prints/saves (`eval/report.json`):

- **Recall@k** — fraction of questions where the gold source document
  appears in the top-k retrieved chunks.
- **Latency** — p50/p95 wall-clock time per query (retrieval + generation).
- **Answer accuracy** — fraction of answers containing the expected
  keyword(s), a cheap proxy for correctness (swap in an LLM-judge or exact
  match if you need something stricter).

Use `eval/eval_dataset.jsonl` as a template: add rows for your own corpus in
the same `{"question", "expected_doc_id", "expected_keywords"}` format and
rerun to get real numbers for your resume bullet.

## Swapping components

The system is intentionally modular so you can substitute pieces without
touching the rest:

| Component | Default | Swap in |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | any `sentence-transformers` model, or OpenAI/Voyage embeddings via API — change `EMBEDDING_MODEL` in `app/config.py` |
| Vector DB | Chroma (local, persisted) | Pinecone or FAISS — implement the same 3 methods (`add`, `query`, `count`) in `app/retriever.py` |
| LLM | Claude (Anthropic API) | any provider — replace the client call in `app/generator.py` |
| Chunker | paragraph-based, fixed size | swap in a semantic/recursive chunker in `app/ingest.py` |



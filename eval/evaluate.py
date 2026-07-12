"""
Evaluation harness for the RAG pipeline.

Computes, over eval/eval_dataset.jsonl:
  - Recall@k       : fraction of questions where the gold document appears
                     among the top-k retrieved chunks.
  - Latency        : p50 / p95 / mean end-to-end query latency (ms).
  - Answer accuracy: fraction of generated answers containing at least one
                     expected keyword (a cheap proxy — swap in an LLM judge
                     or exact-match grading for something stricter).

Usage:
    python -m eval.evaluate --k 3
    python -m eval.evaluate --k 3 --dataset eval/eval_dataset.jsonl --out eval/report.json
"""
import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from app.rag_pipeline import RAGPipeline


def load_dataset(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def keyword_hit(answer: str, keywords: List[str]) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def run_eval(dataset_path: str, k: int, out_path: str = None) -> Dict:
    dataset = load_dataset(dataset_path)
    pipeline = RAGPipeline()

    per_query = []
    for row in dataset:
        result = pipeline.answer(row["question"], top_k=k)
        retrieved_doc_ids = {s.doc_id for s in result.sources}
        hit_recall = row["expected_doc_id"] in retrieved_doc_ids
        hit_answer = keyword_hit(result.answer, row.get("expected_keywords", []))

        per_query.append(
            {
                "question": row["question"],
                "expected_doc_id": row["expected_doc_id"],
                "retrieved_doc_ids": list(retrieved_doc_ids),
                "recall_hit": hit_recall,
                "answer": result.answer,
                "answer_correct": hit_answer,
                "latency_ms": result.latency_ms,
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "generation_latency_ms": result.generation_latency_ms,
            }
        )

    n = len(per_query)
    latencies = [q["latency_ms"] for q in per_query]
    latencies_sorted = sorted(latencies)

    def percentile(data: List[float], p: float) -> float:
        if not data:
            return 0.0
        idx = min(int(len(data) * p), len(data) - 1)
        return data[idx]

    report = {
        "k": k,
        "num_queries": n,
        "recall_at_k": round(sum(q["recall_hit"] for q in per_query) / n, 4),
        "answer_accuracy": round(sum(q["answer_correct"] for q in per_query) / n, 4),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(percentile(latencies_sorted, 0.50), 2),
            "p95": round(percentile(latencies_sorted, 0.95), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "per_query": per_query,
    }

    print(f"\n=== Eval report (k={k}, n={n}) ===")
    print(f"Recall@{k}:        {report['recall_at_k']:.2%}")
    print(f"Answer accuracy:   {report['answer_accuracy']:.2%}")
    print(
        f"Latency (ms):      mean={report['latency_ms']['mean']}  "
        f"p50={report['latency_ms']['p50']}  p95={report['latency_ms']['p95']}"
    )

    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2))
        print(f"\nFull report written to {out_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--dataset", default="eval/eval_dataset.jsonl")
    parser.add_argument("--out", default="eval/report.json")
    args = parser.parse_args()
    run_eval(args.dataset, args.k, args.out)

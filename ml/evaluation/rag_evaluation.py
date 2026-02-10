"""
Evaluate RAG pipeline quality: hit rate, MRR, NDCG.

Usage:
    python rag_evaluation.py --test_file test_questions.json --api_url http://localhost:8000
"""

import argparse
import json
import math

import httpx


def hit_rate(results: list[dict], relevant_ids: list[str], k: int = 10) -> float:
    """Fraction of queries where at least one relevant doc is in top-k."""
    hits = 0
    for result in results[:k]:
        if str(result.get("document_id", "")) in relevant_ids:
            hits = 1
            break
    return hits


def mrr(results: list[dict], relevant_ids: list[str], k: int = 10) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, result in enumerate(results[:k]):
        if str(result.get("document_id", "")) in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(results: list[dict], relevant_ids: list[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for i, result in enumerate(results[:k]):
        if str(result.get("document_id", "")) in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)

    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_rag(test_file: str, api_url: str):
    """Run RAG evaluation on test questions."""
    with open(test_file) as f:
        test_cases = json.load(f)

    print(f"Evaluating {len(test_cases)} test cases...")

    hit_rates = []
    mrrs = []
    ndcgs = []

    client = httpx.Client(base_url=api_url, timeout=60)

    for i, test_case in enumerate(test_cases):
        query = test_case["question"]
        relevant_doc_ids = test_case.get("relevant_document_ids", [])

        try:
            response = client.post(
                "/api/v1/search",
                json={"query": query, "top_k": 10, "mode": "hybrid"},
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            hr = hit_rate(results, relevant_doc_ids)
            m = mrr(results, relevant_doc_ids)
            n = ndcg(results, relevant_doc_ids)

            hit_rates.append(hr)
            mrrs.append(m)
            ndcgs.append(n)

            print(f"  [{i+1}/{len(test_cases)}] {query[:50]}... HR={hr:.2f} MRR={m:.2f} NDCG={n:.2f}")

        except Exception as e:
            print(f"  [{i+1}/{len(test_cases)}] Error: {e}")
            hit_rates.append(0.0)
            mrrs.append(0.0)
            ndcgs.append(0.0)

    # Summary
    print("\n=== RAG Evaluation Results ===")
    print(f"Hit Rate@10: {sum(hit_rates)/len(hit_rates):.4f}")
    print(f"MRR@10:      {sum(mrrs)/len(mrrs):.4f}")
    print(f"NDCG@10:     {sum(ndcgs)/len(ndcgs):.4f}")
    print(f"Total queries: {len(test_cases)}")

    return {
        "hit_rate": sum(hit_rates) / len(hit_rates),
        "mrr": sum(mrrs) / len(mrrs),
        "ndcg": sum(ndcgs) / len(ndcgs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--test_file", type=str, default="test_questions.json")
    parser.add_argument("--api_url", type=str, default="http://localhost:8000")
    args = parser.parse_args()

    evaluate_rag(args.test_file, args.api_url)

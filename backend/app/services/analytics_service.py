"""
Usage analytics tracking for search and RAG operations.
Stores counters and query logs in Redis for fast writes.
"""

import json
import time
from datetime import datetime, timezone

from app.core.logging import get_logger
from app.core.redis import redis_client

logger = get_logger(__name__)

# Redis key prefixes
SEARCH_COUNT_KEY = "analytics:search:count"
SEARCH_LOG_KEY = "analytics:search:log"
RAG_COUNT_KEY = "analytics:rag:count"
RAG_LATENCY_KEY = "analytics:rag:latency"
DOC_ACCESS_KEY = "analytics:doc:access"


async def track_search(query: str, mode: str, result_count: int, latency_ms: float) -> None:
    """Track a search query for analytics."""
    try:
        pipe = redis_client.pipeline()
        pipe.incr(SEARCH_COUNT_KEY)
        pipe.lpush(
            SEARCH_LOG_KEY,
            json.dumps({
                "query": query[:200],
                "mode": mode,
                "result_count": result_count,
                "latency_ms": round(latency_ms, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
        )
        # Keep only the last 1000 search log entries
        pipe.ltrim(SEARCH_LOG_KEY, 0, 999)
        await pipe.execute()
    except Exception as e:
        logger.debug(f"Analytics track_search error: {e}")


async def track_rag_query(query: str, confidence: str, source_count: int, latency_ms: float) -> None:
    """Track a RAG Q&A query."""
    try:
        pipe = redis_client.pipeline()
        pipe.incr(RAG_COUNT_KEY)
        pipe.lpush(
            RAG_LATENCY_KEY,
            json.dumps({
                "query": query[:200],
                "confidence": confidence,
                "source_count": source_count,
                "latency_ms": round(latency_ms, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }),
        )
        pipe.ltrim(RAG_LATENCY_KEY, 0, 999)
        await pipe.execute()
    except Exception as e:
        logger.debug(f"Analytics track_rag_query error: {e}")


async def track_document_access(document_id: str) -> None:
    """Increment document access counter (for popularity-based ranking)."""
    try:
        await redis_client.zincrby(DOC_ACCESS_KEY, 1, document_id)
    except Exception as e:
        logger.debug(f"Analytics track_document_access error: {e}")


async def get_search_stats() -> dict:
    """Get aggregate search usage statistics."""
    try:
        total = await redis_client.get(SEARCH_COUNT_KEY)
        recent_raw = await redis_client.lrange(SEARCH_LOG_KEY, 0, 49)
        recent = [json.loads(r) for r in recent_raw]

        avg_latency = 0.0
        if recent:
            avg_latency = sum(r["latency_ms"] for r in recent) / len(recent)

        return {
            "total_searches": int(total or 0),
            "recent_count": len(recent),
            "avg_latency_ms": round(avg_latency, 1),
            "recent_queries": recent[:10],
        }
    except Exception as e:
        logger.debug(f"Analytics get_search_stats error: {e}")
        return {"total_searches": 0, "recent_count": 0, "avg_latency_ms": 0.0, "recent_queries": []}


async def get_rag_stats() -> dict:
    """Get aggregate RAG usage statistics."""
    try:
        total = await redis_client.get(RAG_COUNT_KEY)
        recent_raw = await redis_client.lrange(RAG_LATENCY_KEY, 0, 49)
        recent = [json.loads(r) for r in recent_raw]

        avg_latency = 0.0
        confidence_dist = {"high": 0, "medium": 0, "low": 0}
        if recent:
            avg_latency = sum(r["latency_ms"] for r in recent) / len(recent)
            for r in recent:
                confidence_dist[r.get("confidence", "low")] = confidence_dist.get(r.get("confidence", "low"), 0) + 1

        return {
            "total_queries": int(total or 0),
            "recent_count": len(recent),
            "avg_latency_ms": round(avg_latency, 1),
            "confidence_distribution": confidence_dist,
        }
    except Exception as e:
        logger.debug(f"Analytics get_rag_stats error: {e}")
        return {"total_queries": 0, "recent_count": 0, "avg_latency_ms": 0.0, "confidence_distribution": {}}


async def get_popular_documents(top_k: int = 10) -> list[dict]:
    """Get most accessed documents."""
    try:
        top = await redis_client.zrevrange(DOC_ACCESS_KEY, 0, top_k - 1, withscores=True)
        return [{"document_id": doc_id, "access_count": int(score)} for doc_id, score in top]
    except Exception as e:
        logger.debug(f"Analytics get_popular_documents error: {e}")
        return []

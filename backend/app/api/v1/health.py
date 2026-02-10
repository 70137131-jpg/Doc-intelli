from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.redis import get_redis

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "doc-intelli"}


@router.get("/health/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_async_session),
):
    checks = {}

    # Check PostgreSQL
    try:
        await db.execute(text("SELECT 1"))
        checks["postgres"] = "connected"
    except Exception as e:
        checks["postgres"] = f"error: {str(e)}"

    # Check Redis
    try:
        redis = await get_redis()
        await redis.ping()
        checks["redis"] = "connected"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"

    all_healthy = all(v == "connected" for v in checks.values())
    return {
        "status": "ready" if all_healthy else "degraded",
        "checks": checks,
    }


@router.get("/analytics")
async def get_analytics():
    """Get usage analytics for search and RAG."""
    from app.services.analytics_service import get_search_stats, get_rag_stats, get_popular_documents

    search_stats = await get_search_stats()
    rag_stats = await get_rag_stats()
    popular_docs = await get_popular_documents()

    return {
        "search": search_stats,
        "rag": rag_stats,
        "popular_documents": popular_docs,
    }

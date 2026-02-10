import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search_service import SearchService

router = APIRouter(tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    db: AsyncSession = Depends(get_async_session),
):
    start = time.time()
    search_service = SearchService(db)
    results, expanded_queries = await search_service.search(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        filters=request.filters,
        expand_query=request.expand_query,
        rerank=request.rerank,
    )
    latency_ms = (time.time() - start) * 1000

    # Track analytics
    try:
        from app.services.analytics_service import track_search

        await track_search(request.query, request.mode, len(results), latency_ms)
    except Exception:
        pass

    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
        search_mode=request.mode,
        expanded_queries=expanded_queries,
    )

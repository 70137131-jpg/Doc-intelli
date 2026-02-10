import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.schemas.search import SearchFilters, SearchResultItem
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.reranker_service import RerankerService

logger = get_logger(__name__)


class SearchService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedding_service = EmbeddingService(db)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        filters: Optional[SearchFilters] = None,
        expand_query: bool = False,
        rerank: bool = True,
    ) -> tuple[list[SearchResultItem], list[str] | None]:
        expanded_queries = None

        # Query expansion
        queries = [query]
        if expand_query:
            try:
                llm = LLMService()
                alternatives = llm.expand_query(query)
                queries.extend(alternatives)
                expanded_queries = alternatives
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}")

        # Collect candidates from all queries
        all_candidates = []

        for q in queries:
            if mode == "semantic":
                candidates = await self._semantic_search(q, top_k * 2, filters)
            elif mode == "keyword":
                candidates = await self._keyword_search(q, top_k * 2, filters)
            else:  # hybrid
                semantic_results = await self._semantic_search(q, top_k * 2, filters)
                keyword_results = await self._keyword_search(q, top_k * 2, filters)
                candidates = self._reciprocal_rank_fusion(
                    [semantic_results, keyword_results]
                )
            all_candidates.extend(candidates)

        # Deduplicate by chunk_id
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            chunk_id = c.get("chunk_id")
            if chunk_id:
                dedupe_key = f"chunk:{chunk_id}"
            else:
                dedupe_key = (
                    f"table:{c.get('document_id')}:{c.get('page_number')}:"
                    f"{hash(c.get('content', '')[:200])}"
                )

            if dedupe_key not in seen:
                seen.add(dedupe_key)
                unique_candidates.append(c)

        # Re-rank
        if rerank and unique_candidates:
            try:
                reranker = RerankerService()
                unique_candidates = reranker.rerank(
                    query, unique_candidates, top_k=top_k
                )
            except Exception as e:
                logger.warning(f"Re-ranking failed, using original order: {e}")
                unique_candidates = unique_candidates[:top_k]
        else:
            unique_candidates = unique_candidates[:top_k]

        # Convert to response objects
        results = [
            SearchResultItem(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                document_name=c.get("document_name", ""),
                content=c["content"],
                page_number=c.get("page_number"),
                chunk_index=c.get("chunk_index"),
                section_header=c.get("section_header"),
                score=c.get("score", 0.0),
                search_method=c.get("search_method", mode),
            )
            for c in unique_candidates
        ]

        return results, expanded_queries

    async def _semantic_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None,
    ) -> list[dict]:
        query_embedding = self.embedding_service.embed_text(query)

        document_ids = None
        document_type = None
        if filters:
            document_ids = filters.document_ids
            document_type = filters.document_type

        return await self.embedding_service.get_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            document_ids=document_ids,
            document_type=document_type,
        )

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None,
    ) -> list[dict]:
        # Build filter clauses
        filter_clauses = []
        params = {"query": query, "top_k": top_k}

        if filters:
            if filters.document_ids:
                filter_clauses.append("dc.document_id = ANY(:doc_ids)")
                params["doc_ids"] = [str(d) for d in filters.document_ids]
            if filters.document_type:
                filter_clauses.append("""
                    dc.document_id IN (
                        SELECT document_id FROM classifications WHERE document_type = :doc_type
                    )
                """)
                params["doc_type"] = filters.document_type
            if filters.date_from:
                filter_clauses.append("d.created_at >= :date_from")
                params["date_from"] = filters.date_from
            if filters.date_to:
                filter_clauses.append("d.created_at <= :date_to")
                params["date_to"] = filters.date_to

        where_clause = ""
        if filter_clauses:
            where_clause = "AND " + " AND ".join(filter_clauses)

        sql = text(f"""
            SELECT
                dc.id as chunk_id,
                dc.document_id,
                dc.content,
                dc.page_number,
                dc.section_header,
                dc.chunk_index,
                d.original_filename as document_name,
                ts_rank(dc.search_vector, plainto_tsquery('english', :query)) as rank
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.search_vector @@ plainto_tsquery('english', :query)
            {where_clause}
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        result = await self.db.execute(sql, params)
        rows = result.fetchall()

        return [
            {
                "chunk_id": row.chunk_id,
                "document_id": row.document_id,
                "content": row.content,
                "page_number": row.page_number,
                "section_header": row.section_header,
                "chunk_index": row.chunk_index,
                "document_name": row.document_name,
                "score": float(row.rank),
                "search_method": "keyword",
            }
            for row in rows
        ]

    async def parent_child_retrieval(
        self,
        child_results: list[dict],
        context_window: int = 1,
    ) -> list[dict]:
        """Expand retrieved chunks with neighboring context (parent-child retrieval)."""
        if not child_results:
            return child_results

        enriched = []
        for item in child_results:
            doc_id = item["document_id"]
            chunk_idx = item.get("chunk_index", 0)

            sql = text("""
                SELECT content, chunk_index
                FROM document_chunks
                WHERE document_id = :doc_id
                  AND chunk_index BETWEEN :start_idx AND :end_idx
                ORDER BY chunk_index
            """)
            result = await self.db.execute(sql, {
                "doc_id": str(doc_id),
                "start_idx": max(0, chunk_idx - context_window),
                "end_idx": chunk_idx + context_window,
            })
            neighbors = result.fetchall()

            if neighbors and len(neighbors) > 1:
                expanded_content = "\n".join(row.content for row in neighbors)
                enriched_item = item.copy()
                enriched_item["content"] = expanded_content
                enriched_item["parent_child"] = True
                enriched.append(enriched_item)
            else:
                enriched.append(item)

        return enriched

    async def table_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Search within extracted table data stored in document_pages.tables_json."""
        sql = text("""
            SELECT dp.document_id, dp.page_number, dp.tables_json,
                   d.original_filename as document_name
            FROM document_pages dp
            JOIN documents d ON d.id = dp.document_id
            WHERE dp.tables_json IS NOT NULL
              AND dp.tables_json::text ILIKE :pattern
            LIMIT :top_k
        """)
        result = await self.db.execute(sql, {"pattern": f"%{query}%", "top_k": top_k})
        rows = result.fetchall()

        results = []
        for row in rows:
            tables_json = row.tables_json
            if isinstance(tables_json, dict):
                tables = tables_json.get("tables") or []
            elif isinstance(tables_json, list):
                tables = tables_json
            else:
                tables = []
            for table in tables:
                table_text = "\n".join(
                    " | ".join(str(cell or "") for cell in r)
                    for r in (table.get("rows") or [])
                )
                if query.lower() in table_text.lower():
                    results.append({
                        "chunk_id": None,
                        "document_id": row.document_id,
                        "content": table_text[:1000],
                        "page_number": row.page_number,
                        "chunk_index": None,
                        "document_name": row.document_name,
                        "score": 0.5,
                        "search_method": "table",
                    })
        return results[:top_k]

    @staticmethod
    def contextual_compression(content: str, query: str, max_sentences: int = 5) -> str:
        """Extract only the most relevant sentences from a chunk for the given query."""
        import re
        sentences = re.split(r"(?<=[.!?])\s+", content)
        if len(sentences) <= max_sentences:
            return content

        query_terms = set(query.lower().split())
        scored = []
        for sent in sentences:
            words = set(sent.lower().split())
            overlap = len(query_terms & words)
            scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:max_sentences]]
        # Preserve original order
        ordered = [s for s in sentences if s in top]
        return " ".join(ordered)

    async def apply_temporal_boost(
        self,
        results: list[dict],
        decay_days: float = 90.0,
        boost_weight: float = 0.15,
    ) -> list[dict]:
        """Boost scores of recently created/updated documents using exponential decay.

        Documents created within `decay_days` get a score boost proportional
        to recency.  Older documents are unaffected.
        """
        if not results:
            return results

        doc_ids = list({str(r["document_id"]) for r in results})
        sql = text("""
            SELECT id, created_at FROM documents WHERE id = ANY(:ids)
        """)
        rows = await self.db.execute(sql, {"ids": doc_ids})
        date_map = {str(row.id): row.created_at for row in rows.fetchall()}

        now = datetime.now(timezone.utc)
        for r in results:
            created = date_map.get(str(r["document_id"]))
            if created:
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).total_seconds() / 86400
                recency_factor = math.exp(-age_days / decay_days)
                r["score"] = r.get("score", 0.0) + boost_weight * recency_factor

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[dict]], k: int = 60
    ) -> list[dict]:
        """Merge multiple ranked lists using Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        items = {}

        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list):
                chunk_id = str(item["chunk_id"])
                scores[chunk_id] += 1.0 / (k + rank + 1)
                if chunk_id not in items:
                    items[chunk_id] = item.copy()
                    items[chunk_id]["search_method"] = "hybrid"

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids:
            item = items[chunk_id]
            item["score"] = scores[chunk_id]
            results.append(item)

        return results

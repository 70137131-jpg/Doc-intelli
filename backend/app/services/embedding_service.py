import uuid
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import DocumentChunk
from app.models.embedding import ChunkEmbedding
from app.core.logging import get_logger
from app.config import settings

logger = get_logger(__name__)

# Lazy-loaded model singleton
_embedding_model = None


def _get_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(settings.embedding_model_name)
        logger.info(f"Loaded embedding model: {settings.embedding_model_name}")
    return _embedding_model


class EmbeddingService:
    def __init__(self, db: AsyncSession):
        self.db = db

    def embed_text(self, text: str) -> list[float]:
        model = _get_model()
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = _get_model()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=64)
        return [e.tolist() for e in embeddings]

    async def get_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        document_ids: Optional[list[uuid.UUID]] = None,
        document_type: Optional[str] = None,
    ) -> list[dict]:
        """Find similar chunks using pgvector cosine distance."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Build the query with optional filters
        filters = []
        params = {"embedding": embedding_str, "top_k": top_k}

        if document_ids:
            filters.append("ce.document_id = ANY(:doc_ids)")
            params["doc_ids"] = [str(d) for d in document_ids]

        if document_type:
            filters.append("""
                ce.document_id IN (
                    SELECT document_id FROM classifications WHERE document_type = :doc_type
                )
            """)
            params["doc_type"] = document_type

        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters)

        query = text(f"""
            SELECT
                ce.chunk_id,
                ce.document_id,
                dc.content,
                dc.page_number,
                dc.section_header,
                dc.chunk_index,
                d.original_filename as document_name,
                1 - (ce.embedding <=> :embedding::vector) as similarity
            FROM chunk_embeddings ce
            JOIN document_chunks dc ON dc.id = ce.chunk_id
            JOIN documents d ON d.id = ce.document_id
            {where_clause}
            ORDER BY ce.embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        result = await self.db.execute(query, params)
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
                "score": float(row.similarity),
                "search_method": "semantic",
            }
            for row in rows
        ]

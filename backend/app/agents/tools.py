"""
Agent tools that can be called by LangGraph agents.
Each tool wraps an existing service and returns structured results.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Optional

from app.core.database import SyncSessionLocal
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk
from app.models.classification import Classification, ExtractedField

logger = get_logger(__name__)


@dataclass
class ToolResult:
    success: bool
    data: dict | list | str
    error: str | None = None


def search_documents(query: str, top_k: int = 5, document_type: str | None = None) -> ToolResult:
    """Search across all documents using hybrid search."""
    try:
        session = SyncSessionLocal()
        from sqlalchemy import text

        # Keyword search (sync version)
        params = {"query": query, "top_k": top_k}
        type_filter = ""
        if document_type:
            type_filter = """
                AND dc.document_id IN (
                    SELECT document_id FROM classifications WHERE document_type = :doc_type
                )
            """
            params["doc_type"] = document_type

        sql = text(f"""
            SELECT
                dc.id as chunk_id,
                dc.document_id,
                dc.content,
                dc.page_number,
                d.original_filename as document_name,
                ts_rank(dc.search_vector, plainto_tsquery('english', :query)) as rank
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE dc.search_vector @@ plainto_tsquery('english', :query)
            {type_filter}
            ORDER BY rank DESC
            LIMIT :top_k
        """)

        result = session.execute(sql, params)
        rows = result.fetchall()
        session.close()

        results = [
            {
                "document_name": row.document_name,
                "document_id": str(row.document_id),
                "page_number": row.page_number,
                "content": row.content[:500],
                "relevance": float(row.rank),
            }
            for row in rows
        ]

        return ToolResult(success=True, data=results)

    except Exception as e:
        logger.error(f"search_documents failed: {e}")
        return ToolResult(success=False, data=[], error=str(e))


def get_document_fields(document_id: str) -> ToolResult:
    """Get extracted fields for a specific document."""
    try:
        session = SyncSessionLocal()
        doc_uuid = uuid.UUID(document_id)

        fields = (
            session.query(ExtractedField)
            .filter(ExtractedField.document_id == doc_uuid)
            .all()
        )

        classification = (
            session.query(Classification)
            .filter(Classification.document_id == doc_uuid)
            .first()
        )

        session.close()

        return ToolResult(
            success=True,
            data={
                "document_type": classification.document_type if classification else "Unknown",
                "fields": {f.field_name: f.field_value for f in fields},
            },
        )
    except Exception as e:
        return ToolResult(success=False, data={}, error=str(e))


def get_document_text(document_id: str, max_chunks: int = 10) -> ToolResult:
    """Get the full text content of a document."""
    try:
        session = SyncSessionLocal()
        doc_uuid = uuid.UUID(document_id)

        doc = session.query(Document).filter(Document.id == doc_uuid).first()
        if not doc:
            session.close()
            return ToolResult(success=False, data="", error="Document not found")

        chunks = (
            session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_uuid)
            .order_by(DocumentChunk.chunk_index)
            .limit(max_chunks)
            .all()
        )
        session.close()

        text = "\n\n".join(c.content for c in chunks)
        return ToolResult(
            success=True,
            data={
                "document_name": doc.original_filename,
                "document_id": document_id,
                "text": text,
                "chunk_count": len(chunks),
            },
        )
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))


def compare_documents(doc_id_1: str, doc_id_2: str) -> ToolResult:
    """Get text and fields from two documents for comparison."""
    result1 = get_document_text(doc_id_1)
    result2 = get_document_text(doc_id_2)
    fields1 = get_document_fields(doc_id_1)
    fields2 = get_document_fields(doc_id_2)

    if not result1.success or not result2.success:
        return ToolResult(
            success=False,
            data={},
            error=f"Failed to load documents: {result1.error or result2.error}",
        )

    return ToolResult(
        success=True,
        data={
            "document_1": {
                "name": result1.data["document_name"],
                "text": result1.data["text"],
                "fields": fields1.data.get("fields", {}) if fields1.success else {},
                "type": fields1.data.get("document_type", "Unknown") if fields1.success else "Unknown",
            },
            "document_2": {
                "name": result2.data["document_name"],
                "text": result2.data["text"],
                "fields": fields2.data.get("fields", {}) if fields2.success else {},
                "type": fields2.data.get("document_type", "Unknown") if fields2.success else "Unknown",
            },
        },
    )


def list_documents_by_type(document_type: str, limit: int = 20) -> ToolResult:
    """List documents of a specific type."""
    try:
        session = SyncSessionLocal()
        results = (
            session.query(Document, Classification)
            .join(Classification, Classification.document_id == Document.id)
            .filter(Classification.document_type == document_type)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .all()
        )
        session.close()

        return ToolResult(
            success=True,
            data=[
                {
                    "document_id": str(doc.id),
                    "filename": doc.original_filename,
                    "type": cls.document_type,
                    "confidence": cls.confidence,
                    "created_at": doc.created_at.isoformat(),
                }
                for doc, cls in results
            ],
        )
    except Exception as e:
        return ToolResult(success=False, data=[], error=str(e))


def generate_with_llm(prompt: str) -> ToolResult:
    """Call the LLM to generate text (summarization, analysis, etc.)."""
    try:
        from app.services.llm_service import LLMService
        llm = LLMService()
        response = llm.generate(prompt)
        return ToolResult(success=True, data=response)
    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))


# Tool registry for agents
TOOL_REGISTRY = {
    "search_documents": {
        "fn": search_documents,
        "description": "Search across all documents using a text query. Returns relevant chunks with document names and page numbers.",
        "parameters": {"query": "str", "top_k": "int (default 5)", "document_type": "str or None"},
    },
    "get_document_fields": {
        "fn": get_document_fields,
        "description": "Get extracted key-value fields for a specific document by its ID.",
        "parameters": {"document_id": "str (UUID)"},
    },
    "get_document_text": {
        "fn": get_document_text,
        "description": "Get the full text content of a document by its ID.",
        "parameters": {"document_id": "str (UUID)", "max_chunks": "int (default 10)"},
    },
    "compare_documents": {
        "fn": compare_documents,
        "description": "Load text and fields from two documents for side-by-side comparison.",
        "parameters": {"doc_id_1": "str (UUID)", "doc_id_2": "str (UUID)"},
    },
    "list_documents_by_type": {
        "fn": list_documents_by_type,
        "description": "List all documents of a specific type (Invoice, Contract, Report, Resume, Letter).",
        "parameters": {"document_type": "str", "limit": "int (default 20)"},
    },
    "generate_with_llm": {
        "fn": generate_with_llm,
        "description": "Generate text using the LLM. Use for summarization, analysis, report writing, etc.",
        "parameters": {"prompt": "str"},
    },
}

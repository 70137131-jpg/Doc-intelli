import uuid
from typing import Optional

from sqlalchemy import func, select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk, DocumentPage
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_document(
        self,
        filename: str,
        original_filename: str,
        mime_type: str,
        file_size_bytes: int,
        storage_key: str,
    ) -> Document:
        document = Document(
            filename=filename,
            original_filename=original_filename,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            storage_key=storage_key,
            status="pending",
        )
        self.db.add(document)
        await self.db.flush()
        return document

    async def get_document(self, document_id: uuid.UUID) -> Optional[Document]:
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def list_documents(
        self,
        page: int = 1,
        size: int = 20,
        status: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> tuple[list[Document], int]:
        query = select(Document)
        count_query = select(func.count(Document.id))

        if status:
            query = query.where(Document.status == status)
            count_query = count_query.where(Document.status == status)
        if mime_type:
            query = query.where(Document.mime_type == mime_type)
            count_query = count_query.where(Document.mime_type == mime_type)

        query = query.order_by(Document.created_at.desc())
        query = query.offset((page - 1) * size).limit(size)

        result = await self.db.execute(query)
        documents = list(result.scalars().all())

        count_result = await self.db.execute(count_query)
        total = count_result.scalar() or 0

        return documents, total

    async def update_status(
        self,
        document_id: uuid.UUID,
        status: str,
        error_message: Optional[str] = None,
        page_count: Optional[int] = None,
        total_chunks: Optional[int] = None,
    ) -> None:
        document = await self.get_document(document_id)
        if document:
            document.status = status
            if error_message is not None:
                document.error_message = error_message
            if page_count is not None:
                document.page_count = page_count
            if total_chunks is not None:
                document.total_chunks = total_chunks
            await self.db.flush()

    async def add_page(
        self,
        document_id: uuid.UUID,
        page_number: int,
        raw_text: str,
        extraction_method: str,
        confidence: Optional[float] = None,
        tables_json: Optional[dict] = None,
    ) -> DocumentPage:
        page = DocumentPage(
            document_id=document_id,
            page_number=page_number,
            raw_text=raw_text,
            extraction_method=extraction_method,
            confidence=confidence,
            tables_json=tables_json,
        )
        self.db.add(page)
        await self.db.flush()
        return page

    async def add_chunk(
        self,
        document_id: uuid.UUID,
        chunk_index: int,
        content: str,
        token_count: int,
        page_number: Optional[int] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        section_header: Optional[str] = None,
        chunking_method: str = "fixed_size",
    ) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            token_count=token_count,
            page_number=page_number,
            char_start=char_start,
            char_end=char_end,
            section_header=section_header,
            chunking_method=chunking_method,
        )
        self.db.add(chunk)
        await self.db.flush()
        return chunk

    async def get_chunks(
        self,
        document_id: uuid.UUID,
        page: int = 1,
        size: int = 50,
    ) -> list[DocumentChunk]:
        result = await self.db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .offset((page - 1) * size)
            .limit(size)
        )
        return list(result.scalars().all())

    async def get_all_chunks(self, document_id: uuid.UUID) -> list[DocumentChunk]:
        result = await self.db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        )
        return list(result.scalars().all())

    async def get_document_text(self, document_id: uuid.UUID, max_tokens: int = 2000) -> str:
        """Get concatenated text from document chunks up to max_tokens."""
        chunks = await self.get_all_chunks(document_id)
        text_parts = []
        total_tokens = 0
        for chunk in chunks:
            if total_tokens + chunk.token_count > max_tokens:
                break
            text_parts.append(chunk.content)
            total_tokens += chunk.token_count
        return "\n".join(text_parts)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self.db.execute(
            delete(Document).where(Document.id == document_id)
        )
        await self.db.flush()

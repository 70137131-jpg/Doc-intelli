import uuid

from sqlalchemy import text

from app.config import settings
from app.core.database import SyncSessionLocal
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk, DocumentPage
from app.services.chunking.fixed_size import FixedSizeChunker
from app.services.extraction.factory import ExtractorFactory
from app.services.storage_service import StorageService
from app.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def process_document(self, document_id: str):
    """Main document processing pipeline: download → extract → chunk → store."""
    doc_uuid = uuid.UUID(document_id)
    session = SyncSessionLocal()

    try:
        # Get document
        document = session.query(Document).filter(Document.id == doc_uuid).first()
        if not document:
            logger.error(f"Document {document_id} not found")
            return

        # Update status to processing
        document.status = "processing"
        session.commit()

        # Step 1: Download from MinIO
        logger.info(f"Downloading document {document_id} from storage")
        storage = StorageService()
        file_data = storage.download_file(settings.minio_bucket_raw, document.storage_key)

        # Step 2: Extract text
        logger.info(f"Extracting text from {document.original_filename}")
        extractor = ExtractorFactory.get_extractor(document.mime_type)
        extraction_result = extractor.extract(file_data, document.original_filename)

        # Store pages
        for page_result in extraction_result.pages:
            page = DocumentPage(
                document_id=doc_uuid,
                page_number=page_result.page_number,
                raw_text=page_result.text,
                extraction_method=page_result.extraction_method,
                confidence=page_result.confidence,
                tables_json=page_result.tables if page_result.tables else None,
            )
            session.add(page)

        document.page_count = extraction_result.total_pages
        session.flush()

        # Step 3: Chunk text
        logger.info(f"Chunking document {document_id}")
        chunker = FixedSizeChunker()
        all_chunks = []

        for page_result in extraction_result.pages:
            if not page_result.text or not page_result.text.strip():
                continue
            page_chunks = chunker.chunk(page_result.text, page_number=page_result.page_number)
            all_chunks.extend(page_chunks)

        # Re-index chunks globally
        for i, chunk_result in enumerate(all_chunks):
            chunk_result.chunk_index = i

        # Store chunks
        for chunk_result in all_chunks:
            chunk = DocumentChunk(
                document_id=doc_uuid,
                chunk_index=chunk_result.chunk_index,
                content=chunk_result.content,
                token_count=chunk_result.token_count,
                page_number=chunk_result.page_number,
                char_start=chunk_result.char_start,
                char_end=chunk_result.char_end,
                section_header=chunk_result.section_header,
                chunking_method="fixed_size",
            )
            session.add(chunk)

        document.total_chunks = len(all_chunks)
        session.flush()

        # Update tsvector for full-text search
        session.execute(
            text("""
                UPDATE document_chunks
                SET search_vector = to_tsvector('english', content)
                WHERE document_id = :doc_id AND search_vector IS NULL
            """),
            {"doc_id": str(doc_uuid)},
        )

        # Mark as completed
        document.status = "completed"
        session.commit()

        logger.info(
            f"Document {document_id} processed: "
            f"{extraction_result.total_pages} pages, {len(all_chunks)} chunks"
        )

        # Trigger follow-up tasks
        from app.tasks.classification_tasks import classify_document_task
        from app.tasks.embedding_tasks import generate_embeddings_task

        classify_document_task.delay(document_id)
        generate_embeddings_task.delay(document_id)

    except Exception as exc:
        session.rollback()
        logger.error(f"Document processing failed for {document_id}: {exc}", exc_info=True)

        # Update status to failed
        try:
            document = session.query(Document).filter(Document.id == doc_uuid).first()
            if document:
                document.status = "failed"
                document.error_message = str(exc)[:500]
                session.commit()
        except Exception:
            pass

        raise self.retry(exc=exc)
    finally:
        session.close()

import uuid

from app.tasks.celery_app import celery_app
from app.core.database import SyncSessionLocal
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk
from app.models.classification import Classification, ExtractedField
from app.config import settings

logger = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def classify_document_task(self, document_id: str):
    """Classify a document and extract fields."""
    doc_uuid = uuid.UUID(document_id)
    session = SyncSessionLocal()

    try:
        document = session.query(Document).filter(Document.id == doc_uuid).first()
        if not document:
            logger.error(f"Document {document_id} not found")
            return

        # Get document text (first ~2000 tokens worth)
        chunks = (
            session.query(DocumentChunk)
            .filter(DocumentChunk.document_id == doc_uuid)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

        text_parts = []
        total_tokens = 0
        for chunk in chunks:
            if total_tokens + chunk.token_count > 2000:
                break
            text_parts.append(chunk.content)
            total_tokens += chunk.token_count

        document_text = "\n".join(text_parts)

        if not document_text.strip():
            logger.warning(f"No text found for document {document_id}")
            return

        # Zero-shot classification with Gemini
        from app.services.llm_service import LLMService

        llm = LLMService()
        classification_result = llm.classify_document(document_text)

        if not classification_result:
            logger.error(f"Classification failed for document {document_id}")
            return

        # Store classification
        existing = (
            session.query(Classification)
            .filter(Classification.document_id == doc_uuid)
            .first()
        )

        if existing:
            existing.document_type = classification_result["category"]
            existing.confidence = classification_result["confidence"]
            existing.method = "gemini_zero_shot"
            existing.reasoning = classification_result.get("reasoning")
            existing.raw_scores = classification_result.get("scores")
        else:
            cls = Classification(
                document_id=doc_uuid,
                document_type=classification_result["category"],
                confidence=classification_result["confidence"],
                method="gemini_zero_shot",
                reasoning=classification_result.get("reasoning"),
                raw_scores=classification_result.get("scores"),
            )
            session.add(cls)

        session.flush()

        # Extract fields based on classification
        doc_type = classification_result["category"]
        field_results = llm.extract_fields(document_text, doc_type)

        if field_results:
            for field_name, field_value in field_results.items():
                existing_field = (
                    session.query(ExtractedField)
                    .filter(
                        ExtractedField.document_id == doc_uuid,
                        ExtractedField.field_name == field_name,
                    )
                    .first()
                )

                if existing_field:
                    existing_field.field_value = str(field_value) if field_value else None
                    existing_field.extraction_method = "gemini_llm"
                else:
                    ef = ExtractedField(
                        document_id=doc_uuid,
                        field_name=field_name,
                        field_value=str(field_value) if field_value else None,
                        field_type=type(field_value).__name__ if field_value else "string",
                        confidence=classification_result["confidence"],
                        extraction_method="gemini_llm",
                    )
                    session.add(ef)

        session.commit()
        logger.info(f"Document {document_id} classified as {doc_type}")

    except Exception as exc:
        session.rollback()
        logger.error(f"Classification failed for {document_id}: {exc}", exc_info=True)
        raise self.retry(exc=exc)
    finally:
        session.close()

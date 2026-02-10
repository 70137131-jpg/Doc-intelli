import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.classification import Classification
from app.models.document import Document
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.core.logging import get_logger

logger = get_logger(__name__)


class ClassificationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_classification(self, document_id: uuid.UUID) -> Classification | None:
        result = await self.db.execute(
            select(Classification).where(Classification.document_id == document_id)
        )
        return result.scalar_one_or_none()

    # Confidence threshold: below this, escalate from lightweight to LLM
    CONFIDENCE_THRESHOLD = 0.7

    async def classify_document(self, document: Document) -> Classification:
        """Multi-model classification pipeline with fallback.

        1. Try lightweight local classifier first (if available)
        2. If confidence < threshold, escalate to Gemini LLM
        3. Store result with method tracking
        """
        doc_service = DocumentService(self.db)
        text = await doc_service.get_document_text(document.id)

        if not text.strip():
            raise ValueError("No text available for classification")

        result = None
        method = "gemini_zero_shot"

        # Step 1: Try lightweight classifier (DeBERTa/local model)
        try:
            from app.services.lightweight_classifier import classify_with_lightweight_model
            lightweight_result = classify_with_lightweight_model(text)
            if lightweight_result and lightweight_result.get("confidence", 0) >= self.CONFIDENCE_THRESHOLD:
                result = lightweight_result
                method = "lightweight_classifier"
                logger.info(
                    f"Lightweight classifier: {result['category']} ({result['confidence']:.2f})"
                )
        except Exception as e:
            logger.debug(f"Lightweight classifier unavailable, using LLM: {e}")

        # Step 2: Fallback to Gemini for low-confidence or missing results
        if result is None:
            llm = LLMService()
            result = llm.classify_document(text)
            method = "gemini_zero_shot"

        if not result:
            raise ValueError("Classification returned no results")

        # Upsert classification
        existing = await self.get_classification(document.id)
        if existing:
            existing.document_type = result["category"]
            existing.confidence = result["confidence"]
            existing.method = method
            existing.reasoning = result.get("reasoning")
            existing.raw_scores = result.get("scores")
            await self.db.flush()
            return existing

        classification = Classification(
            document_id=document.id,
            document_type=result["category"],
            confidence=result["confidence"],
            method=method,
            reasoning=result.get("reasoning"),
            raw_scores=result.get("scores"),
        )
        self.db.add(classification)
        await self.db.flush()
        return classification

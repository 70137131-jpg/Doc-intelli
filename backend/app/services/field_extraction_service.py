import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.classification import ExtractedField
from app.core.logging import get_logger

logger = get_logger(__name__)

# Regex patterns for rule-based extraction
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "date": re.compile(
        r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
        re.IGNORECASE,
    ),
    "amount": re.compile(r"\$[\d,]+\.?\d*|\d+\.?\d*\s*(?:USD|EUR|GBP)"),
    "url": re.compile(r"https?://[^\s<>\"']+"),
}


class FieldExtractionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_fields(self, document_id: uuid.UUID) -> list[ExtractedField]:
        result = await self.db.execute(
            select(ExtractedField)
            .where(ExtractedField.document_id == document_id)
            .order_by(ExtractedField.field_name)
        )
        return list(result.scalars().all())

    async def apply_corrections(
        self, document_id: uuid.UUID, corrections: dict[str, str]
    ) -> None:
        for field_name, corrected_value in corrections.items():
            result = await self.db.execute(
                select(ExtractedField).where(
                    ExtractedField.document_id == document_id,
                    ExtractedField.field_name == field_name,
                )
            )
            field = result.scalar_one_or_none()

            if field:
                field.original_value = field.field_value
                field.field_value = corrected_value
                field.is_corrected = True
                field.extraction_method = "manual"
            else:
                new_field = ExtractedField(
                    document_id=document_id,
                    field_name=field_name,
                    field_value=corrected_value,
                    field_type="string",
                    confidence=1.0,
                    extraction_method="manual",
                    is_corrected=True,
                )
                self.db.add(new_field)

        await self.db.flush()

    async def export_corrections_for_retraining(self) -> list[dict]:
        """Export all human-corrected fields as training data for model improvement.

        Returns list of correction records for the retraining pipeline.
        """
        result = await self.db.execute(
            select(ExtractedField)
            .where(ExtractedField.is_corrected == True)
            .order_by(ExtractedField.field_name)
        )
        corrections = result.scalars().all()

        return [
            {
                "document_id": str(f.document_id),
                "field_name": f.field_name,
                "original_value": f.original_value,
                "corrected_value": f.field_value,
                "extraction_method": f.extraction_method,
            }
            for f in corrections
        ]

    @staticmethod
    def extract_with_regex(text: str) -> dict[str, list[str]]:
        """Extract common fields using regex patterns."""
        results = {}
        for field_name, pattern in PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                results[field_name] = list(set(matches))
        return results

    @staticmethod
    def extract_with_ner(text: str) -> dict[str, list[dict]]:
        """Extract named entities using spaCy NER."""
        from app.services.ner_service import extract_entities
        return extract_entities(text)

    @staticmethod
    def merge_extractions(
        regex_results: dict[str, list[str]],
        ner_results: dict[str, list[dict]],
    ) -> dict[str, list[str]]:
        """Merge regex and NER extraction results, deduplicating values."""
        merged: dict[str, set[str]] = {}

        for field_name, values in regex_results.items():
            merged.setdefault(field_name, set()).update(values)

        for field_name, entities in ner_results.items():
            for ent in entities:
                merged.setdefault(field_name, set()).add(ent["value"])

        return {k: list(v) for k, v in merged.items()}

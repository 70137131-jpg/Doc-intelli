import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ClassificationResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    document_type: str
    confidence: float
    method: str
    reasoning: str | None = None
    raw_scores: dict | None = None
    is_corrected: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class ExtractedFieldResponse(BaseModel):
    id: uuid.UUID
    field_name: str
    field_value: str | None = None
    field_type: str | None = None
    confidence: float | None = None
    extraction_method: str | None = None
    is_corrected: bool = False

    model_config = {"from_attributes": True}


class ExtractedFieldsResponse(BaseModel):
    document_id: uuid.UUID
    document_type: str | None = None
    fields: list[ExtractedFieldResponse]


class FieldCorrectionRequest(BaseModel):
    corrections: dict[str, str] = Field(
        ..., description="Map of field_name to corrected_value"
    )


class BatchClassifyRequest(BaseModel):
    document_ids: list[uuid.UUID]

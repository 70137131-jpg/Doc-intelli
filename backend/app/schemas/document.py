import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    message: str = "Document uploaded and queued for processing"


class DocumentStatus(BaseModel):
    id: uuid.UUID
    status: str
    error_message: str | None = None
    page_count: int | None = None
    total_chunks: int = 0


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    original_filename: str
    mime_type: str
    file_size_bytes: int
    status: str
    error_message: str | None = None
    page_count: int | None = None
    total_chunks: int = 0
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    total: int
    page: int
    size: int
    pages: int
    items: list[DocumentResponse]


class DocumentChunkResponse(BaseModel):
    id: uuid.UUID
    chunk_index: int
    content: str
    token_count: int
    page_number: int | None = None
    section_header: str | None = None
    chunking_method: str | None = None

    model_config = {"from_attributes": True}


class BatchUploadResponse(BaseModel):
    documents: list[DocumentUploadResponse]
    total_uploaded: int
    errors: list[dict] = Field(default_factory=list)

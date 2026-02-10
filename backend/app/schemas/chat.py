import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.search import SearchFilters


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    conversation_id: uuid.UUID | None = None
    filters: SearchFilters | None = None
    rerank: bool = True


class SourceReference(BaseModel):
    document_id: uuid.UUID
    document_name: str
    page_number: int | None = None
    chunk_id: uuid.UUID | None = None
    relevance_score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    confidence: str  # high, medium, low
    conversation_id: uuid.UUID
    message_id: uuid.UUID


class ConversationResponse(BaseModel):
    id: uuid.UUID
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: list[SourceReference] | None = None
    confidence: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SSEEvent(BaseModel):
    type: str  # status, sources, token, done, error
    content: str | list | dict | None = None

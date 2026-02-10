import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    document_type: str | None = None
    document_ids: list[uuid.UUID] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|semantic|keyword)$")
    filters: SearchFilters | None = None
    expand_query: bool = False
    rerank: bool = True


class SearchResultItem(BaseModel):
    chunk_id: uuid.UUID | None = None
    document_id: uuid.UUID
    document_name: str
    content: str
    page_number: int | None = None
    chunk_index: int | None = None
    section_header: str | None = None
    score: float
    search_method: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total_results: int
    search_mode: str
    expanded_queries: list[str] | None = None

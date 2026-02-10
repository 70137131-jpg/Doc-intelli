from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkResult:
    content: str
    token_count: int
    chunk_index: int
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    section_header: str | None = None
    metadata: dict = field(default_factory=dict)


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, text: str, page_number: int | None = None) -> list[ChunkResult]:
        """Split text into chunks."""
        ...

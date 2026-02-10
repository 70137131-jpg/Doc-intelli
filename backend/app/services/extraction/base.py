from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PageResult:
    page_number: int
    text: str
    extraction_method: str
    confidence: float = 1.0
    tables: list[dict] | None = None


@dataclass
class ExtractionResult:
    pages: list[PageResult] = field(default_factory=list)
    total_pages: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


class DocumentExtractor(ABC):
    @abstractmethod
    def extract(self, file_data: bytes, filename: str) -> ExtractionResult:
        """Extract text and metadata from a document."""
        ...

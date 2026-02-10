from charset_normalizer import from_bytes

from app.core.logging import get_logger
from app.services.extraction.base import DocumentExtractor, ExtractionResult, PageResult

logger = get_logger(__name__)


class TxtExtractor(DocumentExtractor):
    def extract(self, file_data: bytes, filename: str) -> ExtractionResult:
        # Detect encoding
        detection = from_bytes(file_data).best()
        if detection:
            text = str(detection)
            encoding = detection.encoding
        else:
            text = file_data.decode("utf-8", errors="replace")
            encoding = "utf-8"

        page = PageResult(
            page_number=1,
            text=text,
            extraction_method="plain_text",
            confidence=1.0,
        )

        return ExtractionResult(
            pages=[page],
            total_pages=1,
            metadata={
                "filename": filename,
                "extractor": "txt",
                "encoding": encoding,
            },
        )

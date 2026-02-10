from app.core.exceptions import UnsupportedFileTypeError
from app.services.extraction.base import DocumentExtractor
from app.services.extraction.pdf_extractor import PdfExtractor
from app.services.extraction.docx_extractor import DocxExtractor
from app.services.extraction.txt_extractor import TxtExtractor
from app.services.extraction.csv_extractor import CsvExtractor

EXTRACTOR_MAP: dict[str, type[DocumentExtractor]] = {
    "application/pdf": PdfExtractor,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxExtractor,
    "text/plain": TxtExtractor,
    "text/csv": CsvExtractor,
    "application/vnd.ms-excel": CsvExtractor,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": CsvExtractor,
}


class ExtractorFactory:
    @staticmethod
    def get_extractor(mime_type: str) -> DocumentExtractor:
        extractor_class = EXTRACTOR_MAP.get(mime_type)
        if not extractor_class:
            raise UnsupportedFileTypeError(mime_type)
        return extractor_class()

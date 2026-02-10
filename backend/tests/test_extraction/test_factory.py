import pytest

from app.core.exceptions import UnsupportedFileTypeError
from app.services.extraction.factory import ExtractorFactory
from app.services.extraction.pdf_extractor import PdfExtractor
from app.services.extraction.docx_extractor import DocxExtractor
from app.services.extraction.txt_extractor import TxtExtractor
from app.services.extraction.csv_extractor import CsvExtractor


class TestExtractorFactory:
    def test_get_pdf_extractor(self):
        extractor = ExtractorFactory.get_extractor("application/pdf")
        assert isinstance(extractor, PdfExtractor)

    def test_get_docx_extractor(self):
        extractor = ExtractorFactory.get_extractor(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        assert isinstance(extractor, DocxExtractor)

    def test_get_txt_extractor(self):
        extractor = ExtractorFactory.get_extractor("text/plain")
        assert isinstance(extractor, TxtExtractor)

    def test_get_csv_extractor(self):
        extractor = ExtractorFactory.get_extractor("text/csv")
        assert isinstance(extractor, CsvExtractor)

    def test_get_excel_extractor(self):
        extractor = ExtractorFactory.get_extractor(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert isinstance(extractor, CsvExtractor)

    def test_unsupported_type_raises(self):
        with pytest.raises(UnsupportedFileTypeError):
            ExtractorFactory.get_extractor("application/x-executable")

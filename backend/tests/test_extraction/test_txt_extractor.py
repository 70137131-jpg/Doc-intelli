from app.services.extraction.txt_extractor import TxtExtractor


class TestTxtExtractor:
    def test_extract_plain_text(self):
        extractor = TxtExtractor()
        data = b"Hello, this is a test document.\nIt has multiple lines."
        result = extractor.extract(data, "test.txt")

        assert result.total_pages == 1
        assert len(result.pages) == 1
        assert "Hello" in result.pages[0].text
        assert result.pages[0].extraction_method == "plain_text"
        assert result.pages[0].confidence == 1.0

    def test_extract_empty_text(self):
        extractor = TxtExtractor()
        result = extractor.extract(b"", "empty.txt")

        assert result.total_pages == 1
        assert result.pages[0].text == ""

    def test_extract_utf8_text(self):
        extractor = TxtExtractor()
        data = "Unicode text: café, naïve, résumé".encode("utf-8")
        result = extractor.extract(data, "unicode.txt")

        assert "café" in result.pages[0].text
        assert "résumé" in result.pages[0].text

    def test_full_text_property(self):
        extractor = TxtExtractor()
        data = b"Line 1\nLine 2\nLine 3"
        result = extractor.extract(data, "test.txt")

        assert "Line 1" in result.full_text
        assert "Line 3" in result.full_text

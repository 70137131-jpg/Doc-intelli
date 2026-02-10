from app.services.extraction.csv_extractor import CsvExtractor


class TestCsvExtractor:
    def test_extract_csv(self):
        extractor = CsvExtractor()
        data = b"name,email,age\nAlice,alice@test.com,30\nBob,bob@test.com,25"
        result = extractor.extract(data, "test.csv")

        assert result.total_pages == 1
        assert "Alice" in result.pages[0].text
        assert "Bob" in result.pages[0].text
        assert result.pages[0].extraction_method == "pandas_csv"

    def test_extract_csv_tables(self):
        extractor = CsvExtractor()
        data = b"product,price\nWidget,9.99\nGadget,19.99"
        result = extractor.extract(data, "products.csv")

        assert result.pages[0].tables is not None
        assert len(result.pages[0].tables) == 1
        table = result.pages[0].tables[0]
        assert table["total_rows"] == 2
        assert table["total_columns"] == 2

    def test_extract_csv_metadata(self):
        extractor = CsvExtractor()
        data = b"a,b,c\n1,2,3\n4,5,6"
        result = extractor.extract(data, "data.csv")

        assert result.metadata["row_count"] == 2
        assert result.metadata["column_count"] == 3

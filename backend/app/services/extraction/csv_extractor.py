import io

from app.core.logging import get_logger
from app.services.extraction.base import DocumentExtractor, ExtractionResult, PageResult

logger = get_logger(__name__)


class CsvExtractor(DocumentExtractor):
    def extract(self, file_data: bytes, filename: str) -> ExtractionResult:
        import pandas as pd

        # Determine file type
        is_excel = filename.endswith((".xlsx", ".xls"))

        try:
            if is_excel:
                df = pd.read_excel(io.BytesIO(file_data))
            else:
                df = pd.read_csv(io.BytesIO(file_data))
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            return ExtractionResult(
                pages=[],
                total_pages=0,
                metadata={"filename": filename, "error": str(e)},
            )

        # Convert to text representation
        headers = list(df.columns)
        header_line = " | ".join(str(h) for h in headers)

        rows_text = []
        for _, row in df.iterrows():
            row_text = " | ".join(str(v) for v in row.values)
            rows_text.append(row_text)

        full_text = f"Columns: {header_line}\n\n" + "\n".join(rows_text)

        # Store table structure
        table_data = {
            "table_index": 0,
            "headers": [str(h) for h in headers],
            "rows": df.head(100).values.tolist(),
            "total_rows": len(df),
            "total_columns": len(headers),
        }

        page = PageResult(
            page_number=1,
            text=full_text,
            extraction_method="pandas_csv" if not is_excel else "pandas_excel",
            confidence=1.0,
            tables=[table_data],
        )

        return ExtractionResult(
            pages=[page],
            total_pages=1,
            metadata={
                "filename": filename,
                "extractor": "csv",
                "row_count": len(df),
                "column_count": len(headers),
            },
        )

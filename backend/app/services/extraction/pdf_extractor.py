import io
import tempfile
import os

from app.core.logging import get_logger
from app.services.extraction.base import DocumentExtractor, ExtractionResult, PageResult

logger = get_logger(__name__)


class PdfExtractor(DocumentExtractor):
    # Minimum characters per page to consider text extraction successful
    MIN_TEXT_THRESHOLD = 50

    def extract(self, file_data: bytes, filename: str) -> ExtractionResult:
        pages = []

        # First try native text extraction with pdfplumber
        try:
            pages = self._extract_with_pdfplumber(file_data)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {filename}: {e}")

        # Check if OCR is needed for any pages
        needs_ocr = []
        for page in pages:
            if not page.text or len(page.text.strip()) < self.MIN_TEXT_THRESHOLD:
                needs_ocr.append(page.page_number)

        # If all pages or most pages need OCR, run OCR on those pages
        if needs_ocr:
            logger.info(f"Running OCR on {len(needs_ocr)} pages of {filename}")
            try:
                ocr_pages = self._extract_with_ocr(file_data, needs_ocr)
                page_map = {p.page_number: p for p in pages}
                for ocr_page in ocr_pages:
                    page_map[ocr_page.page_number] = ocr_page
                pages = sorted(page_map.values(), key=lambda p: p.page_number)

                # PaddleOCR fallback for low-confidence Tesseract results
                still_bad = [
                    p.page_number for p in pages
                    if p.extraction_method == "tesseract_ocr"
                    and (not p.text or p.confidence < 0.5)
                ]
                if still_bad:
                    logger.info(f"Tesseract low-confidence on {len(still_bad)} pages, trying PaddleOCR")
                    try:
                        from app.services.extraction.paddle_ocr_extractor import extract_with_paddle_ocr
                        paddle_pages = extract_with_paddle_ocr(file_data, still_bad)
                        for pp in paddle_pages:
                            existing = page_map.get(pp.page_number)
                            if pp.text and (not existing or pp.confidence > existing.confidence):
                                page_map[pp.page_number] = pp
                        pages = sorted(page_map.values(), key=lambda p: p.page_number)
                    except Exception as e2:
                        logger.warning(f"PaddleOCR fallback failed: {e2}")

            except Exception as e:
                logger.warning(f"OCR extraction failed for {filename}: {e}")

        # Apply text cleaning to all pages
        from app.services.extraction.text_cleaner import clean_page_text
        total = len(pages)
        for page in pages:
            if page.text:
                page.text = clean_page_text(page.text, page.page_number, total)

        return ExtractionResult(
            pages=pages,
            total_pages=len(pages),
            metadata={"filename": filename, "extractor": "pdf"},
        )

    def _extract_with_pdfplumber(self, file_data: bytes) -> list[PageResult]:
        import pdfplumber

        pages = []
        with pdfplumber.open(io.BytesIO(file_data)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                # Extract tables
                tables = []
                try:
                    raw_tables = page.extract_tables()
                    for table_idx, table in enumerate(raw_tables):
                        if table:
                            tables.append({
                                "table_index": table_idx,
                                "rows": table,
                                "headers": table[0] if table else [],
                            })
                except Exception:
                    pass

                pages.append(PageResult(
                    page_number=i + 1,
                    text=text,
                    extraction_method="pdfplumber",
                    confidence=1.0 if len(text.strip()) >= self.MIN_TEXT_THRESHOLD else 0.5,
                    tables=tables if tables else None,
                ))

        return pages

    def _extract_with_ocr(self, file_data: bytes, page_numbers: list[int]) -> list[PageResult]:
        from pdf2image import convert_from_bytes
        import pytesseract

        pages = []

        # Convert PDF pages to images
        images = convert_from_bytes(file_data, dpi=300)

        for page_num in page_numbers:
            if page_num <= len(images):
                image = images[page_num - 1]
                try:
                    # Run Tesseract OCR
                    ocr_data = pytesseract.image_to_data(
                        image, output_type=pytesseract.Output.DICT
                    )
                    text = pytesseract.image_to_string(image)

                    # Calculate average confidence
                    confidences = [
                        int(c)
                        for c in ocr_data["conf"]
                        if str(c).isdigit() and int(c) > 0
                    ]
                    avg_confidence = (
                        sum(confidences) / len(confidences) / 100
                        if confidences
                        else 0.0
                    )

                    pages.append(PageResult(
                        page_number=page_num,
                        text=text.strip(),
                        extraction_method="tesseract_ocr",
                        confidence=avg_confidence,
                    ))
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num}: {e}")
                    pages.append(PageResult(
                        page_number=page_num,
                        text="",
                        extraction_method="tesseract_ocr",
                        confidence=0.0,
                    ))

        return pages

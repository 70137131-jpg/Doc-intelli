"""
PaddleOCR fallback extractor for complex layouts and non-English documents.
Used when Tesseract OCR fails or produces low-confidence results.
"""

import io

from app.core.logging import get_logger
from app.services.extraction.base import PageResult

logger = get_logger(__name__)

_paddle_ocr_instance = None


def _get_paddle_ocr():
    """Lazy singleton for PaddleOCR to avoid slow import on startup."""
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except ImportError:
            logger.warning("PaddleOCR not installed. Install with: pip install paddleocr")
            return None
    return _paddle_ocr_instance


def extract_with_paddle_ocr(file_data: bytes, page_numbers: list[int]) -> list[PageResult]:
    """Run PaddleOCR on specified pages of a PDF. Returns PageResult list."""
    ocr = _get_paddle_ocr()
    if ocr is None:
        return []

    try:
        from pdf2image import convert_from_bytes
        import numpy as np
    except ImportError:
        logger.warning("pdf2image or numpy not installed for PaddleOCR pipeline")
        return []

    pages = []

    try:
        images = convert_from_bytes(file_data, dpi=300)
    except Exception as e:
        logger.error(f"PaddleOCR pdf2image conversion failed: {e}")
        return []

    for page_num in page_numbers:
        if page_num > len(images):
            continue

        image = images[page_num - 1]
        try:
            img_array = np.array(image)
            result = ocr.ocr(img_array, cls=True)

            if not result or not result[0]:
                pages.append(PageResult(
                    page_number=page_num,
                    text="",
                    extraction_method="paddleocr",
                    confidence=0.0,
                ))
                continue

            lines = []
            confidences = []
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                lines.append(text)
                confidences.append(conf)

            full_text = "\n".join(lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            pages.append(PageResult(
                page_number=page_num,
                text=full_text.strip(),
                extraction_method="paddleocr",
                confidence=avg_confidence,
            ))

        except Exception as e:
            logger.error(f"PaddleOCR failed for page {page_num}: {e}")
            pages.append(PageResult(
                page_number=page_num,
                text="",
                extraction_method="paddleocr",
                confidence=0.0,
            ))

    return pages

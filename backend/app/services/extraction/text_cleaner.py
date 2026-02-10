"""
Text cleaning pipeline for extracted document text.
Removes headers, footers, page numbers, watermarks, and other noise.
"""

import re

from app.core.logging import get_logger

logger = get_logger(__name__)

# Common header/footer patterns
PAGE_NUMBER_PATTERNS = [
    re.compile(r"^\s*(?:Page|Pg\.?)\s*\d+\s*(?:of\s*\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE),  # Standalone numbers like "- 5 -"
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$", re.MULTILINE),  # "5/10" format
]

WATERMARK_PATTERNS = [
    re.compile(r"(?:CONFIDENTIAL|DRAFT|SAMPLE|COPY|DO NOT DISTRIBUTE|INTERNAL USE ONLY)",
               re.IGNORECASE),
    re.compile(r"(?:WATERMARK|PRIVILEGED|NOT FOR DISTRIBUTION)", re.IGNORECASE),
]

HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*(?:Confidential|Proprietary|Internal).*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*©\s*\d{4}.*(?:All Rights Reserved|Inc\.|LLC|Ltd).*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(?:Printed|Generated)\s+(?:on|at)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*$",
               re.IGNORECASE | re.MULTILINE),
]

# Excessive whitespace / formatting artifacts
NOISE_PATTERNS = [
    re.compile(r"\f"),  # Form feed characters
    re.compile(r"_{5,}"),  # Long underscore lines
    re.compile(r"-{5,}"),  # Long dash lines
    re.compile(r"={5,}"),  # Long equals lines
    re.compile(r"\.{5,}"),  # Long dot leaders (table of contents artifacts)
]


def clean_text(
    text: str,
    remove_page_numbers: bool = True,
    remove_watermarks: bool = True,
    remove_headers_footers: bool = True,
    remove_noise: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    """Clean extracted document text by removing common noise artifacts."""
    if not text:
        return text

    cleaned = text

    if remove_page_numbers:
        for pattern in PAGE_NUMBER_PATTERNS:
            cleaned = pattern.sub("", cleaned)

    if remove_watermarks:
        for pattern in WATERMARK_PATTERNS:
            cleaned = pattern.sub("", cleaned)

    if remove_headers_footers:
        for pattern in HEADER_FOOTER_PATTERNS:
            cleaned = pattern.sub("", cleaned)

    if remove_noise:
        for pattern in NOISE_PATTERNS:
            cleaned = pattern.sub("", cleaned)

    if normalize_whitespace:
        # Collapse multiple blank lines into at most two
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        # Remove trailing whitespace on each line
        cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)
        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

    return cleaned


def clean_page_text(text: str, page_number: int, total_pages: int) -> str:
    """Clean text for a specific page, with page-position-aware heuristics.

    First and last pages are more likely to have headers/footers.
    """
    cleaned = clean_text(text)

    # First page often has cover page noise
    if page_number == 1:
        # Remove very short first lines that are likely headers
        lines = cleaned.split("\n")
        while lines and len(lines[0].strip()) < 5 and lines[0].strip().isdigit():
            lines.pop(0)
        cleaned = "\n".join(lines)

    # Last page often has disclaimers/footers
    if page_number == total_pages:
        lines = cleaned.split("\n")
        while lines and len(lines[-1].strip()) < 5 and lines[-1].strip().isdigit():
            lines.pop()
        cleaned = "\n".join(lines)

    return cleaned

"""
spaCy NER service for named entity extraction from documents.
Complements regex-based extraction with ML-based entity recognition.
"""

from app.core.logging import get_logger

logger = get_logger(__name__)

_nlp = None


def _get_nlp():
    """Lazy-load spaCy model. Falls back gracefully if not installed."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                return None
        except ImportError:
            logger.warning("spaCy not installed. Install with: pip install spacy")
            return None
    return _nlp


# Map spaCy entity labels to our field names
ENTITY_FIELD_MAP = {
    "PERSON": "person_name",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "DATE": "date",
    "MONEY": "amount",
    "CARDINAL": "number",
    "ORDINAL": "number",
    "FAC": "facility",
    "PRODUCT": "product",
    "EVENT": "event",
    "LAW": "legal_reference",
    "WORK_OF_ART": "title",
    "PERCENT": "percentage",
}


def extract_entities(text: str, max_length: int = 100_000) -> dict[str, list[dict]]:
    """Extract named entities from text using spaCy NER.

    Returns dict mapping field names to lists of entity dicts.
    Example: {"person_name": [{"value": "John Doe", "start": 0, "end": 8, "label": "PERSON"}]}
    """
    nlp = _get_nlp()
    if nlp is None:
        return {}

    # Truncate very long documents to avoid memory issues
    truncated = text[:max_length] if len(text) > max_length else text
    doc = nlp(truncated)

    results: dict[str, list[dict]] = {}

    for ent in doc.ents:
        field_name = ENTITY_FIELD_MAP.get(ent.label_)
        if field_name is None:
            continue

        entity_info = {
            "value": ent.text.strip(),
            "start": ent.start_char,
            "end": ent.end_char,
            "label": ent.label_,
            "confidence": 0.85,  # spaCy doesn't provide per-entity confidence natively
        }

        if field_name not in results:
            results[field_name] = []

        # Deduplicate by value
        existing_values = {e["value"] for e in results[field_name]}
        if entity_info["value"] not in existing_values:
            results[field_name].append(entity_info)

    return results

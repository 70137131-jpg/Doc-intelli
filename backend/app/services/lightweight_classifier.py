"""
Lightweight document classifier using a local DeBERTa/DistilBERT model.
Used as the first-pass classifier; falls back to LLM for low-confidence results.
"""

from app.core.logging import get_logger

logger = get_logger(__name__)

_model = None
_tokenizer = None

LABELS = ["Invoice", "Contract", "Report", "Resume", "Letter", "Other"]


def _load_model():
    """Lazy-load the lightweight classification model."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import os
    model_dir = os.getenv("CLASSIFIER_MODEL_DIR", "ml/classification/output/model")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Classifier model not found at {model_dir}. Train it first.")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)

    return _model, _tokenizer


def classify_with_lightweight_model(text: str, max_length: int = 512) -> dict | None:
    """Classify document text using the lightweight local model.

    Returns:
        {"category": str, "confidence": float, "scores": dict} or None if model unavailable
    """
    try:
        import torch
        model, tokenizer = _load_model()
    except (ImportError, FileNotFoundError) as e:
        logger.debug(f"Lightweight classifier not available: {e}")
        return None

    # Truncate to first N tokens for classification
    inputs = tokenizer(
        text[:2000],  # Pre-truncate to avoid tokenizer slowness on very long text
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_idx = probs.argmax().item()

    # Map model labels to our labels
    id2label = model.config.id2label if hasattr(model.config, "id2label") else None
    if id2label:
        category = id2label.get(predicted_idx, LABELS[predicted_idx % len(LABELS)])
    else:
        category = LABELS[predicted_idx % len(LABELS)]

    return {
        "category": category,
        "confidence": float(probs[predicted_idx]),
        "scores": {
            (id2label.get(i, LABELS[i % len(LABELS)]) if id2label else LABELS[i % len(LABELS)]): float(probs[i])
            for i in range(len(probs))
        },
    }

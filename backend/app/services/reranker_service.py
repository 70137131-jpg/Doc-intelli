from app.core.logging import get_logger
from app.config import settings

logger = get_logger(__name__)

# Lazy-loaded model singleton
_reranker_model = None


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(settings.reranker_model_name)
        logger.info(f"Loaded reranker model: {settings.reranker_model_name}")
    return _reranker_model


class RerankerService:
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """Re-rank search candidates using a cross-encoder model."""
        if not candidates:
            return []

        model = _get_reranker()

        # Create query-document pairs
        pairs = [(query, c["content"]) for c in candidates]

        # Score all pairs
        scores = model.predict(pairs)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # Sort by rerank score descending
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        # Update the main score to use rerank score
        for item in reranked:
            item["score"] = item.pop("rerank_score")
            item["search_method"] = item.get("search_method", "hybrid") + "+reranked"

        return reranked[:top_k]

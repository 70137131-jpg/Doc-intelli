"""
Semantic chunking strategy.

Groups sentences by semantic similarity using embeddings, so that each chunk
contains topically coherent content rather than arbitrary fixed-size splits.
"""

import re

import tiktoken

from app.services.chunking.base import ChunkingStrategy, ChunkResult
from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SemanticChunker(ChunkingStrategy):
    """Splits text into semantically coherent chunks using embedding similarity.

    Algorithm:
    1. Split text into sentences
    2. Compute embeddings for each sentence
    3. Calculate cosine similarity between consecutive sentences
    4. Split at points where similarity drops below threshold (topic boundaries)
    5. Merge small groups to meet minimum chunk size
    """

    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        chunk_size: int | None = None,
        similarity_threshold: float = 0.5,
        min_chunk_sentences: int = 2,
    ):
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self._encoder = tiktoken.get_encoding("cl100k_base")
        self._embedding_model = None

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(settings.embedding_model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, falling back to sentence-based chunking")
                return None
        return self._embedding_model

    def _split_sentences(self, text: str) -> list[str]:
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarities(self, sentences: list[str]) -> list[float]:
        """Compute cosine similarities between consecutive sentences."""
        model = self._get_embedding_model()
        if model is None or len(sentences) < 2:
            return []

        embeddings = model.encode(sentences, show_progress_bar=False)

        import numpy as np
        similarities = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            similarities.append(cos_sim)

        return similarities

    def _find_split_points(self, similarities: list[float]) -> list[int]:
        """Find indices where topic shifts occur (low similarity)."""
        split_points = []
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                split_points.append(i + 1)  # Split AFTER sentence i
        return split_points

    def chunk(self, text: str, page_number: int | None = None) -> list[ChunkResult]:
        if not text or not text.strip():
            return []

        total_tokens = self._count_tokens(text)
        if total_tokens <= self.chunk_size:
            return [
                ChunkResult(
                    content=text.strip(),
                    token_count=total_tokens,
                    chunk_index=0,
                    page_number=page_number,
                    char_start=0,
                    char_end=len(text),
                    metadata={"chunking_method": "semantic"},
                )
            ]

        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return [
                ChunkResult(
                    content=text.strip(),
                    token_count=total_tokens,
                    chunk_index=0,
                    page_number=page_number,
                    char_start=0,
                    char_end=len(text),
                    metadata={"chunking_method": "semantic"},
                )
            ]

        # Compute similarities and find split points
        similarities = self._compute_similarities(sentences)

        if not similarities:
            # Fallback: if no embeddings available, use sentence-based splitting
            return self._fallback_chunk(sentences, page_number)

        split_points = self._find_split_points(similarities)

        # Build groups from split points
        groups = []
        prev = 0
        for sp in split_points:
            groups.append(sentences[prev:sp])
            prev = sp
        groups.append(sentences[prev:])

        # Merge small groups
        merged_groups = self._merge_small_groups(groups)

        # Further split groups that exceed max chunk size
        final_groups = []
        for group in merged_groups:
            group_text = " ".join(group)
            if self._count_tokens(group_text) > self.chunk_size:
                # Split this group into sub-chunks by token limit
                final_groups.extend(self._split_by_tokens(group))
            else:
                final_groups.append(group)

        # Build ChunkResults
        chunks = []
        for i, group in enumerate(final_groups):
            chunk_text = " ".join(group)
            chunks.append(
                ChunkResult(
                    content=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    chunk_index=i,
                    page_number=page_number,
                    metadata={"chunking_method": "semantic"},
                )
            )

        return chunks

    def _merge_small_groups(self, groups: list[list[str]]) -> list[list[str]]:
        """Merge groups that are too small into neighboring groups."""
        if not groups:
            return groups

        merged = [groups[0]]
        for group in groups[1:]:
            prev_text = " ".join(merged[-1])
            curr_text = " ".join(group)
            combined_tokens = self._count_tokens(prev_text + " " + curr_text)

            if (len(merged[-1]) < self.min_chunk_sentences or
                    combined_tokens <= self.chunk_size):
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged

    def _split_by_tokens(self, sentences: list[str]) -> list[list[str]]:
        """Split a list of sentences into sub-groups by token limit."""
        groups = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence)
            if current_tokens + sent_tokens > self.chunk_size and current:
                groups.append(current)
                current = []
                current_tokens = 0
            current.append(sentence)
            current_tokens += sent_tokens

        if current:
            groups.append(current)
        return groups

    def _fallback_chunk(self, sentences: list[str], page_number: int | None) -> list[ChunkResult]:
        """Fallback when embeddings are unavailable: split by token count."""
        groups = self._split_by_tokens(sentences)
        chunks = []
        for i, group in enumerate(groups):
            text = " ".join(group)
            chunks.append(
                ChunkResult(
                    content=text,
                    token_count=self._count_tokens(text),
                    chunk_index=i,
                    page_number=page_number,
                    metadata={"chunking_method": "semantic_fallback"},
                )
            )
        return chunks

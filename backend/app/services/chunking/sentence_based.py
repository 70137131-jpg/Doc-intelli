import re

import tiktoken

from app.services.chunking.base import ChunkingStrategy, ChunkResult
from app.config import settings


class SentenceBasedChunker(ChunkingStrategy):
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap_sentences: int = 2,
    ):
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.overlap_sentences = overlap_sentences
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _split_sentences(self, text: str) -> list[str]:
        sentences = self.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

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
                )
            ]

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    ChunkResult(
                        content=chunk_text,
                        token_count=self._count_tokens(chunk_text),
                        chunk_index=chunk_index,
                        page_number=page_number,
                    )
                )
                chunk_index += 1

                # Keep overlap sentences
                if self.overlap_sentences > 0 and len(current_sentences) > self.overlap_sentences:
                    current_sentences = current_sentences[-self.overlap_sentences:]
                    current_tokens = sum(
                        self._count_tokens(s) for s in current_sentences
                    )
                else:
                    current_sentences = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Emit remaining sentences
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                ChunkResult(
                    content=chunk_text,
                    token_count=self._count_tokens(chunk_text),
                    chunk_index=chunk_index,
                    page_number=page_number,
                )
            )

        return chunks

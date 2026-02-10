import tiktoken

from app.services.chunking.base import ChunkingStrategy, ChunkResult
from app.config import settings


class FixedSizeChunker(ChunkingStrategy):
    def __init__(
        self,
        chunk_size: int | None = None,
        overlap_pct: float | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size_tokens
        self.overlap_pct = overlap_pct or settings.chunk_overlap_pct
        self.overlap_tokens = int(self.chunk_size * self.overlap_pct)
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _encode(self, text: str) -> list[int]:
        return self._encoder.encode(text)

    def _decode(self, tokens: list[int]) -> str:
        return self._encoder.decode(tokens)

    def chunk(self, text: str, page_number: int | None = None) -> list[ChunkResult]:
        if not text or not text.strip():
            return []

        tokens = self._encode(text)
        total_tokens = len(tokens)

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

        chunks = []
        start = 0
        chunk_index = 0
        step = self.chunk_size - self.overlap_tokens

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self._decode(chunk_tokens).strip()

            if chunk_text:
                # Approximate character positions
                char_start = len(self._decode(tokens[:start]))
                char_end = len(self._decode(tokens[:end]))

                chunks.append(
                    ChunkResult(
                        content=chunk_text,
                        token_count=len(chunk_tokens),
                        chunk_index=chunk_index,
                        page_number=page_number,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
                chunk_index += 1

            if end >= total_tokens:
                break

            start += step

        return chunks

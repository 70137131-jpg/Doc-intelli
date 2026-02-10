import pytest

from app.services.chunking.fixed_size import FixedSizeChunker
from app.services.chunking.sentence_based import SentenceBasedChunker


class TestFixedSizeChunker:
    def test_short_text_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap_pct=0.15)
        text = "This is a short text."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self):
        chunker = FixedSizeChunker(chunk_size=20, overlap_pct=0.15)
        text = " ".join(["word"] * 100)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.token_count > 0

    def test_overlap_between_chunks(self):
        chunker = FixedSizeChunker(chunk_size=20, overlap_pct=0.5)
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = chunker.chunk(text)

        # With 50% overlap, chunks should share content
        assert len(chunks) > 2

    def test_empty_text(self):
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0

    def test_page_number_passed(self):
        chunker = FixedSizeChunker(chunk_size=100)
        chunks = chunker.chunk("Some text", page_number=5)

        assert chunks[0].page_number == 5


class TestSentenceBasedChunker:
    def test_single_sentence(self):
        chunker = SentenceBasedChunker(chunk_size=100)
        text = "This is a single sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_multiple_sentences(self):
        chunker = SentenceBasedChunker(chunk_size=10, overlap_sentences=1)
        text = "First sentence here. Second sentence follows. Third sentence now. Fourth sentence added. Fifth sentence too."
        chunks = chunker.chunk(text)

        assert len(chunks) >= 2

    def test_empty_text(self):
        chunker = SentenceBasedChunker()
        chunks = chunker.chunk("")
        assert len(chunks) == 0

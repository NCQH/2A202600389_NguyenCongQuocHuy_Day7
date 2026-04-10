from __future__ import annotations

import math
import re
from typing import List

class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Giải thích
        # Sử dụng biểu thức chính quy để tách văn bản thành các câu dựa trên các dấu chấm câu phổ biến (". ", "! ", "? ", ".\n").
        # Biểu thức (?<=[.!?])\s+ có nghĩa là:
        # (?<=[.!?]) là một positive lookbehind assertion, đảm bảo rằng việc tách chỉ xảy ra sau một trong các dấu chấm câu (., !, ?).
        # \s+ có nghĩa là một hoặc nhiều ký tự khoảng trắng, cho phép tách ngay cả khi có nhiều khoảng trắng giữa các câu.

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk = " ".join(sentences[i : i + self.max_sentences_per_chunk]).strip()
            chunks.append(chunk)
        return chunks

class SectionChunker:
    """
        Split text into sections based on headers (lines starting with ##, ### is not included).
        Max chunk size -> int
    """
    
    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> List[str]:
        if not text:
            return []

        sections = re.split(r"(?m)^##\s+", text.strip())
        chunks: List[str] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # If section is too long, split it into smaller fixed-size chunks
                for i in range(0, len(section), self.chunk_size):
                    chunk = section[i:i + self.chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)

        return chunks

class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [current_text]

        separator = remaining_separators[0]
        parts = current_text.split(separator)

        chunks: list[str] = []
        for part in parts:
            sub_chunks = self._split(part, remaining_separators[1:])
            chunks.extend(sub_chunks)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 500) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=20)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=2)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        section_chunker = SectionChunker(chunk_size=chunk_size)

        # return strategy name, stats
        return {
            "fixed_size": {
                "chunks": fixed_chunker.chunk(text),
                "count": len(fixed_chunker.chunk(text)),
                "avg_length": sum(len(chunk) for chunk in fixed_chunker.chunk(text)) / len(fixed_chunker.chunk(text)) if fixed_chunker.chunk(text) else 0

            },
            "by_sentences": {
                "chunks": sentence_chunker.chunk(text),
                "count": len(sentence_chunker.chunk(text)),
                "avg_length": sum(len(chunk) for chunk in sentence_chunker.chunk(text)) / len(sentence_chunker.chunk(text)) if sentence_chunker.chunk(text) else 0
            },
            "recursive": {
                "chunks": recursive_chunker.chunk(text),
                "count": len(recursive_chunker.chunk(text)),
                "avg_length": sum(len(chunk) for chunk in recursive_chunker.chunk(text)) / len(recursive_chunker.chunk(text)) if recursive_chunker.chunk(text) else 0
            },
            "by_section": {
                "chunks": SectionChunker(chunk_size=chunk_size).chunk(text),
                "count": len(SectionChunker(chunk_size=chunk_size).chunk(text)),
                "avg_length": sum(len(chunk) for chunk in SectionChunker(chunk_size=chunk_size).chunk(text)) / len(SectionChunker(chunk_size=chunk_size).chunk(text)) if SectionChunker(chunk_size=chunk_size).chunk(text) else 0
            }
        }
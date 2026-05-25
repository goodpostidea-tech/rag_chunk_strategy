"""Chunk statistics for strategy comparison."""

from chunk_lab.types import ChunkStats
from chunk_lab.utils import count_tokens
from langchain_core.documents import Document


def compute_stats(
    chunks: list[Document],
    *,
    strategy: str,
    category: str,
    extra: dict | None = None,
) -> ChunkStats:
    if not chunks:
        return ChunkStats(
            chunk_count=0,
            avg_char_length=0.0,
            min_char_length=0,
            max_char_length=0,
            avg_token_estimate=0.0,
            total_chars=0,
            strategy=strategy,
            category=category,
            extra=extra or {},
        )

    lengths = [len(c.page_content) for c in chunks]
    tokens = [count_tokens(c.page_content) for c in chunks]
    total_chars = sum(lengths)

    return ChunkStats(
        chunk_count=len(chunks),
        avg_char_length=total_chars / len(chunks),
        min_char_length=min(lengths),
        max_char_length=max(lengths),
        avg_token_estimate=sum(tokens) / len(tokens),
        total_chars=total_chars,
        strategy=strategy,
        category=category,
        extra=extra or {},
    )


def stats_to_dict(stats: ChunkStats) -> dict:
    return {
        "strategy": stats.strategy,
        "category": stats.category,
        "chunk_count": stats.chunk_count,
        "avg_char_length": round(stats.avg_char_length, 1),
        "min_char_length": stats.min_char_length,
        "max_char_length": stats.max_char_length,
        "avg_token_estimate": round(stats.avg_token_estimate, 1),
        "total_chars": stats.total_chars,
        **stats.extra,
    }

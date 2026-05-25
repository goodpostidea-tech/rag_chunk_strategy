"""Run chunk strategies on documents."""

import time

from langchain_core.documents import Document

from chunk_lab.registry import get_strategy
from chunk_lab.stats import stats_to_dict
from chunk_lab.types import ChunkRunResult
from chunk_lab.utils import make_documents


def run_chunk(
    text: str,
    strategy_name: str,
    *,
    source: str = "inline",
    title: str = "",
    section: str = "",
    strategy_params: dict | None = None,
) -> ChunkRunResult:
    strategy = get_strategy(strategy_name)
    documents = make_documents(text, source=source, title=title, section=section)
    params = strategy_params or {}

    t0 = time.perf_counter()
    chunks = strategy.chunk(documents, **params)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    extra = {}
    if strategy.name == "multi_granularity":
        sizes = {c.metadata.get("granularity_size") for c in chunks}
        extra["granularities"] = sorted(s for s in sizes if s is not None)

    stats = strategy.analyze(chunks, **extra)
    return ChunkRunResult(
        strategy=strategy.name,
        category=strategy.category,
        description=strategy.description,
        chunks=chunks,
        stats=stats,
        elapsed_ms=round(elapsed_ms, 2),
    )


def compare_strategies(
    text: str,
    strategy_names: list[str] | None = None,
    **kwargs,
) -> list[ChunkRunResult]:
    from chunk_lab.registry import STRATEGY_REGISTRY

    names = strategy_names or list(STRATEGY_REGISTRY.keys())
    results: list[ChunkRunResult] = []
    for name in names:
        try:
            results.append(run_chunk(text, name, **kwargs))
        except Exception as exc:
            from chunk_lab.stats import compute_stats

            results.append(
                ChunkRunResult(
                    strategy=name,
                    category="error",
                    description=str(exc),
                    chunks=[],
                    stats=compute_stats([], strategy=name, category="error"),
                    elapsed_ms=0,
                )
            )
    return results


def chunks_to_response_dict(result: ChunkRunResult, *, max_preview: int = 5) -> dict:
    preview = [
        {
            "content": c.page_content[:500],
            "metadata": c.metadata,
            "char_length": len(c.page_content),
        }
        for c in result.chunks[:max_preview]
    ]
    return {
        "strategy": result.strategy,
        "category": result.category,
        "description": result.description,
        "stats": stats_to_dict(result.stats),
        "elapsed_ms": result.elapsed_ms,
        "chunk_count": len(result.chunks),
        "preview": preview,
    }

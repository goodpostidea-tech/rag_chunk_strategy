"""Vector store engineering benchmark — build time, latency, memory, filter slowdown."""

from __future__ import annotations

import gc
import statistics
import time
import uuid
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from chunk_lab.config import Settings, get_settings
from vstore_lab.types import VStoreBenchResponse, VStoreBenchRow
from chunk_lab.models import get_embeddings


def _memory_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def generate_synthetic_documents(
    count: int,
    *,
    prefix: str = "bench",
    category: str = "general",
) -> list[Document]:
    """Generate synthetic chunks for scale testing."""
    docs: list[Document] = []
    for i in range(count):
        docs.append(
            Document(
                page_content=(
                    f"{prefix} document chunk {i}: "
                    f"用于向量库压测的 synthetic 文本，编号 {i % 1000}。"
                ),
                metadata={
                    "doc_id": f"{prefix}_{i // 100}",
                    "chunk_index": i,
                    "category": category if i % 2 == 0 else "other",
                },
            )
        )
    return docs


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def _bench_queries(
    search_fn: Callable[[str, int], list],
    *,
    queries: list[str],
    top_k: int = 5,
) -> tuple[float, float]:
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q, top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
    return _percentile(latencies, 50), _percentile(latencies, 99)


BACKEND_SPECS: list[dict[str, Any]] = [
    {
        "provider": "faiss",
        "index_type": "flat",
        "description": "FAISS Flat — 精确搜索，适合小规模",
    },
    {
        "provider": "faiss",
        "index_type": "ivf",
        "description": "FAISS IVF — 近似索引，适合中大规模",
    },
    {
        "provider": "chroma",
        "index_type": "default",
        "description": "Chroma 嵌入式向量库",
    },
    {
        "provider": "qdrant",
        "index_type": "default",
        "description": "Qdrant（需服务可用）",
        "optional": True,
    },
    {
        "provider": "milvus",
        "index_type": "default",
        "description": "Milvus（需服务可用）",
        "optional": True,
    },
]


def list_vstore_backends(*, include_optional: bool = True) -> list[dict]:
    rows = []
    for spec in BACKEND_SPECS:
        if spec.get("optional") and not include_optional:
            continue
        rows.append(
            {
                "provider": spec["provider"],
                "index_type": spec["index_type"],
                "description": spec["description"],
                "optional": spec.get("optional", False),
            }
        )
    return rows


def _build_faiss(
    docs: list[Document],
    embedding: Embeddings,
    *,
    index_type: str,
) -> Any:
    import faiss
    import numpy as np
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vectors = np.array(embedding.embed_documents(texts), dtype=np.float32)
    dim = vectors.shape[1]
    n = len(texts)

    if index_type == "ivf" and n >= 100:
        nlist = min(int(n**0.5), max(n // 10, 1))
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = min(10, nlist)
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

    docstore = InMemoryDocstore(
        {str(i): Document(page_content=texts[i], metadata=metadatas[i]) for i in range(n)}
    )
    return FAISS(
        embedding_function=embedding,
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: str(i) for i in range(n)},
    )


def _build_vectorstore(
    provider: str,
    index_type: str,
    docs: list[Document],
    embedding: Embeddings,
    settings: Settings,
) -> Any:
    if provider == "faiss":
        return _build_faiss(docs, embedding, index_type=index_type)

    from chunk_lab.vectorstore import create_vectorstore

    name = f"bench_{provider}_{uuid.uuid4().hex[:8]}"
    return create_vectorstore(
        docs,
        embedding,
        settings=settings,
        collection_name=name,
        provider=provider,
    )


def _filtered_search(vs: Any, query: str, top_k: int, filter_meta: dict) -> list:
    """Try metadata-filtered search when backend supports it."""
    if hasattr(vs, "similarity_search"):
        try:
            return vs.similarity_search(query, k=top_k, filter=filter_meta)
        except TypeError:
            pass
        try:
            return vs.similarity_search(
                query,
                k=top_k,
                filter={"category": filter_meta.get("category")},
            )
        except Exception:
            pass
    return vs.similarity_search(query, k=top_k)


def benchmark_backend(
    provider: str,
    vector_count: int,
    *,
    index_type: str = "default",
    settings: Settings | None = None,
    query_count: int = 20,
    top_k: int = 5,
    embedding: Embeddings | None = None,
) -> VStoreBenchRow:
    """Benchmark one backend at a given vector count."""
    cfg = settings or get_settings()
    emb = embedding or get_embeddings(cfg)
    docs = generate_synthetic_documents(vector_count)
    queries = [f"bench query {i} synthetic 压测" for i in range(query_count)]
    filter_meta = {"category": "general"}

    gc.collect()
    mem_before = _memory_mb()

    try:
        t0 = time.perf_counter()
        vs = _build_vectorstore(provider, index_type, docs, emb, cfg)
        build_ms = (time.perf_counter() - t0) * 1000

        def search(q: str, k: int):
            return vs.similarity_search(q, k=k)

        p50, p99 = _bench_queries(search, queries=queries, top_k=top_k)

        filter_slowdown = None
        try:
            fp50, _ = _bench_queries(
                lambda q, k: _filtered_search(vs, q, k, filter_meta),
                queries=queries[:5],
                top_k=top_k,
            )
            if p50 > 0:
                filter_slowdown = round(fp50 / p50, 4)
        except Exception:
            filter_slowdown = None

        mem_after = _memory_mb()
        memory_mb = (mem_after - mem_before) if (mem_before and mem_after) else mem_after

        return VStoreBenchRow(
            provider=provider,
            index_type=index_type,
            vector_count=vector_count,
            build_ms=round(build_ms, 2),
            query_p50_ms=round(p50, 2),
            query_p99_ms=round(p99, 2),
            memory_mb=round(memory_mb, 2) if memory_mb else None,
            filter_slowdown_ratio=filter_slowdown,
        )
    except Exception as exc:
        return VStoreBenchRow(
            provider=provider,
            index_type=index_type,
            vector_count=vector_count,
            build_ms=0.0,
            query_p50_ms=0.0,
            query_p99_ms=0.0,
            error=str(exc),
        )


def run_vstore_benchmark(
    vector_counts: list[int] | None = None,
    providers: list[str] | None = None,
    *,
    settings: Settings | None = None,
) -> VStoreBenchResponse:
    """Run benchmark matrix across sizes and backends."""
    counts = vector_counts or [1000, 10000]
    specs = BACKEND_SPECS
    if providers:
        specs = [s for s in specs if s["provider"] in providers]

    t0 = time.perf_counter()
    rows: list[VStoreBenchRow] = []
    for spec in specs:
        for n in counts:
            row = benchmark_backend(
                spec["provider"],
                n,
                index_type=spec.get("index_type", "default"),
                settings=settings,
            )
            rows.append(row)

    elapsed = (time.perf_counter() - t0) * 1000
    return VStoreBenchResponse(
        vector_counts=counts,
        rows=rows,
        elapsed_ms=round(elapsed, 2),
    )


def vstore_response_to_dict(resp: VStoreBenchResponse) -> dict:
    return {
        "vector_counts": resp.vector_counts,
        "elapsed_ms": resp.elapsed_ms,
        "rows": [
            {
                "provider": r.provider,
                "index_type": r.index_type,
                "vector_count": r.vector_count,
                "build_ms": r.build_ms,
                "query_p50_ms": r.query_p50_ms,
                "query_p99_ms": r.query_p99_ms,
                "memory_mb": r.memory_mb,
                "filter_slowdown_ratio": r.filter_slowdown_ratio,
                "error": r.error,
            }
            for r in resp.rows
        ],
    }

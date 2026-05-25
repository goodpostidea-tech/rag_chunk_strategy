"""Retrieval experiment pipeline — chunk corpus then evaluate retrieval methods."""

import time
from typing import Any

from langchain_core.documents import Document

from retrieval_lab.metrics import evaluate_retrieval_qa
from retrieval_lab.registry import get_retrieval_method, list_retrieval_methods
from retrieval_lab.types import RetrievalEvalResponse, RetrievalQAPair
from chunk_lab.pipeline import run_chunk
from chunk_lab.utils import make_documents


def evaluate_retrieval_experiment(
    text: str,
    retrieval_method: str,
    qa_pairs: list[RetrievalQAPair],
    *,
    chunk_strategy: str = "recursive_baseline",
    top_k: int = 5,
    doc_title: str = "",
    retrieval_params: dict[str, Any] | None = None,
    chunk_params: dict[str, Any] | None = None,
) -> RetrievalEvalResponse:
    """Full retrieval experiment: chunk → index → retrieve → metrics."""
    chunk_run = run_chunk(
        text,
        chunk_strategy,
        title=doc_title,
        strategy_params=chunk_params or {},
    )
    return evaluate_retrieval_on_chunks(
        chunk_run.chunks,
        retrieval_method,
        qa_pairs,
        top_k=top_k,
        chunk_strategy=chunk_strategy,
        retrieval_params=retrieval_params,
        chunk_elapsed_ms=chunk_run.elapsed_ms,
    )


def evaluate_retrieval_on_chunks(
    chunks: list[Document],
    retrieval_method: str,
    qa_pairs: list[RetrievalQAPair],
    *,
    top_k: int = 5,
    chunk_strategy: str = "recursive_baseline",
    retrieval_params: dict[str, Any] | None = None,
    chunk_elapsed_ms: float = 0.0,
) -> RetrievalEvalResponse:
    if not chunks:
        return RetrievalEvalResponse(
            method=retrieval_method,
            chunk_strategy=chunk_strategy,
            top_k=top_k,
            context_recall=0.0,
            context_precision=0.0,
            mrr=0.0,
            hits=0,
            total=len(qa_pairs),
        )

    method = get_retrieval_method(retrieval_method)
    params = retrieval_params or {}

    t0 = time.perf_counter()
    method.build_index(chunks, **params)

    def retrieve_fn(query: str, k: int) -> list[Document]:
        return method.retrieve(query, k)

    details, summary, by_type = evaluate_retrieval_qa(
        qa_pairs, retrieve_fn, top_k=top_k
    )
    elapsed = (time.perf_counter() - t0) * 1000 + chunk_elapsed_ms

    return RetrievalEvalResponse(
        method=retrieval_method,
        chunk_strategy=chunk_strategy,
        top_k=top_k,
        context_recall=summary["context_recall"],
        context_precision=summary["context_precision"],
        mrr=summary["mrr"],
        hits=summary["hits"],
        total=summary["total"],
        by_query_type=by_type,
        details=details,
        elapsed_ms=round(elapsed, 2),
    )


def evaluate_retrieval_on_documents(
    documents: list[Document],
    retrieval_method: str,
    qa_pairs: list[RetrievalQAPair],
    *,
    chunk_strategy: str = "recursive_baseline",
    top_k: int = 5,
    chunk_params: dict[str, Any] | None = None,
    retrieval_params: dict[str, Any] | None = None,
) -> RetrievalEvalResponse:
    from chunk_lab.registry import get_strategy

    strategy = get_strategy(chunk_strategy)
    t0 = time.perf_counter()
    chunks = strategy.chunk(documents, **(chunk_params or {}))
    chunk_ms = (time.perf_counter() - t0) * 1000
    return evaluate_retrieval_on_chunks(
        chunks,
        retrieval_method,
        qa_pairs,
        top_k=top_k,
        chunk_strategy=chunk_strategy,
        retrieval_params=retrieval_params,
        chunk_elapsed_ms=chunk_ms,
    )


def compare_retrieval_methods(
    text: str,
    qa_pairs: list[RetrievalQAPair],
    methods: list[str] | None = None,
    **kwargs: Any,
) -> list[RetrievalEvalResponse]:
    names = methods or [m["name"] for m in list_retrieval_methods()]
    chunk_run = run_chunk(text, kwargs.get("chunk_strategy", "recursive_baseline"))
    results: list[RetrievalEvalResponse] = []
    for name in names:
        try:
            results.append(
                evaluate_retrieval_on_chunks(
                    chunk_run.chunks,
                    name,
                    qa_pairs,
                    top_k=kwargs.get("top_k", 5),
                    chunk_strategy=chunk_run.strategy,
                    chunk_elapsed_ms=0,
                )
            )
        except Exception as exc:
            err_resp = RetrievalEvalResponse(
                method=name,
                chunk_strategy=chunk_run.strategy,
                top_k=kwargs.get("top_k", 5),
                context_recall=0.0,
                context_precision=0.0,
                mrr=0.0,
                hits=0,
                total=len(qa_pairs),
                details=[],
                elapsed_ms=0.0,
            )
            err_resp.by_query_type = {"_error": {"message": str(exc), "count": 0}}
            results.append(err_resp)
    return results


def retrieval_response_to_dict(resp: RetrievalEvalResponse) -> dict:
    return {
        "method": resp.method,
        "chunk_strategy": resp.chunk_strategy,
        "top_k": resp.top_k,
        "context_recall": resp.context_recall,
        "context_precision": resp.context_precision,
        "mrr": resp.mrr,
        "hits": resp.hits,
        "total": resp.total,
        "by_query_type": resp.by_query_type,
        "elapsed_ms": resp.elapsed_ms,
        "details": [
            {
                "question": d.question,
                "query_type": d.query_type,
                "context_recall": d.context_recall,
                "context_precision": d.context_precision,
                "reciprocal_rank": d.reciprocal_rank,
                "top_preview": d.top_preview,
            }
            for d in resp.details
        ],
    }

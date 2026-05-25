"""Retrieval experiment metrics: context recall, precision, MRR."""

from retrieval_lab.types import RetrievalDetail, RetrievalQAPair
from langchain_core.documents import Document


def _needle(qa: RetrievalQAPair) -> str:
    return (qa.expected_in_chunk or qa.expected_answer or "").strip().lower()


def context_recall_hit(qa: RetrievalQAPair, docs: list[Document]) -> bool:
    needle = _needle(qa)
    if not needle:
        return False
    return any(needle in d.page_content.lower() for d in docs)


def reciprocal_rank(qa: RetrievalQAPair, docs: list[Document]) -> float:
    needle = _needle(qa)
    if not needle:
        return 0.0
    for i, doc in enumerate(docs):
        if needle in doc.page_content.lower():
            return 1.0 / (i + 1)
    return 0.0


def context_precision(qa: RetrievalQAPair, docs: list[Document]) -> float:
    """Fraction of retrieved chunks that contain the expected span."""
    needle = _needle(qa)
    if not needle or not docs:
        return 0.0
    relevant = sum(1 for d in docs if needle in d.page_content.lower())
    return relevant / len(docs)


def evaluate_retrieval_qa(
    qa_pairs: list[RetrievalQAPair],
    retrieve_fn,
    *,
    top_k: int,
) -> tuple[list[RetrievalDetail], dict[str, float], dict[str, dict[str, float]]]:
    """Run QA set and aggregate metrics."""
    details: list[RetrievalDetail] = []
    recalls: list[float] = []
    precisions: list[float] = []
    rrs: list[float] = []

    by_type: dict[str, list[RetrievalDetail]] = {}

    for qa in qa_pairs:
        docs = retrieve_fn(qa.question, top_k)
        hit = context_recall_hit(qa, docs)
        prec = context_precision(qa, docs)
        rr = reciprocal_rank(qa, docs)

        recalls.append(1.0 if hit else 0.0)
        precisions.append(prec)
        rrs.append(rr)

        detail = RetrievalDetail(
            question=qa.question,
            query_type=qa.query_type,
            context_recall=hit,
            context_precision=round(prec, 4),
            reciprocal_rank=round(rr, 4),
            top_preview=docs[0].page_content[:200] if docs else "",
        )
        details.append(detail)
        by_type.setdefault(qa.query_type, []).append(detail)

    total = len(qa_pairs) or 1
    summary = {
        "context_recall": round(sum(recalls) / total, 4),
        "context_precision": round(sum(precisions) / total, 4),
        "mrr": round(sum(rrs) / total, 4),
        "hits": int(sum(recalls)),
        "total": len(qa_pairs),
    }

    type_summary: dict[str, dict[str, float]] = {}
    for qt, items in by_type.items():
        n = len(items) or 1
        type_summary[qt] = {
            "context_recall": round(
                sum(1 for d in items if d.context_recall) / n, 4
            ),
            "context_precision": round(
                sum(d.context_precision for d in items) / n, 4
            ),
            "mrr": round(sum(d.reciprocal_rank for d in items) / n, 4),
            "count": len(items),
        }

    return details, summary, type_summary

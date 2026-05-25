"""Reciprocal Rank Fusion for hybrid retrieval."""

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    *,
    top_k: int = 5,
    k: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists with RRF (Cormack et al., k=60 default)."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content[:200] + str(doc.metadata.get("start_index", ""))
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out: list[Document] = []
    for key, _ in fused[:top_k]:
        out.append(doc_map[key])
    return out

"""In-memory BM25 index (rank-bm25)."""

import re
from typing import Any

from langchain_core.documents import Document


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for Chinese + English."""
    text = text.lower()
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text)
    return tokens or [text[:32]] if text else ["empty"]


class Bm25Index:
    """BM25 sparse retrieval over chunk texts."""

    def __init__(self, chunks: list[Document]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.corpus = [_tokenize(c.page_content) for c in chunks]
        self._bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int) -> list[Document]:
        if not self.chunks:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        return [self.chunks[i] for i in ranked if scores[i] > 0] or [
            self.chunks[i] for i in ranked[:top_k]
        ]

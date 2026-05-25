"""Retrieval strategy implementations: dense / BM25 / hybrid / hybrid+rerank."""

from typing import Any

from langchain_core.documents import Document

from chunk_lab.config import Settings, get_settings
from retrieval_lab.base import BaseRetrievalStrategy
from retrieval_lab.bm25_index import Bm25Index
from retrieval_lab.rrf import reciprocal_rank_fusion
from chunk_lab.models import get_embeddings
from chunk_lab.vectorstore import create_vectorstore, similarity_search


class DenseRetrievalStrategy(BaseRetrievalStrategy):
    name = "dense"
    description = "稠密向量检索（嵌入余弦相似度）"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._vectorstore = None

    def build_index(self, chunks: list[Document], **kwargs: Any) -> None:
        embeddings = kwargs.get("embeddings") or get_embeddings(self.settings)
        self._vectorstore = create_vectorstore(
            chunks,
            embeddings,
            settings=self.settings,
            collection_name=kwargs.get("collection_name"),
        )

    def retrieve(self, query: str, top_k: int, **kwargs: Any) -> list[Document]:
        if self._vectorstore is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        return similarity_search(self._vectorstore, query, top_k)


class Bm25RetrievalStrategy(BaseRetrievalStrategy):
    name = "bm25"
    description = "BM25 稀疏检索（关键词匹配，无需 GPU）"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._bm25: Bm25Index | None = None

    def build_index(self, chunks: list[Document], **kwargs: Any) -> None:
        self._bm25 = Bm25Index(chunks)

    def retrieve(self, query: str, top_k: int, **kwargs: Any) -> list[Document]:
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        return self._bm25.search(query, top_k)


class HybridRetrievalStrategy(BaseRetrievalStrategy):
    name = "hybrid"
    description = "混合检索：Dense + BM25，RRF 融合排名"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._dense = DenseRetrievalStrategy(self.settings)
        self._bm25 = Bm25RetrievalStrategy(self.settings)
        self._rrf_k: int = 60

    def build_index(self, chunks: list[Document], **kwargs: Any) -> None:
        self._rrf_k = int(kwargs.get("rrf_k", 60))
        self._dense.build_index(chunks, **kwargs)
        self._bm25.build_index(chunks, **kwargs)

    def retrieve(self, query: str, top_k: int, **kwargs: Any) -> list[Document]:
        fetch_k = max(top_k * 4, 20)
        dense_docs = self._dense.retrieve(query, fetch_k)
        bm25_docs = self._bm25.retrieve(query, fetch_k)
        return reciprocal_rank_fusion(
            [dense_docs, bm25_docs],
            top_k=top_k,
            k=self._rrf_k,
        )


class HybridRerankRetrievalStrategy(BaseRetrievalStrategy):
    name = "hybrid_rerank"
    description = "混合检索 + Cross-Encoder 精排（BGE-reranker 等）"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._hybrid = HybridRetrievalStrategy(self.settings)
        self._rerank_model: str = "BAAI/bge-reranker-base"
        self._reranker = None

    def build_index(self, chunks: list[Document], **kwargs: Any) -> None:
        self._rerank_model = kwargs.get("rerank_model", self._rerank_model)
        self._hybrid.build_index(chunks, **kwargs)

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(
                self._rerank_model,
                device=self.settings.embedding_device,
            )
        return self._reranker

    def retrieve(self, query: str, top_k: int, **kwargs: Any) -> list[Document]:
        candidate_k = int(kwargs.get("candidate_k", max(top_k * 10, 50)))
        candidates = self._hybrid.retrieve(query, candidate_k)
        if not candidates:
            return []
        reranker = self._get_reranker()
        pairs = [(query, d.page_content) for d in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(
            zip(scores, candidates, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )
        return [doc for _, doc in ranked[:top_k]]

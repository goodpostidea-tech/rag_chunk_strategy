"""Retrieval method registry."""

from retrieval_lab.base import BaseRetrievalStrategy
from retrieval_lab.strategies import (
    Bm25RetrievalStrategy,
    DenseRetrievalStrategy,
    HybridRerankRetrievalStrategy,
    HybridRetrievalStrategy,
)

RETRIEVAL_REGISTRY: dict[str, type[BaseRetrievalStrategy]] = {
    DenseRetrievalStrategy.name: DenseRetrievalStrategy,
    Bm25RetrievalStrategy.name: Bm25RetrievalStrategy,
    HybridRetrievalStrategy.name: HybridRetrievalStrategy,
    HybridRerankRetrievalStrategy.name: HybridRerankRetrievalStrategy,
}


def list_retrieval_methods() -> list[dict]:
    return [
        {"name": cls.name, "description": cls.description}
        for cls in RETRIEVAL_REGISTRY.values()
    ]


def get_retrieval_method(name: str, **init_kwargs) -> BaseRetrievalStrategy:
    key = name.strip().lower()
    if key not in RETRIEVAL_REGISTRY:
        available = ", ".join(sorted(RETRIEVAL_REGISTRY))
        raise ValueError(f"Unknown retrieval method '{name}'. Available: {available}")
    return RETRIEVAL_REGISTRY[key](**init_kwargs)

"""Base retrieval strategy interface."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document


class BaseRetrievalStrategy(ABC):
    name: str
    description: str

    @abstractmethod
    def build_index(self, chunks: list[Document], **kwargs: Any) -> None:
        """Build in-memory index from chunked documents."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int, **kwargs: Any) -> list[Document]:
        """Return top-k documents for query."""

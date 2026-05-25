"""Base chunk strategy interface."""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from chunk_lab.config import Settings, get_settings
from chunk_lab.stats import compute_stats
from chunk_lab.types import ChunkStats


class BaseChunkStrategy(ABC):
    name: str
    category: str  # A | B | C | baseline | utility
    description: str

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @abstractmethod
    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        """Split documents into chunks."""

    def analyze(self, chunks: list[Document], **extra: Any) -> ChunkStats:
        return compute_stats(
            chunks,
            strategy=self.name,
            category=self.category,
            extra=extra if isinstance(extra, dict) else {},
        )

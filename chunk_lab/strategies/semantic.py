"""Semantic chunker — embedding similarity breakpoints."""

from typing import Any

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from chunk_lab.base import BaseChunkStrategy
from chunk_lab.models import get_embeddings


class SemanticChunkStrategy(BaseChunkStrategy):
    name = "semantic"
    category = "A"
    description = "语义分块（嵌入相似度边界），建议设置 min_chunk_size 防碎片化"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        embeddings = kwargs.get("embeddings") or get_embeddings(self.settings)
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=kwargs.get(
                "breakpoint_threshold_type",
                self.settings.semantic_breakpoint_type,
            ),
            breakpoint_threshold_amount=kwargs.get(
                "breakpoint_threshold_amount",
                self.settings.semantic_breakpoint_amount,
            ),
            min_chunk_size=kwargs.get(
                "min_chunk_size",
                self.settings.semantic_min_chunk_size,
            ),
            add_start_index=True,
        )
        return splitter.split_documents(documents)

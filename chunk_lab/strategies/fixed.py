"""Fixed-size character splitter — no overlap, no recursive separators."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

from chunk_lab.base import BaseChunkStrategy


class FixedSizeStrategy(BaseChunkStrategy):
    name = "fixed_size"
    category = "baseline"
    description = "固定长度分块（按字符数等分，无智能分割点）"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        size = kwargs.get("chunk_size", self.settings.baseline_chunk_size)
        overlap = kwargs.get("chunk_overlap", 0)
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )
        return splitter.split_documents(documents)

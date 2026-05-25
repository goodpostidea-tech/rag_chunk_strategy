"""Recursive character splitter — FloTorch 2026 baseline."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunk_lab.base import BaseChunkStrategy


class RecursiveBaselineStrategy(BaseChunkStrategy):
    name = "recursive_baseline"
    category = "baseline"
    description = "递归字符分割（默认 512 字符 / 128 overlap），在自然分隔符处切分"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        size = kwargs.get("chunk_size", self.settings.baseline_chunk_size)
        overlap = kwargs.get("chunk_overlap", self.settings.baseline_chunk_overlap)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            add_start_index=True,
        )
        return splitter.split_documents(documents)

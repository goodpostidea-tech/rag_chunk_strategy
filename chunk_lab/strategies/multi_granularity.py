"""Cross-granularity indexing — same content at multiple chunk sizes."""

from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunk_lab.base import BaseChunkStrategy


class MultiGranularityStrategy(BaseChunkStrategy):
    name = "multi_granularity"
    category = "utility"
    description = "跨粒度索引：同一文档按多种 chunk_size 同时切分（用于对比检索粒度）"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        sizes_raw = kwargs.get(
            "sizes",
            self.settings.multi_granularity_sizes,
        )
        if isinstance(sizes_raw, str):
            sizes = [int(s.strip()) for s in sizes_raw.split(",") if s.strip()]
        else:
            sizes = list(sizes_raw)

        overlap_ratio = kwargs.get("overlap_ratio", 0.25)
        all_chunks: list[Document] = []

        for size in sizes:
            overlap = max(1, int(size * overlap_ratio))
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=overlap,
                add_start_index=True,
            )
            for ch in splitter.split_documents(documents):
                all_chunks.append(
                    Document(
                        page_content=ch.page_content,
                        metadata={
                            **ch.metadata,
                            "strategy": self.name,
                            "granularity_size": size,
                            "granularity_overlap": overlap,
                        },
                    )
                )
        return all_chunks

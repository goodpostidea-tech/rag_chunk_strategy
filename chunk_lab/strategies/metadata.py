"""Metadata-enriched chunking — prefix doc/section metadata to each chunk."""

from typing import Any

from langchain_core.documents import Document

from chunk_lab.base import BaseChunkStrategy
from chunk_lab.strategies.baseline import RecursiveBaselineStrategy
from chunk_lab.utils import doc_meta_from_document


def enrich_chunk_with_metadata(chunk: str, doc_meta: dict) -> str:
    prefix = (
        f"文档：{doc_meta.get('title') or 'N/A'} | "
        f"章节：{doc_meta.get('section') or 'N/A'} | "
        f"日期：{doc_meta.get('date') or 'N/A'} | "
        f"来源：{doc_meta.get('source') or 'N/A'}\n\n"
    )
    return prefix + chunk


class MetadataEnrichedStrategy(BaseChunkStrategy):
    name = "metadata_enriched"
    category = "utility"
    description = "递归基线分块 + 文档元数据前缀（Azure 推荐低成本优化）"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        base = RecursiveBaselineStrategy(self.settings)
        raw_chunks = base.chunk(documents, **kwargs)

        doc_meta_map: dict[str, dict] = {}
        for doc in documents:
            source = doc.metadata.get("source", "")
            if source not in doc_meta_map:
                doc_meta_map[source] = doc_meta_from_document(doc)

        enriched: list[Document] = []
        for ch in raw_chunks:
            source = ch.metadata.get("source", "")
            meta = doc_meta_map.get(source) or doc_meta_from_document(
                Document(page_content="", metadata=ch.metadata)
            )
            enriched.append(
                Document(
                    page_content=enrich_chunk_with_metadata(ch.page_content, meta),
                    metadata={**ch.metadata, "enriched": True},
                )
            )
        return enriched

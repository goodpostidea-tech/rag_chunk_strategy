"""Parent-child (small-to-big) chunking — child chunks with parent_id linkage."""

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunk_lab.base import BaseChunkStrategy


class ParentChildStrategy(BaseChunkStrategy):
    name = "parent_child"
    category = "C"
    description = "父子分块：小子块用于检索，metadata 中 parent_id 关联大父块（供 LLM 阅读）"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        child_size = kwargs.get("child_chunk_size", self.settings.child_chunk_size)
        child_overlap = kwargs.get("child_chunk_overlap", self.settings.child_chunk_overlap)
        parent_size = kwargs.get("parent_chunk_size", self.settings.parent_chunk_size)
        parent_overlap = kwargs.get("parent_chunk_overlap", self.settings.parent_chunk_overlap)

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            add_start_index=True,
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            add_start_index=True,
        )

        parent_chunks = parent_splitter.split_documents(documents)

        child_docs: list[Document] = []
        self._parent_store: dict[str, Document] = {}

        for parent in parent_chunks:
            parent_id = uuid.uuid4().hex[:12]
            parent.metadata["parent_id"] = parent_id
            self._parent_store[parent_id] = parent

            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent_id
                child.metadata["parent_content"] = parent.page_content
                child.metadata["strategy"] = self.name
                child.metadata["granularity"] = "child"
                child_docs.append(child)

        return child_docs

    def get_parent(self, parent_id: str) -> Document | None:
        """Retrieve parent document by id (available after chunk() call)."""
        return getattr(self, "_parent_store", {}).get(parent_id)

    def get_all_parents(self) -> dict[str, Document]:
        """Return all parent documents from the last chunk() call."""
        return getattr(self, "_parent_store", {})

"""Utilities: token counting, document loading."""

from pathlib import Path

import tiktoken
from langchain_core.documents import Document


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def load_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding="utf-8")


def make_documents(
    text: str,
    *,
    source: str = "inline",
    title: str = "",
    section: str = "",
) -> list[Document]:
    meta = {"source": source}
    if title:
        meta["title"] = title
    if section:
        meta["section"] = section
    return [Document(page_content=text, metadata=meta)]


def doc_meta_from_document(doc: Document) -> dict:
    return {
        "title": doc.metadata.get("title", ""),
        "section": doc.metadata.get("section", ""),
        "date": doc.metadata.get("date", ""),
        "source": doc.metadata.get("source", ""),
    }

"""Reference papers corpus & local directory datasources."""

from chunk_lab.datasources.corpus import (
    build_corpus_documents,
    fetch_paper,
    fetch_papers,
    get_paper,
    list_papers,
    load_cached_paper_text,
)
from chunk_lab.datasources.local_dir import (
    list_local_files,
    load_local_corpus_text,
    load_local_directory,
)
from chunk_lab.datasources.references_catalog import ReferencePaper, load_catalog

__all__ = [
    "ReferencePaper",
    "load_catalog",
    "list_papers",
    "get_paper",
    "fetch_paper",
    "fetch_papers",
    "load_cached_paper_text",
    "build_corpus_documents",
    "list_local_files",
    "load_local_directory",
    "load_local_corpus_text",
]

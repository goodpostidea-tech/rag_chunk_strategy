"""Corpus builder: fetch, cache, and load reference papers."""

import json
from pathlib import Path

from langchain_core.documents import Document

from chunk_lab.config import get_settings
from chunk_lab.datasources.arxiv_client import (
    build_paper_document_text,
    download_arxiv_pdf_text,
    fetch_arxiv_metadata,
    polite_delay,
)
from chunk_lab.datasources.references_catalog import ReferencePaper, get_by_id, load_catalog

META_SUFFIX = ".meta.json"
TEXT_SUFFIX = ".txt"


def papers_cache_dir() -> Path:
    cfg = get_settings()
    path = Path(cfg.papers_cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paper_paths(paper_id: str) -> tuple[Path, Path]:
    base = papers_cache_dir() / paper_id
    return base.with_suffix(TEXT_SUFFIX), base.with_suffix(META_SUFFIX)


def list_papers(*, arxiv_only: bool = False) -> list[dict]:
    papers = load_catalog()
    if arxiv_only:
        papers = [p for p in papers if p.fetchable_arxiv]
    result = []
    for p in papers:
        text_path, meta_path = _paper_paths(p.id)
        result.append(
            {
                "id": p.id,
                "ref_number": p.ref_number,
                "title": p.title,
                "category": p.category,
                "source_type": p.source_type,
                "arxiv_id": p.arxiv_id,
                "url": p.url,
                "related_strategies": p.related_strategies,
                "cached": text_path.exists(),
                "cached_meta": meta_path.exists(),
            }
        )
    return result


def get_paper(paper_id: str) -> ReferencePaper:
    return get_by_id(paper_id)


def _build_catalog_fallback_text(paper: ReferencePaper) -> str:
    authors = ", ".join(paper.authors) if paper.authors else "N/A"
    lines = [
        f"# {paper.title}",
        "",
        f"**Authors:** {authors}",
        f"**Category:** {paper.category} | **Source:** {paper.source_type}",
    ]
    if paper.published:
        lines.append(f"**Published:** {paper.published}")
    if paper.url:
        lines.append(f"**URL:** {paper.url}")
    if paper.arxiv_id:
        lines.append(f"**arXiv:** https://arxiv.org/abs/{paper.arxiv_id}")
    lines.extend(["", "## Summary", "", paper.summary])
    return "\n".join(lines)


def fetch_paper(
    paper_id: str,
    *,
    full_text: bool = False,
    force: bool = False,
) -> Path:
    """Download and cache one paper; returns path to .txt file."""
    paper = get_by_id(paper_id)
    text_path, meta_path = _paper_paths(paper.id)

    if text_path.exists() and meta_path.exists() and not force:
        return text_path

    meta: dict = {
        "id": paper.id,
        "title": paper.title,
        "source_type": paper.source_type,
        "arxiv_id": paper.arxiv_id,
        "url": paper.url,
        "full_text": full_text,
    }

    if paper.fetchable_arxiv:
        record = fetch_arxiv_metadata(paper.arxiv_id)  # type: ignore[arg-type]
        pdf_text = None
        if full_text:
            pdf_text = download_arxiv_pdf_text(paper.arxiv_id)  # type: ignore[arg-type]
        body = build_paper_document_text(
            record,
            include_full_text=full_text,
            full_text=pdf_text,
        )
        meta.update(
            {
                "fetched_from": "arxiv",
                "arxiv_title": record.title,
                "authors": record.authors,
                "published": record.published,
                "pdf_url": record.pdf_url,
                "char_count": len(body),
            }
        )
    else:
        body = _build_catalog_fallback_text(paper)
        meta.update({"fetched_from": "catalog_summary", "char_count": len(body)})

    text_path.write_text(body, encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return text_path


def fetch_papers(
    paper_ids: list[str] | None = None,
    *,
    all_arxiv: bool = False,
    full_text: bool = False,
    force: bool = False,
) -> list[dict]:
    """Batch fetch papers. Respects arXiv rate limit between PDF downloads."""
    if all_arxiv:
        targets = [p.id for p in load_catalog() if p.fetchable_arxiv]
    elif paper_ids:
        targets = paper_ids
    else:
        targets = [p.id for p in load_catalog()]

    results = []
    for i, pid in enumerate(targets):
        try:
            if i > 0:
                polite_delay(3.0 if full_text else 1.0)
            path = fetch_paper(pid, full_text=full_text, force=force)
            results.append({"id": pid, "status": "ok", "path": str(path)})
        except Exception as exc:
            results.append({"id": pid, "status": "error", "error": str(exc)})
    return results


def load_cached_paper_text(paper_id: str) -> str:
    text_path, _ = _paper_paths(paper_id)
    if not text_path.exists():
        raise FileNotFoundError(
            f"Paper '{paper_id}' not cached. Run: python main.py papers fetch --id {paper_id}"
        )
    return text_path.read_text(encoding="utf-8")


def paper_to_document(paper_id: str, *, fetch_if_missing: bool = True) -> Document:
    text_path, meta_path = _paper_paths(paper_id)
    if not text_path.exists() and fetch_if_missing:
        fetch_paper(paper_id, full_text=False)

    paper = get_by_id(paper_id)
    text = load_cached_paper_text(paper_id)
    metadata = {"paper_id": paper.id, "title": paper.title, "source": "references_catalog"}
    if meta_path.exists():
        metadata.update(json.loads(meta_path.read_text(encoding="utf-8")))
    return Document(page_content=text, metadata=metadata)


def build_corpus_documents(
    paper_ids: list[str] | None = None,
    *,
    fetch_if_missing: bool = True,
    combined: bool = False,
) -> list[Document]:
    """Load multiple papers as LangChain Documents."""
    ids = paper_ids or [p.id for p in load_catalog()]
    docs = [paper_to_document(pid, fetch_if_missing=fetch_if_missing) for pid in ids]

    if combined and docs:
        combined_text = "\n\n---\n\n".join(d.page_content for d in docs)
        return [
            Document(
                page_content=combined_text,
                metadata={"source": "references_corpus", "paper_count": len(docs)},
            )
        ]
    return docs


def load_corpus_text(
    paper_ids: list[str] | None = None,
    *,
    fetch_if_missing: bool = True,
) -> str:
    """Concatenate cached papers into one string for chunk/compare/eval."""
    docs = build_corpus_documents(paper_ids, fetch_if_missing=fetch_if_missing, combined=False)
    return "\n\n---\n\n".join(d.page_content for d in docs)

"""Fetch metadata and PDF text from arXiv."""

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from io import BytesIO

import httpx
from pypdf import PdfReader

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivRecord:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str
    abs_url: str


def _strip_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_arxiv_metadata(arxiv_id: str, *, timeout: float = 60.0) -> ArxivRecord:
    """Query arXiv Atom API for one paper."""
    params = {"id_list": arxiv_id.replace("arxiv:", "")}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(ARXIV_API, params=params)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        raise ValueError(f"No arXiv entry found for {arxiv_id}")

    title = _strip_text(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
    abstract = entry.findtext("atom:summary", default="", namespaces=ATOM_NS).strip()
    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)[:10]
    authors = [
        _strip_text(a.findtext("atom:name", default="", namespaces=ATOM_NS))
        for a in entry.findall("atom:author", ATOM_NS)
    ]
    aid = arxiv_id.replace("arxiv:", "")
    return ArxivRecord(
        arxiv_id=aid,
        title=title,
        authors=authors,
        abstract=abstract,
        published=published,
        pdf_url=ARXIV_PDF.format(arxiv_id=aid),
        abs_url=f"https://arxiv.org/abs/{aid}",
    )


def download_arxiv_pdf_text(arxiv_id: str, *, timeout: float = 120.0) -> str:
    """Download PDF and extract plain text."""
    aid = arxiv_id.replace("arxiv:", "")
    url = ARXIV_PDF.format(arxiv_id=aid)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        reader = PdfReader(BytesIO(resp.content))
        pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages).strip()


def build_paper_document_text(
    record: ArxivRecord,
    *,
    include_full_text: bool = False,
    full_text: str | None = None,
) -> str:
    """Format arXiv paper as a single document for chunking."""
    authors = ", ".join(record.authors) if record.authors else "N/A"
    parts = [
        f"# {record.title}",
        "",
        f"**Authors:** {authors}",
        f"**Published:** {record.published}",
        f"**arXiv:** {record.abs_url}",
        "",
        "## Abstract",
        "",
        record.abstract,
    ]
    if include_full_text and full_text:
        parts.extend(["", "## Full Text", "", full_text[:500_000]])
    return "\n".join(parts)


def polite_delay(seconds: float = 3.0) -> None:
    """arXiv recommends at most 1 request / 3 seconds."""
    time.sleep(seconds)

"""OCR engines for ocr_local and ocr_api parsers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from chunk_lab.config import Settings, get_settings
from parse_lab.ocr.api import parse_api_ocr
from parse_lab.ocr.config import DEFAULT_OCR_LANG
from parse_lab.ocr.local import parse_local_ocr
from parse_lab.parsers._utils import headings_from_markdown

_MD_TABLE_ROW = re.compile(r"^\|.+\|$", re.MULTILINE)


def _resolve_local_engine() -> str:
    try:
        from rapidocr import RapidOCR  # noqa: F401

        return "rapidocr"
    except ImportError:
        return "tesseract"


def parse_local_ocr_file(
    file_path: Path,
    *,
    settings: Settings | None = None,
) -> tuple[str, dict[str, Any]]:
    cfg = settings or get_settings()
    engine = _resolve_local_engine()
    page_texts, total_pages, truncated = parse_local_ocr(
        file_path,
        engine=engine,
        lang=DEFAULT_OCR_LANG,
        max_pages=cfg.parse_ocr_max_pages,
        dpi=cfg.parse_ocr_dpi,
    )
    ordered = [t for _, t in sorted(page_texts, key=lambda x: x[0]) if t]
    text = "\n\n".join(ordered)
    return text, {
        "engine": f"local:{engine}",
        "lang": DEFAULT_OCR_LANG,
        "pages_total": total_pages,
        "pages_parsed": len(page_texts),
        "pages_truncated": truncated,
        "dpi": cfg.parse_ocr_dpi,
        "max_pages_cap": cfg.parse_ocr_max_pages,
    }


def parse_api_ocr_file(
    file_path: Path,
    *,
    settings: Settings | None = None,
) -> tuple[str, dict[str, Any]]:
    cfg = settings or get_settings()
    page_texts, total_pages, truncated = parse_api_ocr(file_path, cfg)
    ordered = [t for _, t in sorted(page_texts, key=lambda x: x[0]) if t]
    text = "\n\n".join(ordered)
    return text, {
        "engine": f"api:{cfg.parse_ocr_api_provider}",
        "provider": cfg.parse_ocr_api_provider,
        "model": cfg.parse_ocr_api_model,
        "pages_total": total_pages,
        "pages_parsed": len(page_texts),
        "pages_truncated": truncated,
        "dpi": cfg.parse_ocr_dpi,
        "max_pages_cap": cfg.parse_ocr_max_pages,
    }


def build_parse_result_extra(text: str, extra: dict[str, Any]) -> tuple[list, list, dict]:
    headings = headings_from_markdown(text)
    table_rows = len(_MD_TABLE_ROW.findall(text))
    extra = {**extra, "heading_count": len(headings), "markdown_table_rows": table_rows}
    return headings, table_rows, extra

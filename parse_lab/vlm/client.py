"""Multimodal LLM client for per-page PDF extraction."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from chunk_lab.config import Settings, get_settings
from chunk_lab.models import get_llm
from chunk_lab.observability import ensure_observability_initialized, get_run_config
from parse_lab.parsers._utils import headings_from_markdown
from parse_lab.types import VlmParseConfig
from parse_lab.vlm.page_render import DEFAULT_MAX_PAGES, render_pdf_pages
from parse_lab.vlm.prompts import PAGE_EXTRACT_PROMPT

_PAGE_SEP = "\n\n---\n\n"
_MD_TABLE_ROW = re.compile(r"^\|.+\|$", re.MULTILINE)


def vlm_config_from_settings(settings: Settings | None = None) -> VlmParseConfig:
    """Build VLM parse config from system settings (parse_vlm_*)."""
    base = settings or get_settings()
    return VlmParseConfig(
        provider=base.parse_vlm_provider,
        model=base.parse_vlm_model,
        api_base=base.parse_vlm_api_base,
        api_key=base.parse_vlm_api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY"),
        max_pages=base.parse_vlm_max_pages,
        max_workers=base.parse_vlm_max_workers,
        temperature=base.parse_vlm_temperature,
        max_tokens=base.parse_vlm_max_tokens,
        dpi=base.parse_vlm_dpi,
        max_edge=base.parse_vlm_max_edge,
    )


def build_vlm_llm(cfg: VlmParseConfig, settings: Settings | None = None):
    overrides: dict[str, Any] = {
        "model": cfg.model,
        "model_provider": cfg.provider,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    if cfg.api_base:
        overrides["api_base"] = cfg.api_base
    if cfg.api_key:
        overrides["api_key"] = cfg.api_key
    return get_llm(settings, **overrides)


def _extract_page_text(llm, jpeg_bytes: bytes, page_no: int) -> str:
    from parse_lab.vlm.page_render import jpeg_to_data_url

    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"{PAGE_EXTRACT_PROMPT}\n\nPage number: {page_no}"},
            {"type": "image_url", "image_url": {"url": jpeg_to_data_url(jpeg_bytes)}},
        ]
    )
    ensure_observability_initialized()
    resp = llm.invoke(
        [msg],
        config=get_run_config(run_name="vlm_pdf_parse", tags=["parse", "vlm"]),
    )
    raw = resp.content
    if isinstance(raw, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in raw]
        text = "".join(parts).strip()
    else:
        text = str(raw).strip()
    # Strip accidental markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:markdown)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_pdf_with_vlm(
    file_path: Path,
    cfg: VlmParseConfig,
    *,
    settings: Settings | None = None,
) -> tuple[str, dict[str, Any]]:
    """Parse PDF via VLM page-by-page; returns markdown text and extra metadata."""
    resolved = cfg if isinstance(cfg, VlmParseConfig) else vlm_config_from_settings(settings)
    if not resolved.model.strip():
        raise ValueError("未配置视觉模型：请在「系统配置 → 视觉模型」中设置 parse_vlm_model")
    if not resolved.api_key and resolved.provider not in ("ollama",):
        raise ValueError(
            "vlm_pdf 需要 API Key：请在「系统配置 → 视觉模型」中设置 parse_vlm_api_key 或环境变量"
        )

    pages = render_pdf_pages(
        file_path,
        max_pages=resolved.max_pages,
        dpi=resolved.dpi,
        max_edge=resolved.max_edge,
    )
    if not pages:
        return "", {
            "engine": "vlm",
            "model": resolved.model,
            "provider": resolved.provider,
            "pages": 0,
            "pages_parsed": 0,
            "format": "markdown",
        }

    llm = build_vlm_llm(resolved, settings)
    workers = max(1, min(resolved.max_workers, len(pages)))
    page_texts: dict[int, str] = {}

    if workers == 1:
        for page_no, jpeg in pages:
            page_texts[page_no] = _extract_page_text(llm, jpeg, page_no)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_extract_page_text, llm, jpeg, page_no): page_no
                for page_no, jpeg in pages
            }
            for fut in as_completed(futures):
                page_no = futures[fut]
                page_texts[page_no] = fut.result()

    ordered = [page_texts[pno] for pno, _ in sorted(pages, key=lambda x: x[0]) if page_texts.get(pno)]
    text = _PAGE_SEP.join(t for t in ordered if t)

    import fitz

    doc = fitz.open(str(file_path))
    try:
        total_pages = doc.page_count
    finally:
        doc.close()

    truncated = total_pages > len(pages)

    headings = headings_from_markdown(text)
    table_rows = len(_MD_TABLE_ROW.findall(text))

    return text, {
        "engine": "vlm",
        "model": resolved.model,
        "provider": resolved.provider,
        "pages_total": total_pages,
        "pages_parsed": len(pages),
        "pages_truncated": truncated,
        "format": "markdown",
        "heading_count": len(headings),
        "markdown_table_rows": table_rows,
        "max_pages_cap": resolved.max_pages,
    }

"""Multimodal LLM OCR for providers without dedicated OCR REST API."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from chunk_lab.config import Settings
from chunk_lab.models import get_llm
from parse_lab.ocr.prompts import CLOUD_OCR_PAGE_PROMPT
from parse_lab.vlm.page_render import jpeg_to_data_url


def _provider_for_llm(settings: Settings) -> str:
    p = (settings.parse_ocr_api_provider or "openai").strip().lower()
    if p in ("alibaba", "dashscope"):
        return "dashscope"
    if p == "qwen":
        return "qwen"
    if p == "google":
        return "google_genai"
    return p


def ocr_page(jpeg_bytes: bytes, page_no: int, settings: Settings) -> str:
    overrides: dict[str, Any] = {
        "model": settings.parse_ocr_api_model or "gpt-4o",
        "model_provider": _provider_for_llm(settings),
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if settings.parse_ocr_api_base:
        overrides["api_base"] = settings.parse_ocr_api_base
    if settings.parse_ocr_api_key:
        overrides["api_key"] = settings.parse_ocr_api_key
    llm = get_llm(settings, **overrides)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"{CLOUD_OCR_PAGE_PROMPT}\n\nPage: {page_no}"},
            {"type": "image_url", "image_url": {"url": jpeg_to_data_url(jpeg_bytes)}},
        ]
    )
    resp = llm.invoke([msg])
    raw = resp.content
    if isinstance(raw, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
        ).strip()
    return str(raw).strip()

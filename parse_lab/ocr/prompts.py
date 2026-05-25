"""Prompts for cloud OCR (multimodal API path)."""

CLOUD_OCR_PAGE_PROMPT = """You are an OCR engine. Extract ALL visible text from this document page image.

Rules:
- Output plain text only (no markdown fences, no commentary).
- Preserve reading order.
- Do not invent content."""

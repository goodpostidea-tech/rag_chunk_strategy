"""Online OCR via configured API provider (ocr_api parser)."""

from __future__ import annotations

from pathlib import Path

from chunk_lab.config import Settings, get_settings
from parse_lab.ocr.providers.router import ocr_api_page
from parse_lab.vlm.page_render import render_pdf_pages

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def parse_api_ocr(
    file_path: Path,
    settings: Settings | None = None,
) -> tuple[list[tuple[int, str]], int, bool]:
    cfg = settings or get_settings()
    max_pages = cfg.parse_ocr_max_pages
    dpi = cfg.parse_ocr_dpi
    prov = (cfg.parse_ocr_api_provider or "").strip().lower()
    ext = file_path.suffix.lower()

    def _run_page(jpeg: bytes, page_no: int) -> str:
        return ocr_api_page(jpeg, page_no, cfg)

    if ext == ".pdf":
        cap = max_pages if max_pages > 0 else 0
        pages_img = render_pdf_pages(file_path, max_pages=cap or None, dpi=dpi, max_edge=2048)
        page_texts = [(pno, _run_page(jpeg, pno)) for pno, jpeg in pages_img]
        import fitz

        doc = fitz.open(str(file_path))
        try:
            total = doc.page_count
        finally:
            doc.close()
        return page_texts, total, total > len(pages_img)

    if ext in _IMAGE_EXTS:
        data = file_path.read_bytes()
        return [(1, _run_page(data, 1))], 1, False

    raise ValueError(f"OCR API 不支持扩展名: {ext}")

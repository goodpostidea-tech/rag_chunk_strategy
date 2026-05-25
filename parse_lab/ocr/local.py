"""Local OCR engines (no cloud API)."""

from __future__ import annotations

import io
from pathlib import Path

from parse_lab.vlm.page_render import render_pdf_pages

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def ocr_tesseract_image_bytes(jpeg_bytes: bytes, *, lang: str) -> str:
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(jpeg_bytes))
    return pytesseract.image_to_string(img, lang=lang).strip()


def ocr_tesseract_file(path: Path, *, lang: str) -> str:
    import pytesseract

    return pytesseract.image_to_string(str(path), lang=lang).strip()


def ocr_rapidocr_image_bytes(jpeg_bytes: bytes) -> str:
    from rapidocr import RapidOCR

    engine = RapidOCR()
    result, _ = engine(jpeg_bytes)
    if not result:
        return ""
    return "\n".join(str(line[1]) for line in result if len(line) > 1).strip()


def ocr_rapidocr_file(path: Path) -> str:
    from rapidocr import RapidOCR

    engine = RapidOCR()
    result, _ = engine(str(path))
    if not result:
        return ""
    return "\n".join(str(line[1]) for line in result if len(line) > 1).strip()


def parse_local_ocr(
    file_path: Path,
    *,
    engine: str,
    lang: str,
    max_pages: int,
    dpi: int,
) -> tuple[list[tuple[int, str]], int, bool]:
    """Return (page_no, text) list, total_pages, truncated."""
    ext = file_path.suffix.lower()
    eng = engine.strip().lower()

    if ext == ".pdf":
        cap = max_pages if max_pages > 0 else 0
        pages_img = render_pdf_pages(file_path, max_pages=cap or None, dpi=dpi, max_edge=2048)
        page_texts: list[tuple[int, str]] = []
        for page_no, jpeg in pages_img:
            if eng == "rapidocr":
                text = ocr_rapidocr_image_bytes(jpeg)
            else:
                text = ocr_tesseract_image_bytes(jpeg, lang=lang)
            page_texts.append((page_no, text))
        total = _pdf_page_count(file_path)
        return page_texts, total, total > len(pages_img)

    if ext in _IMAGE_EXTS:
        if eng == "rapidocr":
            text = ocr_rapidocr_file(file_path)
        else:
            text = ocr_tesseract_file(file_path, lang=lang)
        return [(1, text)], 1, False

    raise ValueError(f"本地 OCR 不支持扩展名: {ext}")


def _pdf_page_count(file_path: Path) -> int:
    import fitz

    doc = fitz.open(str(file_path))
    try:
        return doc.page_count
    finally:
        doc.close()

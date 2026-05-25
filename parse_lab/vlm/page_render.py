"""Render PDF pages to JPEG for VLM input."""

from __future__ import annotations

import base64
from pathlib import Path

# Default cap to control cost/latency
DEFAULT_MAX_PAGES = 50
DEFAULT_DPI = 150
DEFAULT_MAX_EDGE = 1280


def render_pdf_pages(
    file_path: Path,
    *,
    max_pages: int | None = None,
    dpi: int = DEFAULT_DPI,
    max_edge: int = DEFAULT_MAX_EDGE,
) -> list[tuple[int, bytes]]:
    """Return list of (page_number_1based, jpeg_bytes)."""
    import fitz

    doc = fitz.open(str(file_path))
    try:
        total = doc.page_count
        if total == 0:
            return []

        limit = max_pages if max_pages is not None else DEFAULT_MAX_PAGES
        if limit <= 0:
            page_indices = list(range(total))
        else:
            page_indices = _sample_page_indices(total, limit)

        out: list[tuple[int, bytes]] = []

        for idx in page_indices:
            page = doc.load_page(idx)
            zoom = dpi / 72.0
            rect = page.rect
            w, h = rect.width * zoom, rect.height * zoom
            if max(w, h) > max_edge:
                zoom *= max_edge / max(w, h)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            jpeg = pix.tobytes("jpeg", jpg_quality=85)
            out.append((idx + 1, jpeg))
        return out
    finally:
        doc.close()


def _sample_page_indices(total: int, limit: int) -> list[int]:
    """First, last, and evenly spaced middle pages when truncating."""
    if total <= limit:
        return list(range(total))
    if limit <= 2:
        return [0, total - 1][:limit]
    indices = {0, total - 1}
    step = (total - 1) / (limit - 1)
    for i in range(1, limit - 1):
        indices.add(int(round(i * step)))
    return sorted(indices)


def jpeg_to_data_url(jpeg_bytes: bytes) -> str:
    b64 = base64.standard_b64encode(jpeg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

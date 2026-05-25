"""Vision-language model document parsers."""

import time
from pathlib import Path

from parse_lab.base import BaseDocumentParser
from parse_lab.parsers._utils import headings_from_markdown
from parse_lab.types import ParsedTable, ParseResult
from parse_lab.vlm.client import parse_pdf_with_vlm
from parse_lab.vlm.context import get_vlm_config


class VlmPdfParser(BaseDocumentParser):
    name = "vlm_pdf"
    description = "视觉模型按页 OCR → Markdown（需多模态 API，较慢）"
    file_types = (".pdf",)
    optional = True

    def parse(self, file_path: Path) -> ParseResult:
        t0 = time.perf_counter()
        cfg = get_vlm_config()
        if cfg is None:
            elapsed = (time.perf_counter() - t0) * 1000
            return ParseResult(
                parser=self.name,
                file_path=str(file_path),
                file_type=file_path.suffix.lower(),
                text="",
                char_count=0,
                table_count=0,
                heading_count=0,
                elapsed_ms=round(elapsed, 2),
                error="vlm_pdf 未配置：请先在「系统配置 → 视觉模型」中设置提供商、模型与 API Key",
            )
        try:
            text, extra = parse_pdf_with_vlm(file_path, cfg)
            headings = headings_from_markdown(text)
            table_rows = int(extra.get("markdown_table_rows", 0))
            tables = [
                ParsedTable(rows=[["markdown_table"]], source="vlm_markdown")
                for _ in range(table_rows)
            ]
            elapsed = (time.perf_counter() - t0) * 1000
            return ParseResult(
                parser=self.name,
                file_path=str(file_path),
                file_type=file_path.suffix.lower(),
                text=text,
                char_count=len(text),
                table_count=len(tables),
                heading_count=len(headings),
                tables=tables,
                headings=headings,
                elapsed_ms=round(elapsed, 2),
                extra=extra,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return ParseResult(
                parser=self.name,
                file_path=str(file_path),
                file_type=file_path.suffix.lower(),
                text="",
                char_count=0,
                table_count=0,
                heading_count=0,
                elapsed_ms=round(elapsed, 2),
                error=str(exc),
            )

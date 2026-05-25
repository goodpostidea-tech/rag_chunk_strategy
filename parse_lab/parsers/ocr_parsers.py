"""OCR parsers: ocr_local (本地包) 与 ocr_api (在线服务商)."""

import re
import time
from pathlib import Path

from parse_lab.base import BaseDocumentParser
from parse_lab.ocr.engine import (
    build_parse_result_extra,
    parse_api_ocr_file,
    parse_local_ocr_file,
)
from parse_lab.ocr.config import ocr_api_available
from parse_lab.types import ParsedTable, ParseResult

_MD_TABLE_ROW = re.compile(r"^\|.+\|$", re.MULTILINE)


def _wrap_result(
    parser_name: str,
    file_path: Path,
    text: str,
    extra: dict,
    elapsed_ms: float,
    error: str | None = None,
) -> ParseResult:
    if error:
        return ParseResult(
            parser=parser_name,
            file_path=str(file_path),
            file_type=file_path.suffix.lower(),
            text="",
            char_count=0,
            table_count=0,
            heading_count=0,
            elapsed_ms=elapsed_ms,
            error=error,
        )
    headings, table_rows, extra = build_parse_result_extra(text, extra)
    tables = [ParsedTable(rows=[["ocr"]], source=parser_name) for _ in range(table_rows)]
    return ParseResult(
        parser=parser_name,
        file_path=str(file_path),
        file_type=file_path.suffix.lower(),
        text=text,
        char_count=len(text),
        table_count=len(tables),
        heading_count=len(headings),
        tables=tables,
        headings=headings,
        elapsed_ms=elapsed_ms,
        extra=extra,
    )


class OcrLocalParser(BaseDocumentParser):
    name = "ocr_local"
    description = "本地 OCR（RapidOCR / Tesseract，无需 API）"
    file_types = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    optional = True

    def parse(self, file_path: Path) -> ParseResult:
        t0 = time.perf_counter()
        try:
            text, extra = parse_local_ocr_file(file_path)
            elapsed = (time.perf_counter() - t0) * 1000
            return _wrap_result(self.name, file_path, text, extra, round(elapsed, 2))
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return _wrap_result(self.name, file_path, "", {}, round(elapsed, 2), str(exc))


class OcrApiParser(BaseDocumentParser):
    name = "ocr_api"
    description = "OCR API（Azure / 百度 / 腾讯 / 阿里等，见系统配置 → OCR API）"
    file_types = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
    optional = True

    def parse(self, file_path: Path) -> ParseResult:
        t0 = time.perf_counter()
        ok, hint = ocr_api_available()
        if not ok:
            elapsed = (time.perf_counter() - t0) * 1000
            return _wrap_result(
                self.name,
                file_path,
                "",
                {},
                round(elapsed, 2),
                hint or "请先在「系统配置 → OCR API」中配置服务商",
            )
        try:
            text, extra = parse_api_ocr_file(file_path)
            elapsed = (time.perf_counter() - t0) * 1000
            return _wrap_result(self.name, file_path, text, extra, round(elapsed, 2))
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return _wrap_result(self.name, file_path, "", {}, round(elapsed, 2), str(exc))

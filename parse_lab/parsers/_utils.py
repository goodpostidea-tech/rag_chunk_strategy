"""Shared helpers for document parsers."""

import logging
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from parse_lab.types import ParsedHeading, ParsedTable, ParseResult

# pdfplumber → pdfminer：部分 PDF 的 /FontBBox 缺失，会打无害 warning 刷屏
_PDF_FONT_NOISE = "FontBBox from font descriptor"


class _PdfFontNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return _PDF_FONT_NOISE not in record.getMessage()


def _install_pdf_log_filters() -> None:
    if getattr(_install_pdf_log_filters, "_done", False):
        return
    flt = _PdfFontNoiseFilter()
    for name in ("pdfminer", "pdfminer.pdffont", "pdfplumber"):
        logging.getLogger(name).addFilter(flt)
    _install_pdf_log_filters._done = True  # type: ignore[attr-defined]


_install_pdf_log_filters()


@contextmanager
def _quiet_pdf_font_warnings():
    """临时抬高 pdfminer 日志级别（解析单文件期间）。"""
    loggers = [logging.getLogger(n) for n in ("pdfminer", "pdfminer.pdffont")]
    old = [(lg, lg.level) for lg in loggers]
    try:
        for lg in loggers:
            lg.setLevel(logging.ERROR)
        yield
    finally:
        for lg, level in old:
            lg.setLevel(level)


def run_timed(parser_name: str, file_path: Path, fn) -> ParseResult:
    """Execute parse fn and wrap timing + errors."""
    t0 = time.perf_counter()
    quiet = (
        _quiet_pdf_font_warnings()
        if file_path.suffix.lower() == ".pdf"
        else nullcontext()
    )
    with quiet:
        try:
            text, tables, headings, extra = fn()
            elapsed = (time.perf_counter() - t0) * 1000
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
                elapsed_ms=round(elapsed, 2),
                extra=extra,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return ParseResult(
                parser=parser_name,
                file_path=str(file_path),
                file_type=file_path.suffix.lower(),
                text="",
                char_count=0,
                table_count=0,
                heading_count=0,
                elapsed_ms=round(elapsed, 2),
                error=str(exc),
            )


def headings_from_markdown(text: str) -> list[ParsedHeading]:
    import re

    out: list[ParsedHeading] = []
    for m in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        out.append(ParsedHeading(level=len(m.group(1)), text=m.group(2).strip()))
    return out

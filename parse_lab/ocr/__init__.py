"""OCR parsing — local (ocr_local) and API (ocr_api)."""

from parse_lab.ocr.engine import parse_api_ocr_file, parse_local_ocr_file

__all__ = ["parse_local_ocr_file", "parse_api_ocr_file"]

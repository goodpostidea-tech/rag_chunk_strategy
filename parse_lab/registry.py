"""Parser registry with optional dependency handling."""

from parse_lab.base import BaseDocumentParser
from parse_lab.parsers import (
    DoclingParser,
    Docx2txtParser,
    MammothParser,
    OcrApiParser,
    OcrLocalParser,
    PdfPlumberParser,
    PyMuPdf4LlmParser,
    PyMuPdfParser,
    PypdfParser,
    PythonDocxParser,
    VlmPdfParser,
)

PARSER_REGISTRY: dict[str, type[BaseDocumentParser]] = {
    PypdfParser.name: PypdfParser,
    PyMuPdfParser.name: PyMuPdfParser,
    PdfPlumberParser.name: PdfPlumberParser,
    PyMuPdf4LlmParser.name: PyMuPdf4LlmParser,
    DoclingParser.name: DoclingParser,
    Docx2txtParser.name: Docx2txtParser,
    PythonDocxParser.name: PythonDocxParser,
    MammothParser.name: MammothParser,
    VlmPdfParser.name: VlmPdfParser,
    OcrLocalParser.name: OcrLocalParser,
    OcrApiParser.name: OcrApiParser,
}

_PARSER_ALIASES = {
    "tesseract_ocr": "ocr_local",
    "ocr": "ocr_api",
}


def _check_optional_available(cls: type[BaseDocumentParser]) -> tuple[bool, str | None]:
    if not getattr(cls, "optional", False):
        return True, None
    if cls.name == "pymupdf4llm":
        try:
            import pymupdf4llm  # noqa: F401
            return True, None
        except ImportError:
            return False, "pip install pymupdf4llm"
    if cls.name == "docling":
        try:
            import docling  # noqa: F401
            return True, None
        except ImportError:
            return False, "pip install docling"
    if cls.name == "mammoth":
        try:
            import mammoth  # noqa: F401
            return True, None
        except ImportError:
            return False, "pip install mammoth"
    if cls.name == "vlm_pdf":
        try:
            import fitz  # noqa: F401
            return True, None
        except ImportError:
            return False, "pip install pymupdf"
    if cls.name == "ocr_local":
        from parse_lab.ocr.config import ocr_local_available

        return ocr_local_available()
    if cls.name == "ocr_api":
        from parse_lab.ocr.config import ocr_api_available

        return ocr_api_available()
    return True, None


def list_parsers(*, include_unavailable: bool = False) -> list[dict]:
    rows: list[dict] = []
    for cls in PARSER_REGISTRY.values():
        available, hint = _check_optional_available(cls)
        if not available and not include_unavailable:
            continue
        rows.append(
            {
                "name": cls.name,
                "description": cls.description,
                "file_types": list(cls.file_types),
                "optional": cls.optional,
                "available": available,
                "install_hint": hint,
            }
        )
    return rows


def get_parser(name: str) -> BaseDocumentParser:
    key = name.strip().lower()
    key = _PARSER_ALIASES.get(key, key)
    if key not in PARSER_REGISTRY:
        available = ", ".join(sorted(PARSER_REGISTRY))
        raise ValueError(f"Unknown parser '{name}'. Available: {available}")
    cls = PARSER_REGISTRY[key]
    ok, hint = _check_optional_available(cls)
    if not ok:
        raise ImportError(f"Parser '{name}' requires extra dependency: {hint}")
    return cls()


def parsers_for_file(ext: str) -> list[str]:
    ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    return [cls.name for cls in PARSER_REGISTRY.values() if ext in cls.file_types]

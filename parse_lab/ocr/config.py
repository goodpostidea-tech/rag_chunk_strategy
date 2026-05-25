"""OCR availability checks for ocr_local and ocr_api parsers."""

from __future__ import annotations

from chunk_lab.config import Settings, get_settings
from parse_lab.ocr.providers.router import ocr_api_provider_available

# 本地 OCR 默认：优先 RapidOCR，否则 Tesseract
DEFAULT_OCR_LANG = "chi_sim+eng"


def ocr_local_available(settings: Settings | None = None) -> tuple[bool, str | None]:
    """ocr_local: RapidOCR 或 Tesseract 任一可用即可。"""
    try:
        from rapidocr import RapidOCR  # noqa: F401

        return True, None
    except ImportError:
        pass
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True, None
    except ImportError:
        return False, "请安装: uv pip install rapidocr 或 pytesseract pillow"
    except Exception:
        return (
            False,
            "请安装 Tesseract-OCR（PATH）或 uv pip install rapidocr",
        )


def ocr_api_available(settings: Settings | None = None) -> tuple[bool, str | None]:
    """ocr_api: 系统配置中的 OCR API 服务商凭证。"""
    return ocr_api_provider_available(settings or get_settings())

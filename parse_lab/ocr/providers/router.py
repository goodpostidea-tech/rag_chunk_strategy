"""Route page OCR to the configured API provider."""

from __future__ import annotations

from chunk_lab.config import Settings

from parse_lab.ocr.providers import azure, baidu, multimodal, tencent

# 主流 OCR API 提供商
OCR_API_PROVIDERS: dict[str, str] = {
    "azure": "Microsoft Azure AI Vision (Read API)",
    "baidu": "百度智能云 OCR",
    "tencent": "腾讯云 OCR",
    "alibaba": "阿里云 OCR（DashScope 多模态）",
    "google": "Google Cloud Vision（多模态接口）",
    "huawei": "华为云 OCR（预留）",
    "openai": "OpenAI Vision",
    "dashscope": "DashScope 多模态 OCR",
    "qwen": "通义千问多模态 OCR",
    "openai_compatible": "OpenAI 兼容多模态 OCR",
}

_MULTIMODAL = frozenset(
    {"alibaba", "google", "openai", "dashscope", "qwen", "openai_compatible"}
)


def ocr_api_provider_available(settings: Settings) -> tuple[bool, str | None]:
    prov = (settings.parse_ocr_api_provider or "").strip().lower()
    if not prov:
        return False, "请选择 OCR API 提供商"
    if prov not in OCR_API_PROVIDERS:
        return False, f"不支持的 OCR 提供商: {prov}"

    key = settings.parse_ocr_api_key or ""
    if prov == "baidu":
        if not key or not (settings.parse_ocr_api_secret or "").strip():
            return False, "百度 OCR 需配置 API Key 与 Secret Key"
        return True, None
    if prov == "tencent":
        if not key or not (settings.parse_ocr_api_secret or "").strip():
            return False, "腾讯云 OCR 需配置 SecretId 与 SecretKey"
        return True, None
    if prov == "azure":
        if not key:
            return False, "请配置 OCR API Key"
        if not (settings.parse_ocr_api_base or "").strip():
            return False, "Azure OCR 需填写 API Base（认知服务终结点）"
        return True, None
    if prov == "huawei":
        return False, "华为云 OCR 尚未接入，请换用其他提供商"
    if prov in _MULTIMODAL:
        if not key:
            return False, "请配置 OCR API Key"
        if prov not in ("azure", "google") and not (settings.parse_ocr_api_model or "").strip():
            return False, "请填写 OCR API 模型 ID"
        return True, None
    if not key:
        return False, "请配置 OCR API Key"
    return True, None


def ocr_api_page(jpeg_bytes: bytes, page_no: int, settings: Settings) -> str:
    prov = (settings.parse_ocr_api_provider or "azure").strip().lower()
    if prov == "azure":
        return azure.ocr_page(jpeg_bytes, settings)
    if prov == "baidu":
        return baidu.ocr_page(jpeg_bytes, settings)
    if prov == "tencent":
        return tencent.ocr_page(jpeg_bytes, settings)
    if prov in _MULTIMODAL:
        return multimodal.ocr_page(jpeg_bytes, page_no, settings)
    raise ValueError(f"不支持的 OCR API 提供商: {prov}")

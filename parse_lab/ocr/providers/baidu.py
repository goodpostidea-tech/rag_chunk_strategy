"""Baidu Cloud OCR API."""

from __future__ import annotations

import base64

import httpx

from chunk_lab.config import Settings

_token_cache: dict[str, str] = {}


def _access_token(settings: Settings) -> str:
    api_key = settings.parse_ocr_api_key or ""
    secret = settings.parse_ocr_api_secret or ""
    cache_key = f"{api_key}:{secret}"
    if cache_key in _token_cache:
        return _token_cache[cache_key]
    url = (
        "https://aip.baidubce.com/oauth/2.0/token"
        f"?grant_type=client_credentials&client_id={api_key}&client_secret={secret}"
    )
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url)
        resp.raise_for_status()
        data = resp.json()
    token = data.get("access_token", "")
    if not token:
        raise RuntimeError(f"百度 OCR 获取 token 失败: {data}")
    _token_cache[cache_key] = token
    return token


def ocr_page(jpeg_bytes: bytes, settings: Settings) -> str:
    token = _access_token(settings)
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={token}"
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            url,
            data={"image": b64},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        data = resp.json()
    if "error_code" in data:
        raise RuntimeError(f"百度 OCR 错误: {data}")
    lines = [item.get("words", "") for item in data.get("words_result", [])]
    return "\n".join(t for t in lines if t).strip()

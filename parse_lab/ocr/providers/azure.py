"""Azure AI Vision Read API."""

from __future__ import annotations

import time

import httpx

from chunk_lab.config import Settings


def ocr_page(jpeg_bytes: bytes, settings: Settings) -> str:
    api_key = settings.parse_ocr_api_key or ""
    base = (settings.parse_ocr_api_base or "").strip()
    if not base:
        raise ValueError("Azure OCR 需填写 API Base（xxx.cognitiveservices.azure.com）")
    if not base.startswith("http"):
        base = f"https://{base}"
    analyze_url = f"{base.rstrip('/')}/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Content-Type": "application/octet-stream",
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(analyze_url, headers=headers, content=jpeg_bytes)
        resp.raise_for_status()
        op_url = resp.headers.get("Operation-Location")
        if not op_url:
            raise RuntimeError("Azure OCR 未返回 Operation-Location")
        for _ in range(60):
            time.sleep(1.0)
            poll = client.get(op_url, headers={"Ocp-Apim-Subscription-Key": api_key})
            poll.raise_for_status()
            data = poll.json()
            status = data.get("status", "")
            if status == "succeeded":
                lines: list[str] = []
                for block in data.get("analyzeResult", {}).get("readResults", []):
                    for ln in block.get("lines", []):
                        t = ln.get("text", "").strip()
                        if t:
                            lines.append(t)
                return "\n".join(lines)
            if status == "failed":
                raise RuntimeError(f"Azure OCR 失败: {data}")
        raise TimeoutError("Azure OCR 轮询超时")

"""Tencent Cloud OCR API (GeneralBasicOCR, TC3-HMAC-SHA256)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

import httpx

from chunk_lab.config import Settings

_SERVICE = "ocr"
_HOST = "ocr.tencentcloudapi.com"
_ACTION = "GeneralBasicOCR"
_VERSION = "2018-11-19"


def _tc3_sign(secret_key: str, date: str, payload: str, timestamp: int) -> str:
    def _h(key: bytes | str, msg: str) -> bytes:
        k = key.encode("utf-8") if isinstance(key, str) else key
        return hmac.new(k, msg.encode("utf-8"), hashlib.sha256).digest()

    hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = (
        "POST\n/\n\n"
        "content-type:application/json\n"
        f"host:{_HOST}\n\n"
        "content-type;host\n"
        f"{hashed_payload}"
    )
    credential_scope = f"{date}/{_SERVICE}/tc3_request"
    string_to_sign = (
        "TC3-HMAC-SHA256\n"
        f"{timestamp}\n"
        f"{credential_scope}\n"
        + hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    )
    secret_date = _h(f"TC3{secret_key}", date)
    secret_service = _h(secret_date, _SERVICE)
    secret_signing = _h(secret_service, "tc3_request")
    return hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def ocr_page(jpeg_bytes: bytes, settings: Settings) -> str:
    secret_id = settings.parse_ocr_api_key or ""
    secret_key = settings.parse_ocr_api_secret or ""
    region = (settings.parse_ocr_api_region or "ap-guangzhou").strip()
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    body = json.dumps({"ImageBase64": b64}, separators=(",", ":"))
    timestamp = int(time.time())
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    signature = _tc3_sign(secret_key, date, body, timestamp)
    auth = (
        "TC3-HMAC-SHA256 "
        f"Credential={secret_id}/{date}/{_SERVICE}/tc3_request, "
        "SignedHeaders=content-type;host, "
        f"Signature={signature}"
    )
    headers = {
        "Authorization": auth,
        "Content-Type": "application/json",
        "Host": _HOST,
        "X-TC-Action": _ACTION,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Version": _VERSION,
        "X-TC-Region": region,
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"https://{_HOST}", headers=headers, content=body)
        resp.raise_for_status()
        data = resp.json()
    resp_body = data.get("Response", {})
    if "Error" in resp_body:
        raise RuntimeError(f"腾讯云 OCR 错误: {resp_body['Error']}")
    lines = [item.get("DetectedText", "") for item in resp_body.get("TextDetections", [])]
    return "\n".join(t for t in lines if t).strip()

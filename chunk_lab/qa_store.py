"""Persist user QA datasets and load builtin/custom pairs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chunk_lab.types import EvalQAPair

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_BUILTIN_QA = _DATA_DIR / "papers_qa.json"
_CUSTOM_QA = _DATA_DIR / "custom_qa.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_builtin_qa_pairs() -> list[EvalQAPair]:
    raw = _read_json(_BUILTIN_QA)
    return [_normalize_pair(item) for item in raw]


def _normalize_pair(item: dict) -> EvalQAPair:
    return EvalQAPair.from_dict(item)


def custom_qa_exists() -> bool:
    return _CUSTOM_QA.is_file()


def load_custom_qa_meta() -> dict[str, Any] | None:
    if not custom_qa_exists():
        return None
    data = _read_json(_CUSTOM_QA)
    if isinstance(data, list):
        return {
            "name": "用户自定义",
            "source": "custom",
            "updated_at": None,
            "qa_pairs": [_normalize_pair(item).model_dump() for item in data],
        }
    pairs = [_normalize_pair(p) for p in (data.get("qa_pairs") or [])]
    return {
        "name": data.get("name", "用户自定义"),
        "source": "custom",
        "updated_at": data.get("updated_at"),
        "qa_pairs": [p.model_dump() for p in pairs],
    }


def load_custom_qa_pairs() -> list[EvalQAPair]:
    meta = load_custom_qa_meta()
    if not meta:
        return []
    return [EvalQAPair(**p) for p in meta["qa_pairs"]]


def load_active_qa_pairs(*, prefer_custom: bool = True) -> tuple[list[EvalQAPair], str]:
    """Return (pairs, source) where source is 'custom' or 'builtin'."""
    if prefer_custom and custom_qa_exists():
        return load_custom_qa_pairs(), "custom"
    return load_builtin_qa_pairs(), "builtin"


def save_custom_qa_dataset(
    qa_pairs: list[EvalQAPair],
    *,
    name: str = "用户自定义",
) -> dict[str, Any]:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "source": "custom",
        "updated_at": _utc_now(),
        "qa_pairs": [p.model_dump() for p in qa_pairs],
    }
    _CUSTOM_QA.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def reset_custom_qa_dataset() -> None:
    if _CUSTOM_QA.is_file():
        _CUSTOM_QA.unlink()


def resolve_qa_pairs_by_source(source: str) -> tuple[list[EvalQAPair], str]:
    src = (source or "active").lower()
    if src == "builtin":
        return load_builtin_qa_pairs(), "builtin"
    if src == "custom":
        pairs = load_custom_qa_pairs()
        if not pairs:
            raise ValueError("尚未保存用户自定义 QA 测试集")
        return pairs, "custom"
    return load_active_qa_pairs()


def dataset_summary() -> dict[str, Any]:
    builtin = load_builtin_qa_pairs()
    custom_meta = load_custom_qa_meta()
    active_pairs, active_source = load_active_qa_pairs()
    return {
        "active_source": active_source,
        "active_count": len(active_pairs),
        "builtin_count": len(builtin),
        "custom_count": len(custom_meta["qa_pairs"]) if custom_meta else 0,
        "custom_updated_at": custom_meta.get("updated_at") if custom_meta else None,
        "custom_name": custom_meta.get("name") if custom_meta else None,
    }

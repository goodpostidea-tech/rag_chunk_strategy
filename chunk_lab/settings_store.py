"""Runtime settings overlay — DB-backed, per-workspace."""

from typing import Any

from chunk_lab.config import Settings

_cache: dict[str, dict[str, Any]] = {}


def _load_overlay(workspace_id: str = "default") -> dict[str, Any]:
    if workspace_id in _cache:
        return _cache[workspace_id]
    from chunk_lab.db import load_settings_overlay
    overlay = load_settings_overlay(workspace_id)
    _cache[workspace_id] = overlay
    return overlay


def get_effective_settings(workspace_id: str = "default") -> Settings:
    overlay = _load_overlay(workspace_id)
    base = Settings()
    if not overlay:
        return base
    merged = {**base.model_dump(), **overlay}
    return Settings(**merged)


def get_overlay(workspace_id: str = "default") -> dict[str, Any]:
    return dict(_load_overlay(workspace_id))


def _is_masked(value: Any) -> bool:
    return isinstance(value, str) and "••••" in value


def update_overlay(partial: dict[str, Any], workspace_id: str = "default") -> Settings:
    """Merge partial updates; skip masked secrets; empty clears optional fields."""
    from chunk_lab.db import delete_settings_key, save_settings_key

    nullable_suffixes = (
        "_api_key", "_secret_key", "_public_key", "_password", "_token",
        "_api_base", "_host", "_url", "_uri",
    )
    overlay = _load_overlay(workspace_id)
    for key, value in partial.items():
        if key in SECRET_KEYS and _is_masked(value):
            continue
        if value is None:
            overlay.pop(key, None)
            delete_settings_key(workspace_id, key)
        elif value == "" and key.endswith(nullable_suffixes):
            overlay[key] = None
            save_settings_key(workspace_id, key, None)
        else:
            overlay[key] = value
            save_settings_key(workspace_id, key, value)
    _cache[workspace_id] = overlay
    return get_effective_settings(workspace_id)


def reset_overlay(workspace_id: str = "default") -> Settings:
    from chunk_lab.db import clear_settings

    clear_settings(workspace_id)
    _cache.pop(workspace_id, None)
    return get_effective_settings(workspace_id)


def invalidate_cache(workspace_id: str | None = None) -> None:
    if workspace_id:
        _cache.pop(workspace_id, None)
    else:
        _cache.clear()


def _mask_secret(value: Any) -> Any:
    if value is None or value == "":
        return None
    s = str(value)
    if len(s) <= 8:
        return "••••••••"
    return s[:4] + "••••" + s[-4:]


SECRET_KEYS = {
    "chunk_llm_api_key",
    "chunk_embedding_api_key",
    "qdrant_api_key",
    "milvus_password",
    "milvus_token",
    "pinecone_api_key",
    "langsmith_api_key",
    "langfuse_public_key",
    "langfuse_secret_key",
    "eval_judge_llm_api_key",
    "parse_vlm_api_key",
    "parse_ocr_api_key",
    "parse_ocr_api_secret",
}


def settings_to_public_dict(cfg: Settings) -> dict[str, Any]:
    data = cfg.model_dump()
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k in SECRET_KEYS:
            out[k] = _mask_secret(v)
            out[f"{k}_set"] = bool(v)
        else:
            out[k] = v
    return out

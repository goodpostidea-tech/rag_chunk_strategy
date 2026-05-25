"""Observability: LangSmith & Langfuse (both disabled by default)."""

from __future__ import annotations

import os
from typing import Any

from chunk_lab.config import Settings, get_settings

_langfuse_handler: Any | None = None
_langfuse_client: Any | None = None
_initialized = False


def _resolve_langsmith_api_key(cfg: Settings) -> str | None:
    return cfg.langsmith_api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")


def _resolve_langfuse_keys(cfg: Settings) -> tuple[str | None, str | None, str | None]:
    public_key = cfg.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = cfg.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = cfg.langfuse_host or os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    return public_key, secret_key, host


def setup_observability(settings: Settings | None = None) -> dict[str, Any]:
    """
    Apply observability configuration.
    Called once at process startup (CLI / API lifespan).
    """
    global _langfuse_handler, _langfuse_client, _initialized

    cfg = settings or get_settings()

    # LangSmith: control via env (LangChain auto-instruments when tracing is on)
    if cfg.langsmith_enabled:
        api_key = _resolve_langsmith_api_key(cfg)
        if not api_key:
            raise ValueError(
                "CHUNK_LANGSMITH_ENABLED=true but no API key. "
                "Set CHUNK_LANGSMITH_API_KEY or LANGSMITH_API_KEY."
            )
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project
        if cfg.langsmith_endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = cfg.langsmith_endpoint
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    # Langfuse: LangChain callback handler
    _langfuse_handler = None
    _langfuse_client = None
    if cfg.langfuse_enabled:
        public_key, secret_key, host = _resolve_langfuse_keys(cfg)
        if not public_key or not secret_key:
            raise ValueError(
                "CHUNK_LANGFUSE_ENABLED=true but keys missing. "
                "Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY or CHUNK_LANGFUSE_*."
            )
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        client_kwargs: dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            client_kwargs["host"] = host
        _langfuse_client = Langfuse(**client_kwargs)
        _langfuse_handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

    _initialized = True
    return observability_info(cfg)


def shutdown_observability() -> None:
    """Flush pending Langfuse events on shutdown."""
    global _langfuse_client, _langfuse_handler
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
            _langfuse_client.shutdown()
        except Exception:
            pass
    _langfuse_handler = None
    _langfuse_client = None


def get_callbacks() -> list[Any]:
    """LangChain callbacks to attach to LLM invocations."""
    if _langfuse_handler is not None:
        return [_langfuse_handler]
    return []


def get_run_config(
    *,
    run_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RunnableConfig kwargs for llm.invoke / chain.invoke."""
    cfg: dict[str, Any] = {}
    callbacks = get_callbacks()
    if callbacks:
        cfg["callbacks"] = callbacks

    merged_tags = ["rag-chunk-strategy"]
    if tags:
        merged_tags.extend(tags)
    cfg["tags"] = merged_tags

    merged_meta: dict[str, Any] = {"app": "rag-chunk-strategy"}
    if metadata:
        merged_meta.update(metadata)
    cfg["metadata"] = merged_meta

    if run_name:
        cfg["run_name"] = run_name

    return cfg


def observability_info(settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    return {
        "langsmith": {
            "enabled": cfg.langsmith_enabled,
            "project": cfg.langsmith_project if cfg.langsmith_enabled else None,
            "configured": bool(_resolve_langsmith_api_key(cfg)) if cfg.langsmith_enabled else False,
        },
        "langfuse": {
            "enabled": cfg.langfuse_enabled,
            "project": cfg.langfuse_project if cfg.langfuse_enabled else None,
            "host": (cfg.langfuse_host or os.getenv("LANGFUSE_BASE_URL")) if cfg.langfuse_enabled else None,
            "handler_active": _langfuse_handler is not None,
            "configured": bool(_resolve_langfuse_keys(cfg)[0] and _resolve_langfuse_keys(cfg)[1])
            if cfg.langfuse_enabled
            else False,
        },
    }


def ensure_observability_initialized() -> None:
    """Idempotent init for code paths that skip CLI/API startup."""
    global _initialized
    if not _initialized:
        setup_observability()

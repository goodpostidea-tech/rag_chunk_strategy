"""已合并到 chunk_api。本地只需: uv run uvicorn chunk_api:app --reload --port 8765"""

from chunk_api import app  # noqa: F401

__all__ = ["app"]

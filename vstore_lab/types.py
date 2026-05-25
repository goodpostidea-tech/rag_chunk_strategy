"""Types for vector store benchmark experiment."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class VStoreBenchRow:
    provider: str
    index_type: str
    vector_count: int
    build_ms: float
    query_p50_ms: float
    query_p99_ms: float
    memory_mb: float | None = None
    filter_slowdown_ratio: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VStoreBenchResponse:
    vector_counts: list[int]
    rows: list[VStoreBenchRow]
    elapsed_ms: float = 0.0


class VStoreBenchRequest(BaseModel):
    vector_counts: list[int] = Field(default_factory=lambda: [1000, 10000])
    providers: list[str] | None = None

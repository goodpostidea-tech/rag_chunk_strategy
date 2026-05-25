"""Types for document parsing experiment."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class ParsedTable:
    rows: list[list[str]]
    source: str = "unknown"


@dataclass
class ParsedHeading:
    level: int
    text: str
    source: str = "markdown"


@dataclass
class ParseResult:
    parser: str
    file_path: str
    file_type: str
    text: str
    char_count: int
    table_count: int
    heading_count: int
    tables: list[ParsedTable] = field(default_factory=list)
    headings: list[ParsedHeading] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseMetrics:
    parser: str
    file_path: str
    doc_profile: str
    text_completeness: float
    table_structure_score: float
    heading_detection_score: float
    char_count: int
    table_count: int
    heading_count: int
    elapsed_ms: float
    error: str | None = None


@dataclass
class ParseRunResult:
    parser: str
    results: list[ParseResult]
    metrics: list[ParseMetrics] = field(default_factory=list)


@dataclass
class ParseCompareResult:
    file_path: str
    doc_profile: str
    results: list[ParseResult]
    metrics: list[ParseMetrics]
    ranking: list[dict[str, Any]] = field(default_factory=list)


class VlmParseConfig(BaseModel):
    """Multimodal LLM settings for vlm_pdf parser."""

    provider: str
    model: str
    api_base: str | None = None
    api_key: str | None = None
    max_pages: int | None = Field(
        default=None,
        description="Max pages to send to VLM; None uses server default cap (50). 0 = all pages.",
    )
    max_workers: int = Field(default=3, ge=1, le=8)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=16384)
    dpi: int = Field(default=150, ge=72, le=300)
    max_edge: int = Field(default=1280, ge=512, le=2048)


class ParseRunRequest(BaseModel):
    file_path: str
    parsers: list[str] | None = None
    doc_profile: str | None = None


class ParseBatchRequest(BaseModel):
    dir_path: str
    parsers: list[str] | None = None
    recursive: bool = False
    file_types: list[str] | None = None

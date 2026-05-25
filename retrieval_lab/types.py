"""Types for retrieval strategy experiment."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

QueryType = Literal["factual", "semantic", "keyword"]


@dataclass
class RetrievalQAPair:
    question: str
    expected_answer: str = ""
    expected_in_chunk: str | None = None
    query_type: QueryType = "factual"

    @classmethod
    def from_dict(cls, item: dict) -> "RetrievalQAPair":
        data = {k: v for k, v in item.items() if k != "paper_id"}
        if not data.get("expected_answer") and data.get("expected_in_chunk"):
            data["expected_answer"] = data["expected_in_chunk"]
        qt = data.get("query_type", "factual")
        if qt not in ("factual", "semantic", "keyword"):
            qt = "factual"
        return cls(
            question=data["question"],
            expected_answer=data.get("expected_answer", ""),
            expected_in_chunk=data.get("expected_in_chunk"),
            query_type=qt,  # type: ignore[arg-type]
        )


@dataclass
class RetrievalDetail:
    question: str
    query_type: str
    context_recall: bool
    context_precision: float
    reciprocal_rank: float
    top_preview: str = ""


@dataclass
class RetrievalEvalResponse:
    method: str
    chunk_strategy: str
    top_k: int
    context_recall: float
    context_precision: float
    mrr: float
    hits: int
    total: int
    by_query_type: dict[str, dict[str, float]] = field(default_factory=dict)
    details: list[RetrievalDetail] = field(default_factory=list)
    elapsed_ms: float = 0.0


class RetrievalEvalRequest(BaseModel):
    text: str | None = None
    local_dir: str | None = None
    retrieval_method: str = "hybrid"
    chunk_strategy: str = "recursive_baseline"
    qa_pairs: list[dict[str, Any]] | None = None
    qa_source: str = "active"
    top_k: int | None = None
    doc_title: str = ""
    retrieval_params: dict[str, Any] = Field(default_factory=dict)
    chunk_params: dict[str, Any] = Field(default_factory=dict)
    local_recursive: bool = True
    local_file_types: list[str] | None = None


class RetrievalCompareRequest(BaseModel):
    text: str | None = None
    local_dir: str | None = None
    methods: list[str] | None = None
    chunk_strategy: str = "recursive_baseline"
    qa_pairs: list[dict[str, Any]] | None = None
    qa_source: str = "active"
    top_k: int | None = None
    local_recursive: bool = True
    local_file_types: list[str] | None = None

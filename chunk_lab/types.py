"""Shared types for chunk lab."""

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field


@dataclass
class ChunkStats:
    chunk_count: int
    avg_char_length: float
    min_char_length: int
    max_char_length: int
    avg_token_estimate: float
    total_chars: int
    strategy: str
    category: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRunResult:
    strategy: str
    category: str
    description: str
    chunks: list[Document]
    stats: ChunkStats
    elapsed_ms: float


class ChunkRequest(BaseModel):
    text: str | None = None
    strategy: str = "recursive_baseline"
    doc_title: str = ""
    doc_section: str = ""
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    # 本地目录语料源（与 text 二选一）
    local_dir: str | None = None
    local_recursive: bool = True
    local_file_types: list[str] | None = None


class ChunkResponse(BaseModel):
    strategy: str
    category: str
    stats: dict[str, Any]
    elapsed_ms: float
    chunks: list[dict[str, Any]]


class CompareRequest(BaseModel):
    text: str | None = None
    strategies: list[str] | None = None
    doc_title: str = ""
    local_dir: str | None = None
    local_recursive: bool = True
    local_file_types: list[str] | None = None


class CompareResponse(BaseModel):
    results: list[dict[str, Any]]


class EvalQAPair(BaseModel):
    question: str
    expected_answer: str = ""
    expected_in_chunk: str | None = None
    query_type: str = "factual"  # factual | semantic | keyword（检索实验用）

    @classmethod
    def from_dict(cls, item: dict) -> "EvalQAPair":
        """Normalize QA dict (e.g. papers_qa.json may omit expected_answer)."""
        data = {k: v for k, v in item.items() if k != "paper_id"}
        if not data.get("expected_answer") and data.get("expected_in_chunk"):
            data["expected_answer"] = data["expected_in_chunk"]
        if data.get("query_type") not in ("factual", "semantic", "keyword"):
            data["query_type"] = "factual"
        return cls(**data)


class EvalRequest(BaseModel):
    text: str | None = None
    strategy: str = "recursive_baseline"
    qa_pairs: list[EvalQAPair] | None = None
    top_k: int | None = None
    doc_title: str = ""
    run_name: str = ""
    # 未传 qa_pairs 时：custom=用户自定义集 | builtin=内置 | active=优先自定义
    qa_source: str = "active"
    # LLM-as-Judge：默认关闭；True 时按 CHUNK_EVAL_JUDGE_MODE 执行
    llm_judge: bool | None = None
    eval_mode: str | None = None  # substring | judge | both
    # 本地目录语料源（与 text 二选一）
    local_dir: str | None = None
    local_recursive: bool = True
    local_file_types: list[str] | None = None


class TestConnectionRequest(BaseModel):
    provider: str
    api_base: str | None = None
    api_key: str | None = None
    model: str = "deepseek-chat"


class SettingsUpdateRequest(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


class EvalResponse(BaseModel):
    strategy: str
    eval_mode: str = "substring"
    recall_at_k: float
    hits: int
    total: int
    # LLM-as-Judge 指标（eval_mode 为 judge/both 时有值）
    judge_score: float | None = None
    judge_hits: int | None = None
    judge_pass_rate: float | None = None
    details: list[dict[str, Any]]
    # 持久化标识
    run_id: str | None = None
    run_name: str = ""
    index_id: str | None = None


# --- RAG chat (vector-indexed corpus) ---


class RagIndexRequest(BaseModel):
    text: str | None = None
    strategy: str = "recursive_baseline"
    doc_title: str = ""
    session_id: str | None = None
    index_id: str | None = None


class RagIndexResponse(BaseModel):
    session_id: str
    strategy: str
    chunk_count: int
    stats: dict[str, Any]
    doc_title: str = ""


class RagChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int | None = None
    index_id: str | None = None


class RagChatResponse(BaseModel):
    session_id: str
    strategy: str
    question: str
    answer: str
    top_k: int
    sources: list[dict[str, Any]]


# --- QA dataset ---


class QADatasetPayload(BaseModel):
    name: str = "用户自定义"
    qa_pairs: list[EvalQAPair]


class QADatasetResponse(BaseModel):
    source: str
    name: str | None = None
    updated_at: str | None = None
    qa_pairs: list[EvalQAPair]
    count: int


class QAGenerateRequest(BaseModel):
    text: str
    num_pairs: int = Field(default=5, ge=1, le=20)
    doc_title: str = ""
    save_as_custom: bool = False
    dataset_name: str = "AI 生成测试集"


class QAGenerateResponse(BaseModel):
    qa_pairs: list[EvalQAPair]
    count: int
    saved: bool = False
    dataset_name: str | None = None


# --- Workspace ---


class WorkspaceCreate(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9\-]*$", max_length=64)
    name: str


class WorkspaceResponse(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str


# --- Eval history ---


class EvalHistoryItem(BaseModel):
    id: str
    workspace_id: str
    strategy: str
    eval_mode: str
    recall_at_k: float
    hits: int
    total: int
    judge_score: float | None = None
    judge_pass_rate: float | None = None
    top_k: int
    doc_title: str = ""
    run_name: str = ""
    qa_source: str = "builtin"
    index_id: str | None = None
    created_at: str


# --- Vector index ---


class VectorIndexInfo(BaseModel):
    id: str
    workspace_id: str
    strategy: str
    doc_title: str = ""
    run_name: str = ""
    chunk_count: int
    embedding_provider: str
    embedding_model: str
    source_eval_id: str | None = None
    stats: dict[str, Any] | None = None
    created_at: str



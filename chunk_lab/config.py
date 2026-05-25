"""Application settings — DeepSeek defaults, overridable via env + runtime overlay."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (contextual / proposition strategies) ---
    chunk_llm_model: str = "deepseek-chat"
    chunk_llm_provider: str = "deepseek"
    chunk_llm_temperature: float = 0.0
    chunk_llm_api_base: str | None = None
    chunk_llm_api_key: str | None = None
    chunk_llm_max_tokens: int = 4096

    # --- Embeddings (semantic chunker & retrieval eval) ---
    # Providers: huggingface | openai | deepseek | mistral | qwen | dashscope
    chunk_embedding_provider: str = "huggingface"
    chunk_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    chunk_embedding_api_base: str | None = None
    chunk_embedding_api_key: str | None = None
    embedding_device: str = "cpu"

    # --- Vector store (default: in-memory FAISS) ---
    # Providers: faiss | qdrant | milvus | chroma | pinecone
    vector_store_provider: str = "faiss"
    vector_store_collection_prefix: str = "chunk_lab"

    # Qdrant（可用 QDRANT_BASE_URL / QDRANT_API_KEY）
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_token: str = ""

    # Chroma（留空 persist 目录则内存模式）
    chroma_persist_dir: str = ""

    # Pinecone
    pinecone_index_name: str = "chunk-lab"
    pinecone_api_key: str | None = None
    pinecone_namespace: str = ""

    # --- Baseline splitter (FloTorch 2026 recommended starting point) ---
    baseline_chunk_size: int = 512
    baseline_chunk_overlap: int = 128

    # --- Semantic ---
    semantic_min_chunk_size: int = 150
    semantic_breakpoint_type: str = "percentile"
    semantic_breakpoint_amount: float = 95.0

    # --- Parent-child ---
    child_chunk_size: int = 200
    child_chunk_overlap: int = 20
    parent_chunk_size: int = 800
    parent_chunk_overlap: int = 80

    # --- Multi-granularity ---
    multi_granularity_sizes: str = "256,512,1024"

    # --- Contextual (LLM prefix per chunk) ---
    contextual_max_chunks: int = 20

    # --- Eval ---
    eval_top_k: int = 5
    papers_cache_dir: str = "data/papers"

    # --- LLM-as-Judge（默认关闭，CHUNK_EVAL_JUDGE_ENABLED=true 开启）---
    eval_judge_enabled: bool = False
    # judge: 仅 LLM 评判 | both: 子串匹配 + LLM 评判 | substring: 等同关闭 judge
    eval_judge_mode: str = "both"
    eval_judge_temperature: float = 0.0
    eval_judge_max_tokens: int = 1024
    # 未设置则复用 CHUNK_LLM_*（默认 DeepSeek）
    eval_judge_llm_model: str | None = None
    eval_judge_llm_provider: str | None = None
    eval_judge_llm_api_base: str | None = None
    eval_judge_llm_api_key: str | None = None

    # --- Parse lab: VLM PDF parser (vlm_pdf) ---
    parse_vlm_provider: str = "openai"
    parse_vlm_model: str = "gpt-4o"
    parse_vlm_api_base: str | None = None
    parse_vlm_api_key: str | None = None
    parse_vlm_max_pages: int = 50
    parse_vlm_max_workers: int = 3
    parse_vlm_temperature: float = 0.0
    parse_vlm_max_tokens: int = 4096
    parse_vlm_dpi: int = 150
    parse_vlm_max_edge: int = 1280

    # --- Parse lab: OCR API（ocr_api 解析器，在系统配置 → OCR API）---
    parse_ocr_api_provider: str = "azure"
    parse_ocr_api_model: str = "gpt-4o"
    parse_ocr_api_base: str | None = None
    parse_ocr_api_key: str | None = None
    parse_ocr_api_secret: str | None = None
    parse_ocr_api_region: str = "ap-guangzhou"
    parse_ocr_max_pages: int = 50
    parse_ocr_dpi: int = 200

    # --- Observability（默认均关闭）---
    langsmith_enabled: bool = False
    langsmith_project: str = "rag-chunk-strategy"
    langsmith_api_key: str | None = None
    langsmith_endpoint: str | None = None

    langfuse_enabled: bool = False
    langfuse_project: str = "rag-chunk-strategy"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None


def get_settings() -> Settings:
    from chunk_lab.settings_store import get_effective_settings

    return get_effective_settings()

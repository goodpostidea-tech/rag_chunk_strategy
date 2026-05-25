"""Factory for LLM, embedding models, and vector stores."""

import os
import threading

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from chunk_lab.config import Settings, get_settings

_EMBEDDING_CACHE: dict[str, Embeddings] = {}
_EMBEDDING_LOCK = threading.Lock()

KNOWN_LLM_PROVIDERS = (
    "anthropic", "anthropic_bedrock", "azure_ai", "azure_openai", "baseten",
    "bedrock", "bedrock_converse", "cohere", "deepseek", "fireworks",
    "google_anthropic_vertex", "google_genai", "google_vertexai", "groq",
    "huggingface", "ibm", "litellm", "mistralai", "nvidia", "ollama",
    "openai", "openrouter", "perplexity", "together", "upstage", "xai",
)

EMBEDDING_PROVIDERS = (
    "huggingface",
    "openai",
    "deepseek",
    "mistral",
    "qwen",
    "dashscope",
)


_LLM_PROVIDER_ALIASES = {
    "mistral": "mistralai",
}


def get_llm(settings: Settings | None = None, **overrides) -> BaseChatModel:
    """Build chat model; defaults to DeepSeek, overridable via env or kwargs."""
    cfg = settings or get_settings()
    kwargs: dict = {
        "model": overrides.pop("model", cfg.chunk_llm_model),
        "model_provider": overrides.pop("model_provider", cfg.chunk_llm_provider),
        "temperature": overrides.pop("temperature", cfg.chunk_llm_temperature),
        "max_tokens": overrides.pop("max_tokens", cfg.chunk_llm_max_tokens),
    }
    if cfg.chunk_llm_api_base:
        kwargs["api_base"] = cfg.chunk_llm_api_base
    api_key = cfg.chunk_llm_api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    kwargs.update(overrides)

    provider = kwargs["model_provider"]
    provider = _LLM_PROVIDER_ALIASES.get(provider, provider)
    kwargs["model_provider"] = provider

    if provider not in KNOWN_LLM_PROVIDERS:
        from langchain_openai import ChatOpenAI

        base_url = kwargs.pop("api_base", "") or ""
        if not base_url:
            raise ValueError(
                f"未知 LLM 提供商 '{provider}'，需要填写 API Base 以使用 OpenAI 兼容模式。"
            )
        return ChatOpenAI(
            model=kwargs["model"],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens"),
            api_key=kwargs.get("api_key") or "",
            base_url=base_url,
        )

    # 已知 provider 使用各自 SDK，不传 api_base（仅 openai 系需要 base_url）
    api_base = kwargs.pop("api_base", None)
    if api_base and provider in ("openai", "azure_openai"):
        kwargs["base_url"] = api_base

    return init_chat_model(**kwargs)


def _embedding_cache_key(cfg: Settings) -> str:
    return (
        f"{cfg.chunk_embedding_provider}:{cfg.chunk_embedding_model}:"
        f"{cfg.embedding_device}:{cfg.chunk_embedding_api_base or ''}"
    )


def _resolve_hf_device(requested: str) -> str:
    """Map requested device to a safe runtime device (avoid meta-tensor .to() on CUDA)."""
    dev = (requested or "cpu").strip().lower()
    if dev in ("cuda", "gpu", "auto"):
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return dev


class SentenceTransformerEmbeddings(Embeddings):
    """Local embeddings via sentence-transformers (stable device load, no meta tensors)."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name,
            device=_resolve_hf_device(device),
            trust_remote_code=True,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors).tolist()

    def embed_query(self, text: str) -> list[float]:
        import numpy as np

        vector = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector)[0].tolist()


def _cached_embeddings(key: str, factory) -> Embeddings:
    """Process-wide singleton — parallel /eval must not reload the same HF model."""
    with _EMBEDDING_LOCK:
        if key not in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE[key] = factory()
        return _EMBEDDING_CACHE[key]


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Build embedding model from configured provider."""
    cfg = settings or get_settings()
    provider = cfg.chunk_embedding_provider.lower().strip()
    cache_key = _embedding_cache_key(cfg)

    if provider == "huggingface":
        return _cached_embeddings(
            cache_key,
            lambda: SentenceTransformerEmbeddings(
                cfg.chunk_embedding_model,
                device=cfg.embedding_device,
            ),
        )

    if provider == "mistral":
        from langchain_mistralai import MistralAIEmbeddings

        api_key = cfg.chunk_embedding_api_key or os.getenv("MISTRAL_API_KEY")
        return MistralAIEmbeddings(
            model=cfg.chunk_embedding_model,
            api_key=api_key,
        )

    if provider in ("qwen", "dashscope", "tongyi"):
        from langchain_community.embeddings import DashScopeEmbeddings

        api_key = (
            cfg.chunk_embedding_api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
        )
        return DashScopeEmbeddings(
            model=cfg.chunk_embedding_model,
            dashscope_api_key=api_key,
        )

    if provider == "deepseek":
        from langchain_openai import OpenAIEmbeddings

        api_key = cfg.chunk_embedding_api_key or os.getenv("DEEPSEEK_API_KEY")
        return OpenAIEmbeddings(
            model=cfg.chunk_embedding_model,
            api_key=api_key,
            base_url=cfg.chunk_embedding_api_base or "https://api.deepseek.com",
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = cfg.chunk_embedding_api_key or os.getenv("OPENAI_API_KEY")
        emb_kwargs: dict = {"model": cfg.chunk_embedding_model, "api_key": api_key}
        if cfg.chunk_embedding_api_base:
            emb_kwargs["base_url"] = cfg.chunk_embedding_api_base
        return OpenAIEmbeddings(**emb_kwargs)

    # 未知 provider → OpenAI 兼容模式
    from langchain_openai import OpenAIEmbeddings

    api_key = cfg.chunk_embedding_api_key or os.getenv("OPENAI_API_KEY") or ""
    base_url = cfg.chunk_embedding_api_base
    if not base_url:
        raise ValueError(
            f"未知 embedding 提供商 '{provider}'，需要填写 API Base 以使用 OpenAI 兼容模式。"
            f" 已知提供商: {', '.join(EMBEDDING_PROVIDERS)}"
        )
    return OpenAIEmbeddings(model=cfg.chunk_embedding_model, api_key=api_key, base_url=base_url)


def embedding_info(settings: Settings | None = None) -> dict:
    cfg = settings or get_settings()
    return {
        "provider": cfg.chunk_embedding_provider,
        "model": cfg.chunk_embedding_model,
        "supported_providers": list(EMBEDDING_PROVIDERS),
        "openai_compatible_fallback": True,
    }

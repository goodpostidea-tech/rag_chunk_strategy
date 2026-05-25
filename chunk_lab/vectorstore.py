"""Vector store factory — default in-memory FAISS, optional Qdrant/Milvus/Chroma/Pinecone."""

import os
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from chunk_lab.config import Settings, get_settings

SUPPORTED_VECTOR_STORES = ("faiss", "qdrant", "milvus", "chroma", "pinecone")


def _collection_name(settings: Settings, suffix: str | None = None) -> str:
    base = settings.vector_store_collection_prefix
    uid = suffix or uuid.uuid4().hex[:8]
    return f"{base}_{uid}"


def _resolve_qdrant_url(settings: Settings) -> str:
    return (
        settings.qdrant_url
        or os.getenv("QDRANT_BASE_URL")
        or os.getenv("QDRANT_URL")
        or "http://localhost:6333"
    )


def _resolve_qdrant_api_key(settings: Settings) -> str | None:
    return settings.qdrant_api_key or os.getenv("QDRANT_API_KEY")


def create_vectorstore(
    documents: list[Document],
    embedding: Embeddings,
    *,
    settings: Settings | None = None,
    collection_name: str | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a vector store from documents using configured provider."""
    cfg = settings or get_settings()
    provider = (kwargs.pop("provider", None) or cfg.vector_store_provider).lower()
    name = collection_name or _collection_name(cfg)

    if provider == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.from_documents(documents, embedding, **kwargs)

    if provider == "chroma":
        from langchain_community.vectorstores import Chroma

        persist = cfg.chroma_persist_dir or None
        return Chroma.from_documents(
            documents,
            embedding,
            collection_name=name,
            persist_directory=persist,
            **kwargs,
        )

    if provider == "qdrant":
        from langchain_community.vectorstores import Qdrant

        return Qdrant.from_documents(
            documents,
            embedding,
            url=_resolve_qdrant_url(cfg),
            api_key=_resolve_qdrant_api_key(cfg),
            collection_name=name,
            **kwargs,
        )

    if provider == "milvus":
        from langchain_community.vectorstores import Milvus

        connection_args: dict = {"uri": cfg.milvus_uri}
        if cfg.milvus_user:
            connection_args["user"] = cfg.milvus_user
        if cfg.milvus_password:
            connection_args["password"] = cfg.milvus_password
        if cfg.milvus_token:
            connection_args["token"] = cfg.milvus_token

        return Milvus.from_documents(
            documents,
            embedding,
            collection_name=name,
            connection_args=connection_args,
            **kwargs,
        )

    if provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        api_key = cfg.pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY or CHUNK_PINECONE_API_KEY is required for Pinecone.")

        index_name = cfg.pinecone_index_name
        namespace = cfg.pinecone_namespace or name
        return PineconeVectorStore.from_documents(
            documents,
            embedding,
            index_name=index_name,
            namespace=namespace,
            pinecone_api_key=api_key,
            **kwargs,
        )

    raise ValueError(
        f"Unknown vector_store_provider '{provider}'. "
        f"Supported: {', '.join(SUPPORTED_VECTOR_STORES)}"
    )


def create_empty_vectorstore(
    embedding: Embeddings,
    *,
    settings: Settings | None = None,
    collection_name: str | None = None,
) -> VectorStore:
    """Create a minimal store for ParentDocumentRetriever.add_documents."""
    cfg = settings or get_settings()
    provider = cfg.vector_store_provider.lower()
    name = collection_name or _collection_name(cfg)

    if provider == "faiss":
        from langchain_community.vectorstores import FAISS

        return FAISS.from_texts([" "], embedding)

    if provider == "chroma":
        from langchain_community.vectorstores import Chroma

        persist = cfg.chroma_persist_dir or None
        return Chroma(
            collection_name=name,
            embedding_function=embedding,
            persist_directory=persist,
        )

    if provider == "qdrant":
        from langchain_community.vectorstores import Qdrant
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(
            url=_resolve_qdrant_url(cfg),
            api_key=_resolve_qdrant_api_key(cfg),
        )
        dim = len(embedding.embed_query(" "))
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        return Qdrant(
            client=client,
            collection_name=name,
            embeddings=embedding,
        )

    if provider == "milvus":
        from langchain_community.vectorstores import Milvus

        connection_args: dict = {"uri": cfg.milvus_uri}
        if cfg.milvus_token:
            connection_args["token"] = cfg.milvus_token
        return Milvus(
            embedding_function=embedding,
            collection_name=name,
            connection_args=connection_args,
        )

    # Pinecone / fallback: seed with placeholder document
    return create_vectorstore(
        [Document(page_content=" ")],
        embedding,
        settings=cfg,
        collection_name=name,
    )


def extract_child_documents(vectorstore: VectorStore) -> list[Document] | None:
    """Read indexed documents when the backend supports native get()."""
    if hasattr(vectorstore, "get"):
        try:
            data = vectorstore.get()  # type: ignore[union-attr]
            contents = data.get("documents") or []
            metadatas = data.get("metadatas") or [{}] * len(contents)
            docs = []
            for content, meta in zip(contents, metadatas, strict=False):
                if content and not (meta or {}).get("_init"):
                    docs.append(Document(page_content=content, metadata=meta or {}))
            return docs or None
        except Exception:
            return None
    return None


def similarity_search(
    vectorstore: VectorStore,
    query: str,
    k: int,
) -> list[Document]:
    return vectorstore.similarity_search(query, k=k)


def vectorstore_info(settings: Settings | None = None) -> dict:
    cfg = settings or get_settings()
    return {
        "provider": cfg.vector_store_provider,
        "supported": list(SUPPORTED_VECTOR_STORES),
        "collection_prefix": cfg.vector_store_collection_prefix,
    }

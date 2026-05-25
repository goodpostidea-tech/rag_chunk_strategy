"""RAG chat over indexed chunk vector stores (in-memory sessions)."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain.agents import create_agent


from chunk_lab.config import get_settings
from chunk_lab.models import get_embeddings, get_llm
from chunk_lab.observability import ensure_observability_initialized, get_run_config
from chunk_lab.pipeline import run_chunk
from chunk_lab.stats import stats_to_dict
from chunk_lab.vectorstore import create_vectorstore

RAG_SYSTEM_PROMPT = """你是基于知识库检索的问答助手。请**仅根据**下方提供的检索上下文回答用户问题。
若上下文不足以回答，请明确说明「根据当前知识库无法确定」，不要编造。
回答使用简洁中文；若引用了某段上下文，在句末标注 [1]、[2] 等编号。

{context}"""


@dataclass
class RagSession:
    session_id: str
    strategy: str
    vectorstore: VectorStore
    chunk_count: int
    doc_title: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    stats: dict[str, Any] = field(default_factory=dict)


class RagSessionStore:
    """Thread-safe in-memory session registry (ephemeral; restart clears)."""

    def __init__(self, *, ttl_seconds: int = 3600) -> None:
        self._sessions: dict[str, RagSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.last_used_at > self._ttl]
        for sid in expired:
            del self._sessions[sid]

    def create(
        self,
        text: str,
        strategy: str,
        *,
        doc_title: str = "",
        session_id: str | None = None,
    ) -> RagSession:
        run = run_chunk(text, strategy, title=doc_title)
        if not run.chunks:
            raise ValueError("分块结果为空，无法建立向量索引")

        settings = get_settings()
        embeddings = get_embeddings(settings)
        sid = session_id or uuid.uuid4().hex
        vs = create_vectorstore(
            run.chunks,
            embeddings,
            settings=settings,
            collection_name=f"chat_{strategy}_{sid[:8]}",
        )
        session = RagSession(
            session_id=sid,
            strategy=strategy,
            vectorstore=vs,
            chunk_count=len(run.chunks),
            doc_title=doc_title,
            stats=stats_to_dict(run.stats),
        )
        with self._lock:
            self._purge_expired()
            self._sessions[sid] = session
        return session

    def create_from_index(
        self,
        index_id: str,
        *,
        session_id: str | None = None,
    ) -> RagSession:
        """Load a persisted FAISS index and create a chat session without re-embedding."""
        from langchain_community.vectorstores import FAISS

        from chunk_lab.db import get_vector_index, resolve_index_path

        info = get_vector_index(index_id)
        if not info:
            raise ValueError(f"向量索引不存在: {index_id}")

        disk_path = resolve_index_path(index_id)
        if not disk_path:
            raise ValueError(f"索引文件不存在: {index_id}")

        settings = get_settings()
        embeddings = get_embeddings(settings)
        vs = FAISS.load_local(
            str(disk_path), embeddings, allow_dangerous_deserialization=True
        )

        sid = session_id or uuid.uuid4().hex
        session = RagSession(
            session_id=sid,
            strategy=info["strategy"],
            vectorstore=vs,
            chunk_count=info["chunk_count"],
            doc_title=info.get("doc_title", ""),
            stats=info.get("stats") or {},
        )
        with self._lock:
            self._purge_expired()
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> RagSession | None:
        with self._lock:
            self._purge_expired()
            session = self._sessions.get(session_id)
            if session:
                session.last_used_at = time.time()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            self._purge_expired()
            return [
                {
                    "session_id": s.session_id,
                    "strategy": s.strategy,
                    "chunk_count": s.chunk_count,
                    "doc_title": s.doc_title,
                    "created_at": s.created_at,
                    "last_used_at": s.last_used_at,
                }
                for s in self._sessions.values()
            ]


# Process-wide singleton
_session_store = RagSessionStore()


def get_session_store() -> RagSessionStore:
    return _session_store


def _build_rag_chain(session: RagSession, top_k: int):
    """Build a retrieval QA chain using LCEL."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough

    settings = get_settings()
    llm = get_llm(settings)

    retriever = session.vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    def _retrieve(x):
        return retriever.invoke(x["input"])

    def _format_context(x):
        docs = x["context_docs"]
        return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    chain = (
        RunnablePassthrough.assign(context_docs=_retrieve)
        | RunnablePassthrough.assign(context=_format_context)
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def chat_with_session(
    session_id: str,
    message: str,
    *,
    top_k: int | None = None,
) -> dict[str, Any]:
    session = get_session_store().get(session_id)
    if not session:
        raise ValueError(f"会话不存在或已过期: {session_id}")

    question = message.strip()
    if not question:
        raise ValueError("消息不能为空")

    settings = get_settings()
    k = top_k or settings.eval_top_k

    ensure_observability_initialized()
    chain, retriever = _build_rag_chain(session, k)

    context_docs = retriever.invoke(question)

    answer = chain.invoke(
        {"input": question},
        config=get_run_config(run_name="rag_chat", tags=["rag", "chat"]),
    )

    sources = [
        {
            "index": i + 1,
            "content": d.page_content[:800],
            "char_length": len(d.page_content),
            "metadata": d.metadata,
        }
        for i, d in enumerate(context_docs)
    ]

    return {
        "session_id": session_id,
        "strategy": session.strategy,
        "question": question,
        "answer": answer,
        "top_k": k,
        "sources": sources,
    }

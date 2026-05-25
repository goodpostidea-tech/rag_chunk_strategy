"""Contextual chunking — LLM-generated context prefix per chunk (Anthropic-style)."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from chunk_lab.base import BaseChunkStrategy
from chunk_lab.models import get_llm
from chunk_lab.observability import ensure_observability_initialized, get_run_config
from chunk_lab.strategies.baseline import RecursiveBaselineStrategy

DOCUMENT_CONTEXT_PROMPT = """<document>
{doc_content}
</document>

请为以下 chunk 提供简短上下文（1-3 句），说明其在整篇文档中的位置和作用。
仅输出上下文文本，不要其他内容。"""

CHUNK_CONTEXT_PROMPT = """<chunk>
{chunk_content}
</chunk>"""


class ContextualChunkStrategy(BaseChunkStrategy):
    name = "contextual"
    category = "B"
    description = "上下文增强分块：基线切分后由 LLM 为每个 chunk 生成文档内位置说明前缀"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        llm = kwargs.get("llm") or get_llm(self.settings)
        max_chunks = kwargs.get("max_chunks", self.settings.contextual_max_chunks)
        max_doc_chars = kwargs.get("max_doc_chars", 12000)

        base = RecursiveBaselineStrategy(self.settings)
        raw_chunks = base.chunk(documents, **kwargs)

        if len(raw_chunks) > max_chunks:
            raw_chunks = raw_chunks[:max_chunks]

        full_text = "\n\n".join(d.page_content for d in documents)
        if len(full_text) > max_doc_chars:
            full_text = full_text[:max_doc_chars]

        contextualized: list[Document] = []
        ensure_observability_initialized()

        for i, ch in enumerate(raw_chunks):
            prompt = (
                DOCUMENT_CONTEXT_PROMPT.format(doc_content=full_text)
                + "\n\n"
                + CHUNK_CONTEXT_PROMPT.format(chunk_content=ch.page_content[:4000])
            )
            try:
                resp = llm.invoke(
                    [HumanMessage(content=prompt)],
                    config=get_run_config(
                        run_name="contextual_chunk",
                        tags=["chunk", "contextual"],
                        metadata={"chunk_index": i},
                    ),
                )
                context = resp.content if isinstance(resp.content, str) else str(resp.content)
            except Exception:
                context = ""

            prefix = f"{context.strip()}\n\n" if context.strip() else ""
            contextualized.append(
                Document(
                    page_content=f"{prefix}{ch.page_content}",
                    metadata={
                        **ch.metadata,
                        "strategy": self.name,
                        "chunk_index": i,
                        "has_context_prefix": bool(prefix),
                    },
                )
            )
        return contextualized

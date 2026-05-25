"""Proposition-based chunking — LLM extracts atomic facts."""

import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunk_lab.base import BaseChunkStrategy
from chunk_lab.models import get_llm
from chunk_lab.observability import ensure_observability_initialized, get_run_config

PROPOSITION_PROMPT = """将以下文本拆分成原子级事实命题。
要求：
1. 每个命题完整表达一个独立事实
2. 解析所有代词指代（将"它"、"他们"替换为具体名称）
3. 每个命题可单独理解，无需上下文
4. 保留原文的关键细节

输出 JSON 格式（仅输出 JSON，不要其他内容）：
{{"propositions": ["命题1", "命题2", ...]}}

文本：
{text}"""


def _parse_propositions(raw: str) -> list[str]:
    raw = raw.strip()
    try:
        data = json.loads(raw)
        return data.get("propositions", [])
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data.get("propositions", [])
            except json.JSONDecodeError:
                pass
    return [raw] if raw else []


class PropositionChunkStrategy(BaseChunkStrategy):
    name = "proposition"
    category = "B"
    description = "命题分块：LLM 将文本拆为自包含原子事实，每命题独立成 chunk"

    def chunk(self, documents: list[Document], **kwargs: Any) -> list[Document]:
        llm = kwargs.get("llm") or get_llm(self.settings)
        window_size = kwargs.get("window_size", 4000)

        window_splitter = RecursiveCharacterTextSplitter(
            chunk_size=window_size,
            chunk_overlap=200,
        )

        all_props: list[Document] = []
        ensure_observability_initialized()

        for doc_idx, doc in enumerate(documents):
            windows = window_splitter.split_documents([doc])

            for win_idx, window in enumerate(windows):
                text = window.page_content
                prompt = PROPOSITION_PROMPT.format(text=text)
                try:
                    resp = llm.invoke(
                        [HumanMessage(content=prompt)],
                        config=get_run_config(
                            run_name="proposition_chunk",
                            tags=["chunk", "proposition"],
                            metadata={"doc_index": doc_idx, "window_index": win_idx},
                        ),
                    )
                    content = resp.content if isinstance(resp.content, str) else str(resp.content)
                    propositions = _parse_propositions(content)
                except Exception:
                    propositions = [text]

                for prop_idx, prop in enumerate(propositions):
                    if not prop.strip():
                        continue
                    all_props.append(
                        Document(
                            page_content=prop.strip(),
                            metadata={
                                **doc.metadata,
                                "strategy": self.name,
                                "doc_index": doc_idx,
                                "window_index": win_idx,
                                "proposition_index": prop_idx,
                            },
                        )
                    )
        return all_props

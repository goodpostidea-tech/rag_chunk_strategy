"""LLM-based QA test set generation from user corpus."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage

from chunk_lab.models import get_llm
from chunk_lab.observability import ensure_observability_initialized, get_run_config
from chunk_lab.types import EvalQAPair

GENERATE_QA_PROMPT = """你是 RAG 评测数据标注专家。请根据下方文档生成 {num_pairs} 条检索评测用 QA 对。

要求：
1. 问题应覆盖文档中的关键事实，难度适中，避免过于笼统
2. expected_answer 为完整参考答案（可简短）
3. expected_in_chunk 为应出现在检索 chunk 中的关键子串（尽量取自原文，便于子串匹配评测）
4. 仅输出 JSON 数组，不要 markdown 或其它说明

文档标题：{title}

文档内容（已截断）：
{text}

输出格式示例：
[
  {{"question": "...", "expected_answer": "...", "expected_in_chunk": "..."}}
]"""


def _parse_qa_json(raw: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("模型未返回有效 JSON 数组") from None
        data = json.loads(match.group())
    if not isinstance(data, list):
        raise ValueError("模型返回的不是 JSON 数组")
    return data


def generate_qa_from_corpus(
    text: str,
    *,
    num_pairs: int = 5,
    doc_title: str = "",
    max_corpus_chars: int = 12000,
) -> list[EvalQAPair]:
    """Use configured LLM to generate QA pairs from corpus text."""
    corpus = text.strip()
    if not corpus:
        raise ValueError("语料不能为空")

    num_pairs = max(1, min(num_pairs, 20))
    llm = get_llm()
    prompt = GENERATE_QA_PROMPT.format(
        num_pairs=num_pairs,
        title=doc_title or "未命名文档",
        text=corpus[:max_corpus_chars],
    )
    ensure_observability_initialized()
    resp = llm.invoke(
        [HumanMessage(content=prompt)],
        config=get_run_config(run_name="qa_generate", tags=["qa", "generate"]),
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    items = _parse_qa_json(content)

    pairs: list[EvalQAPair] = []
    for item in items[:num_pairs]:
        if not isinstance(item, dict) or not item.get("question"):
            continue
        pairs.append(
            EvalQAPair(
                question=str(item["question"]).strip(),
                expected_answer=str(
                    item.get("expected_answer") or item.get("expected_in_chunk") or ""
                ).strip(),
                expected_in_chunk=(
                    str(item["expected_in_chunk"]).strip()
                    if item.get("expected_in_chunk")
                    else None
                ),
            )
        )
    if not pairs:
        raise ValueError("未能从模型输出解析出有效 QA 对")
    return pairs

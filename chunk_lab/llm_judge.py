"""LLM-as-Judge for retrieval / chunk quality evaluation."""

import json
import re
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from chunk_lab.config import Settings, get_settings
from chunk_lab.models import get_llm
from chunk_lab.observability import ensure_observability_initialized, get_run_config

JUDGE_PROMPT = """你是一个 RAG 检索质量评审员（LLM-as-Judge）。

请根据「问题」「参考答案」「检索到的上下文」，判断上下文是否足以支持回答该问题，且与参考答案一致或等价。

## 问题
{question}

## 参考答案
{expected_answer}

## 检索上下文（Top-{top_k}）
{context}

## 评判标准
1. **相关性**：上下文是否与问题主题相关
2. **充分性**：是否包含回答所需的关键事实
3. **一致性**：上下文是否支持参考答案（允许同义表述，不要求字面一致）

仅输出 JSON，不要其他内容：
{{"pass": true, "score": 0.85, "reason": "一句话说明"}}

其中 pass 为布尔值，score 为 0.0~1.0 的浮点数。"""


@dataclass
class JudgeVerdict:
    pass_: bool
    score: float
    reason: str
    raw: str


def get_judge_llm(settings: Settings | None = None) -> BaseChatModel:
    """Judge LLM — defaults to eval judge config, falls back to chunk LLM."""
    cfg = settings or get_settings()
    overrides: dict = {
        "temperature": cfg.eval_judge_temperature,
        "max_tokens": cfg.eval_judge_max_tokens,
    }
    if cfg.eval_judge_llm_model:
        overrides["model"] = cfg.eval_judge_llm_model
    if cfg.eval_judge_llm_provider:
        overrides["model_provider"] = cfg.eval_judge_llm_provider
    if cfg.eval_judge_llm_api_base:
        overrides["api_base"] = cfg.eval_judge_llm_api_base
    if cfg.eval_judge_llm_api_key:
        overrides["api_key"] = cfg.eval_judge_llm_api_key
    return get_llm(cfg, **overrides)


def _parse_verdict(raw: str) -> JudgeVerdict:
    text = raw.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return JudgeVerdict(pass_=False, score=0.0, reason="parse_error", raw=raw)
        data = json.loads(match.group())

    passed = bool(data.get("pass", data.get("passed", False)))
    score = float(data.get("score", 1.0 if passed else 0.0))
    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", ""))
    return JudgeVerdict(pass_=passed, score=score, reason=reason, raw=raw)


def judge_retrieval(
    question: str,
    expected_answer: str,
    context: str,
    *,
    llm: BaseChatModel | None = None,
    top_k: int = 5,
    settings: Settings | None = None,
) -> JudgeVerdict:
    """Run LLM-as-Judge on retrieved context vs expected answer."""
    model = llm or get_judge_llm(settings)
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        context=context[:12000] if context else "(empty)",
        top_k=top_k,
    )
    ensure_observability_initialized()
    resp = model.invoke(
        [HumanMessage(content=prompt)],
        config=get_run_config(run_name="llm_judge", tags=["eval", "judge"]),
    )
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    return _parse_verdict(content)


def format_retrieved_context(chunks: list, *, max_chars: int = 10000) -> str:
    """Format retrieved documents for judge prompt."""
    parts: list[str] = []
    total = 0
    for i, doc in enumerate(chunks):
        block = f"[Chunk {i + 1}]\n{doc.page_content}"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(block[:remaining] + "...")
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts) if parts else "(no context retrieved)"

"""Retrieval evaluation: substring recall@k (default) + optional LLM-as-Judge."""

import uuid
from typing import Literal

from langchain_core.documents import Document

from chunk_lab.config import Settings, get_settings
from chunk_lab.llm_judge import format_retrieved_context, judge_retrieval
from chunk_lab.models import get_embeddings
from chunk_lab.pipeline import run_chunk
from chunk_lab.stats import stats_to_dict
from chunk_lab.types import EvalQAPair, EvalResponse
from chunk_lab.vectorstore import create_vectorstore, similarity_search

EvalMode = Literal["substring", "judge", "both"]


_VALID_EVAL_MODES = frozenset({"substring", "judge", "both"})


def resolve_eval_mode(
    settings: Settings | None = None,
    *,
    override: str | None = None,
    llm_judge_flag: bool | None = None,
) -> EvalMode:
    """Resolve eval mode from env, CLI flag, or API override."""
    cfg = settings or get_settings()

    normalized = (override or "").strip().lower()
    if normalized in _VALID_EVAL_MODES:
        return normalized  # type: ignore[return-value]

    enabled = llm_judge_flag if llm_judge_flag is not None else cfg.eval_judge_enabled
    if not enabled:
        return "substring"

    # 评测面板显式开启 LLM Judge 时默认 both，便于对比「子串 Recall」与「Judge 通过率」
    if llm_judge_flag is True and not normalized:
        return "both"

    mode = (cfg.eval_judge_mode or "both").lower()
    if mode in _VALID_EVAL_MODES:
        return mode  # type: ignore[return-value]
    return "both"


def _substring_hit(qa: EvalQAPair, docs: list[Document]) -> bool:
    needle = (qa.expected_in_chunk or qa.expected_answer or "").strip().lower()
    if not needle:
        return False
    return any(needle in d.page_content.lower() for d in docs)


def _build_vectorstore(chunks: list[Document], strategy_name: str, settings: Settings):
    embeddings = get_embeddings(settings)
    vs = create_vectorstore(
        chunks,
        embeddings,
        settings=settings,
        collection_name=f"eval_{strategy_name}_{uuid.uuid4().hex[:8]}",
    )
    return vs, embeddings


def _persist_faiss_index(
    vs,
    *,
    workspace_id: str,
    strategy_name: str,
    doc_title: str,
    run_name: str = "",
    chunk_count: int,
    settings: Settings,
    stats: dict | None = None,
) -> str | None:
    """Save FAISS vectorstore to disk and record in DB. Returns index_id or None."""
    from langchain_community.vectorstores import FAISS

    if not isinstance(vs, FAISS):
        return None

    from chunk_lab.db import save_vector_index

    index_id, disk_path = save_vector_index(
        workspace_id,
        strategy=strategy_name,
        doc_title=doc_title,
        run_name=run_name,
        chunk_count=chunk_count,
        embedding_provider=settings.chunk_embedding_provider,
        embedding_model=settings.chunk_embedding_model,
        stats=stats,
    )
    vs.save_local(str(disk_path))
    return index_id


def _eval_chunks(
    chunks: list[Document],
    strategy_name: str,
    qa_pairs: list[EvalQAPair],
    *,
    top_k: int,
    eval_mode: EvalMode = "substring",
    doc_title: str = "",
    run_name: str = "",
    workspace_id: str = "default",
    chunk_stats: dict | None = None,
) -> EvalResponse:
    if not chunks:
        return EvalResponse(
            strategy=strategy_name,
            eval_mode=eval_mode,
            recall_at_k=0.0,
            hits=0,
            total=len(qa_pairs),
            details=[],
        )

    settings = get_settings()
    vs, _embeddings = _build_vectorstore(chunks, strategy_name, settings)

    run_substring = eval_mode in ("substring", "both")
    run_judge = eval_mode in ("judge", "both")

    substring_hits = 0
    judge_hits = 0
    judge_scores: list[float] = []
    details: list[dict] = []

    for qa in qa_pairs:
        docs = similarity_search(vs, qa.question, top_k)
        expected = qa.expected_answer or qa.expected_in_chunk or ""

        sub_found = _substring_hit(qa, docs) if run_substring else None
        if run_substring and sub_found:
            substring_hits += 1

        judge_pass = None
        judge_score = None
        judge_reason = None
        if run_judge:
            context = format_retrieved_context(docs)
            verdict = judge_retrieval(
                qa.question,
                expected,
                context,
                top_k=top_k,
                settings=settings,
            )
            judge_pass = verdict.pass_
            judge_score = verdict.score
            judge_reason = verdict.reason
            judge_scores.append(verdict.score)
            if verdict.pass_:
                judge_hits += 1

        details.append(
            {
                "question": qa.question,
                "substring_found": sub_found,
                "judge_pass": judge_pass,
                "judge_score": judge_score,
                "judge_reason": judge_reason,
                "top_preview": docs[0].page_content[:200] if docs else "",
            }
        )

    total = len(qa_pairs)
    denom = total or 1
    recall = round(substring_hits / denom, 4) if run_substring else 0.0
    avg_judge = round(sum(judge_scores) / len(judge_scores), 4) if judge_scores else None
    judge_rate = round(judge_hits / denom, 4) if run_judge else None

    display_recall = recall if run_substring else (judge_rate or 0.0)

    # Persist FAISS index
    index_id = _persist_faiss_index(
        vs,
        workspace_id=workspace_id,
        strategy_name=strategy_name,
        doc_title=doc_title,
        run_name=run_name,
        chunk_count=len(chunks),
        settings=settings,
        stats=chunk_stats,
    )

    # Persist eval run
    from chunk_lab.db import save_eval_run, update_vector_index_eval

    run_id = save_eval_run(
        workspace_id,
        strategy=strategy_name,
        eval_mode=eval_mode,
        recall_at_k=display_recall,
        hits=substring_hits if run_substring else judge_hits,
        total=total,
        top_k=top_k,
        doc_title=doc_title,
        run_name=run_name,
        judge_score=avg_judge,
        judge_hits=judge_hits if run_judge else None,
        judge_pass_rate=judge_rate,
        details=details,
        index_id=index_id,
    )

    # Link index back to eval run
    if index_id:
        update_vector_index_eval(index_id, run_id)

    return EvalResponse(
        strategy=strategy_name,
        eval_mode=eval_mode,
        recall_at_k=display_recall,
        hits=substring_hits if run_substring else judge_hits,
        total=total,
        judge_score=avg_judge,
        judge_hits=judge_hits if run_judge else None,
        judge_pass_rate=judge_rate,
        details=details,
        run_id=run_id,
        run_name=run_name,
        index_id=index_id,
    )


def evaluate_retrieval_on_documents(
    source_docs: list[Document],
    strategy_name: str,
    qa_pairs: list[EvalQAPair],
    *,
    top_k: int | None = None,
    eval_mode: EvalMode | None = None,
    llm_judge: bool | None = None,
    run_name: str = "",
    workspace_id: str = "default",
) -> EvalResponse:
    """Chunk pre-loaded documents then run evaluation."""
    settings = get_settings()
    k = top_k or settings.eval_top_k
    mode = eval_mode or resolve_eval_mode(settings, llm_judge_flag=llm_judge)

    from chunk_lab.registry import get_strategy

    strategy = get_strategy(strategy_name)
    chunks = strategy.chunk(source_docs)
    chunk_stats = stats_to_dict(strategy.analyze(chunks))

    return _eval_chunks(
        chunks, strategy_name, qa_pairs,
        top_k=k, eval_mode=mode,
        doc_title=source_docs[0].metadata.get("title", "") if source_docs else "",
        run_name=run_name,
        workspace_id=workspace_id,
        chunk_stats=chunk_stats,
    )


def evaluate_retrieval(
    text: str,
    strategy_name: str,
    qa_pairs: list[EvalQAPair],
    *,
    top_k: int | None = None,
    title: str = "",
    eval_mode: EvalMode | None = None,
    llm_judge: bool | None = None,
    run_name: str = "",
    workspace_id: str = "default",
) -> EvalResponse:
    settings = get_settings()
    k = top_k or settings.eval_top_k
    mode = eval_mode or resolve_eval_mode(settings, llm_judge_flag=llm_judge)

    run = run_chunk(text, strategy_name, title=title)
    if not run.chunks:
        return EvalResponse(
            strategy=strategy_name,
            eval_mode=mode,
            recall_at_k=0.0,
            hits=0,
            total=len(qa_pairs),
            details=[],
        )

    chunk_stats = stats_to_dict(run.stats)
    return _eval_chunks(
        run.chunks, strategy_name, qa_pairs,
        top_k=k, eval_mode=mode,
        doc_title=title,
        run_name=run_name,
        workspace_id=workspace_id,
        chunk_stats=chunk_stats,
    )

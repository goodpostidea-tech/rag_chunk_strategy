"""Retrieval experiment routes — mounted at /retrieval on chunk_api."""

from fastapi import APIRouter, HTTPException

from chunk_lab.config import get_settings
from chunk_lab.datasources.local_dir import load_local_corpus_text
from chunk_lab.qa_store import resolve_qa_pairs_by_source
from retrieval_lab.pipeline import (
    compare_retrieval_methods,
    evaluate_retrieval_experiment,
    retrieval_response_to_dict,
)
from retrieval_lab.registry import list_retrieval_methods
from retrieval_lab.types import RetrievalCompareRequest, RetrievalEvalRequest, RetrievalQAPair

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


def _resolve_text(
    text: str | None,
    local_dir: str | None,
    local_recursive: bool = True,
    local_file_types: list[str] | None = None,
) -> str:
    if text:
        return text
    if local_dir:
        corpus_text, _ = load_local_corpus_text(
            local_dir, recursive=local_recursive, file_types=local_file_types
        )
        return corpus_text
    raise ValueError("需要提供 text 或 local_dir")


def _resolve_qa(req_qa: list[dict] | None, qa_source: str) -> list[RetrievalQAPair]:
    if req_qa:
        return [RetrievalQAPair.from_dict(item) for item in req_qa]
    pairs, _ = resolve_qa_pairs_by_source(qa_source)
    return [
        RetrievalQAPair(
            question=p.question,
            expected_answer=p.expected_answer,
            expected_in_chunk=p.expected_in_chunk,
            query_type=getattr(p, "query_type", "factual"),  # type: ignore[arg-type]
        )
        for p in pairs
    ]


@router.get("/health")
def retrieval_health():
    cfg = get_settings()
    return {
        "status": "ok",
        "service": "retrieval_lab",
        "embedding": f"{cfg.chunk_embedding_provider}:{cfg.chunk_embedding_model}",
    }


@router.get("/methods")
def methods_list():
    return {"methods": list_retrieval_methods()}


@router.post("/eval")
def retrieval_eval(req: RetrievalEvalRequest):
    try:
        text = _resolve_text(
            req.text, req.local_dir, req.local_recursive, req.local_file_types
        )
        qa = _resolve_qa(req.qa_pairs, req.qa_source)
        cfg = get_settings()
        resp = evaluate_retrieval_experiment(
            text,
            req.retrieval_method,
            qa,
            chunk_strategy=req.chunk_strategy,
            top_k=req.top_k or cfg.eval_top_k,
            doc_title=req.doc_title,
            retrieval_params=req.retrieval_params,
            chunk_params=req.chunk_params,
        )
        return retrieval_response_to_dict(resp)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/compare")
def retrieval_compare(req: RetrievalCompareRequest):
    try:
        text = _resolve_text(
            req.text, req.local_dir, req.local_recursive, req.local_file_types
        )
        qa = _resolve_qa(req.qa_pairs, req.qa_source)
        cfg = get_settings()
        results = compare_retrieval_methods(
            text,
            qa,
            methods=req.methods,
            chunk_strategy=req.chunk_strategy,
            top_k=req.top_k or cfg.eval_top_k,
        )
        return {"results": [retrieval_response_to_dict(r) for r in results]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

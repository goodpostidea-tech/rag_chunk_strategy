"""FastAPI for dynamic chunk strategy testing."""

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chunk_lab.config import get_settings
from chunk_lab.models import embedding_info
from chunk_lab.vectorstore import vectorstore_info
from chunk_lab.datasources.corpus import (
    build_corpus_documents,
    fetch_paper,
    fetch_papers,
    list_papers,
    load_corpus_text,
)
from chunk_lab.evaluator import evaluate_retrieval, evaluate_retrieval_on_documents, resolve_eval_mode
from chunk_lab.observability import observability_info, setup_observability, shutdown_observability
from chunk_lab.pipeline import chunks_to_response_dict, compare_strategies, run_chunk
from chunk_lab.registry import list_strategies
from chunk_lab.settings_schema import SETTING_GROUPS
from chunk_lab.settings_store import (
    get_overlay,
    reset_overlay,
    settings_to_public_dict,
    update_overlay,
)
from chunk_lab.qa_generator import generate_qa_from_corpus
from chunk_lab.qa_store import (
    dataset_summary,
    load_active_qa_pairs,
    load_builtin_qa_pairs,
    load_custom_qa_meta,
    reset_custom_qa_dataset,
    resolve_qa_pairs_by_source,
    save_custom_qa_dataset,
)
from chunk_lab.rag_chat import chat_with_session, get_session_store
from chunk_lab.routers import (
    parse_lab_router,
    retrieval_lab_router,
    vstore_lab_router,
)
from chunk_lab.types import (
    ChunkRequest,
    ChunkResponse,
    CompareRequest,
    CompareResponse,
    EvalQAPair,
    EvalRequest,
    EvalResponse,
    QADatasetPayload,
    QADatasetResponse,
    QAGenerateRequest,
    QAGenerateResponse,
    RagChatRequest,
    RagChatResponse,
    RagIndexRequest,
    RagIndexResponse,
    SettingsUpdateRequest,
    TestConnectionRequest,
    WorkspaceCreate,
    WorkspaceResponse,
    EvalHistoryItem,
    VectorIndexInfo,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    from chunk_lab.db import init_db
    init_db()
    setup_observability()
    yield
    shutdown_observability()


app = FastAPI(
    title="RAG Chunk Strategy Lab",
    description="Compare LangChain dynamic chunk strategies (default LLM: DeepSeek)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    cfg = get_settings()
    return {
        "status": "ok",
        "llm": f"{cfg.chunk_llm_provider}:{cfg.chunk_llm_model}",
        "embedding": embedding_info(cfg),
        "vector_store": vectorstore_info(cfg),
        "observability": observability_info(cfg),
        "experiments": {
            "parse": "/parse",
            "retrieval": "/retrieval",
            "vstore": "/vstore",
        },
    }


app.include_router(parse_lab_router)
app.include_router(retrieval_lab_router)
app.include_router(vstore_lab_router)


@app.get("/config")
def config():
    """Current embedding & vector store configuration."""
    cfg = get_settings()
    return {
        "embedding": embedding_info(cfg),
        "vector_store": vectorstore_info(cfg),
        "eval": {
            "judge_enabled": cfg.eval_judge_enabled,
            "judge_mode": cfg.eval_judge_mode,
            "resolved_mode": resolve_eval_mode(cfg),
            "judge_llm": cfg.eval_judge_llm_model or cfg.chunk_llm_model,
        },
        "observability": observability_info(cfg),
    }


@app.get("/settings/schema")
def settings_schema():
    return {"groups": SETTING_GROUPS}


@app.get("/settings")
def get_settings_api(workspace: str = "default"):
    from chunk_lab.settings_store import get_effective_settings
    cfg = get_effective_settings(workspace)
    return {
        "values": settings_to_public_dict(cfg),
        "overlay_active": bool(get_overlay(workspace)),
        "groups": SETTING_GROUPS,
        "workspace": workspace,
        "summary": {
            "llm": f"{cfg.chunk_llm_provider}:{cfg.chunk_llm_model}",
            "embedding": f"{cfg.chunk_embedding_provider}:{cfg.chunk_embedding_model}",
            "vector_store": cfg.vector_store_provider,
            "vision": f"{cfg.parse_vlm_provider}:{cfg.parse_vlm_model}",
            "ocr_api": f"{cfg.parse_ocr_api_provider}:{cfg.parse_ocr_api_model}",
            "langsmith": cfg.langsmith_enabled,
            "langfuse": cfg.langfuse_enabled,
        },
    }


@app.put("/settings")
def put_settings(req: SettingsUpdateRequest, workspace: str = "default"):
    try:
        cfg = update_overlay(req.values, workspace)
        setup_observability(cfg)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "values": settings_to_public_dict(cfg),
        "workspace": workspace,
        "message": "配置已应用并持久化到数据库",
    }


@app.post("/settings/reset")
def reset_settings_api(workspace: str = "default"):
    cfg = reset_overlay(workspace)
    setup_observability(cfg)
    return {"ok": True, "values": settings_to_public_dict(cfg), "workspace": workspace}


@app.post("/settings/test-connection")
def test_connection(req: TestConnectionRequest):
    """Test connectivity to an LLM provider with given credentials."""
    try:
        headers = {"Content-Type": "application/json"}
        if req.api_key:
            headers["Authorization"] = f"Bearer {req.api_key}"

        # Determine endpoint from provider
        base = req.api_base or {
            "deepseek": "https://api.deepseek.com",
            "openai": "https://api.openai.com/v1",
            "mistral": "https://api.mistral.ai/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }.get(req.provider)

        if not base:
            return {"ok": False, "message": f"未知提供商: {req.provider}，请填写 API Base"}

        url = f"{base.rstrip('/')}/models"
        if "localhost" in base or "127.0.0.1" in base:
            url = f"{base.rstrip('/')}/v1/models"

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers)

        if resp.status_code == 200:
            return {"ok": True, "message": f"连接成功（{req.provider}: {req.model}）"}
        elif resp.status_code == 401:
            return {"ok": False, "message": "API Key 无效（401 Unauthorized）"}
        else:
            return {"ok": False, "message": f"连接异常（HTTP {resp.status_code}）"}
    except httpx.ConnectError:
        return {"ok": False, "message": "地址不可达，请检查 API Base URL 或网络连接"}
    except httpx.TimeoutException:
        return {"ok": False, "message": "连接超时（10 秒），请检查 API Base URL"}
    except Exception as e:
        return {"ok": False, "message": f"连接测试失败: {str(e)}"}


@app.get("/strategies")
def strategies():
    return {"strategies": list_strategies()}


def _resolve_text(
    text: str | None,
    local_dir: str | None,
    local_recursive: bool = True,
    local_file_types: list[str] | None = None,
) -> str:
    """Resolve corpus text from either direct text or local directory."""
    if text:
        return text
    if local_dir:
        from chunk_lab.datasources.local_dir import load_local_corpus_text
        corpus_text, _ = load_local_corpus_text(
            local_dir, recursive=local_recursive, file_types=local_file_types
        )
        return corpus_text
    raise ValueError("需要提供 text 或 local_dir")


@app.post("/chunk", response_model=ChunkResponse)
def chunk(req: ChunkRequest):
    try:
        text = _resolve_text(req.text, req.local_dir, req.local_recursive, req.local_file_types)
        result = run_chunk(
            text,
            req.strategy,
            title=req.doc_title,
            section=req.doc_section,
            strategy_params=req.strategy_params,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    data = chunks_to_response_dict(result, max_preview=10)
    return ChunkResponse(
        strategy=data["strategy"],
        category=data["category"],
        stats=data["stats"],
        elapsed_ms=data["elapsed_ms"],
        chunks=data["preview"],
    )


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    try:
        text = _resolve_text(req.text, req.local_dir, req.local_recursive, req.local_file_types)
        results = compare_strategies(
            text,
            req.strategies,
            title=req.doc_title,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return CompareResponse(
        results=[
            {
                **chunks_to_response_dict(r, max_preview=2),
                "error": r.category == "error",
            }
            for r in results
        ]
    )


def _resolve_eval_qa_pairs(req: EvalRequest) -> list:
    if req.qa_pairs:
        return req.qa_pairs
    pairs, _ = resolve_qa_pairs_by_source(req.qa_source)
    return pairs


@app.post("/eval", response_model=EvalResponse)
def eval_retrieval(req: EvalRequest, workspace: str = "default"):
    try:
        text = _resolve_text(req.text, req.local_dir, req.local_recursive, req.local_file_types)
        qa_pairs = _resolve_eval_qa_pairs(req)
        return evaluate_retrieval(
            text,
            req.strategy,
            qa_pairs,
            top_k=req.top_k,
            title=req.doc_title,
            eval_mode=req.eval_mode,  # type: ignore[arg-type]
            llm_judge=req.llm_judge,
            run_name=req.run_name,
            workspace_id=workspace,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- RAG chat (indexed vector store) ---


@app.post("/rag/index", response_model=RagIndexResponse)
def rag_index(req: RagIndexRequest):
    """Chunk corpus, embed, and create an in-memory chat session.
    Alternatively, pass index_id to load a persisted FAISS index."""
    try:
        if req.index_id:
            session = get_session_store().create_from_index(
                req.index_id,
                session_id=req.session_id,
            )
        else:
            if not req.text:
                raise ValueError("需要提供 text 或 index_id")
            session = get_session_store().create(
                req.text,
                req.strategy,
                doc_title=req.doc_title,
                session_id=req.session_id,
            )
        return RagIndexResponse(
            session_id=session.session_id,
            strategy=session.strategy,
            chunk_count=session.chunk_count,
            stats=session.stats,
            doc_title=session.doc_title,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rag/chat", response_model=RagChatResponse)
def rag_chat(req: RagChatRequest):
    """Chat against a previously indexed session (retrieve + LLM answer).
    If session doesn't exist but index_id is provided, auto-create from persisted index."""
    try:
        store = get_session_store()
        session = store.get(req.session_id)
        if not session and req.index_id:
            session = store.create_from_index(
                req.index_id,
                session_id=req.session_id,
            )
        if not session:
            raise ValueError(f"会话不存在或已过期: {req.session_id}")
        data = chat_with_session(req.session_id, req.message, top_k=req.top_k)
        return RagChatResponse(**data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/rag/sessions")
def rag_list_sessions():
    return {"sessions": get_session_store().list_sessions()}


@app.delete("/rag/sessions/{session_id}")
def rag_delete_session(session_id: str):
    if not get_session_store().delete(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True, "session_id": session_id}


# --- QA dataset (custom / builtin / AI generate) ---


@app.get("/qa/dataset/summary")
def qa_dataset_summary():
    return dataset_summary()


@app.get("/qa/dataset", response_model=QADatasetResponse)
def qa_get_active_dataset():
    pairs, source = load_active_qa_pairs()
    meta = load_custom_qa_meta() if source == "custom" else None
    return QADatasetResponse(
        source=source,
        name=meta.get("name") if meta else ("内置参考文献 QA" if source == "builtin" else None),
        updated_at=meta.get("updated_at") if meta else None,
        qa_pairs=pairs,
        count=len(pairs),
    )


@app.get("/qa/dataset/builtin", response_model=QADatasetResponse)
def qa_get_builtin_dataset():
    pairs = load_builtin_qa_pairs()
    return QADatasetResponse(
        source="builtin",
        name="内置参考文献 QA",
        qa_pairs=pairs,
        count=len(pairs),
    )


@app.get("/qa/dataset/custom", response_model=QADatasetResponse)
def qa_get_custom_dataset():
    meta = load_custom_qa_meta()
    if not meta:
        raise HTTPException(status_code=404, detail="尚未保存用户自定义 QA 测试集")
    pairs = [EvalQAPair(**p) for p in meta["qa_pairs"]]
    return QADatasetResponse(
        source="custom",
        name=meta.get("name"),
        updated_at=meta.get("updated_at"),
        qa_pairs=pairs,
        count=len(pairs),
    )


@app.put("/qa/dataset/custom", response_model=QADatasetResponse)
def qa_save_custom_dataset(payload: QADatasetPayload):
    if not payload.qa_pairs:
        raise HTTPException(status_code=400, detail="qa_pairs 不能为空")
    meta = save_custom_qa_dataset(payload.qa_pairs, name=payload.name)
    return QADatasetResponse(
        source="custom",
        name=meta["name"],
        updated_at=meta["updated_at"],
        qa_pairs=payload.qa_pairs,
        count=len(payload.qa_pairs),
    )


@app.post("/qa/dataset/custom/reset")
def qa_reset_custom_dataset():
    reset_custom_qa_dataset()
    pairs, source = load_active_qa_pairs()
    return {
        "ok": True,
        "message": "已删除自定义测试集，评测将使用内置集",
        "active_source": source,
        "active_count": len(pairs),
    }


@app.post("/qa/generate", response_model=QAGenerateResponse)
def qa_generate(req: QAGenerateRequest):
    """Use LLM to generate QA pairs from user corpus; optionally save as custom dataset."""
    try:
        pairs = generate_qa_from_corpus(
            req.text,
            num_pairs=req.num_pairs,
            doc_title=req.doc_title,
        )
        saved = False
        if req.save_as_custom:
            save_custom_qa_dataset(pairs, name=req.dataset_name)
            saved = True
        return QAGenerateResponse(
            qa_pairs=pairs,
            count=len(pairs),
            saved=saved,
            dataset_name=req.dataset_name if saved else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/papers")
def papers_list(arxiv_only: bool = False):
    return {"papers": list_papers(arxiv_only=arxiv_only)}


@app.get("/papers/corpus-text")
def papers_corpus_text(paper_ids: str | None = None, fetch_if_missing: bool = True):
    """Load concatenated paper text for chunk/compare in the web lab."""
    ids = [s.strip() for s in paper_ids.split(",")] if paper_ids else None
    try:
        text = load_corpus_text(ids, fetch_if_missing=fetch_if_missing)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "text": text,
        "char_count": len(text),
        "paper_ids": ids,
    }


@app.get("/papers/qa")
def papers_qa():
    """Legacy alias — returns active QA dataset (custom if set, else builtin)."""
    pairs, source = load_active_qa_pairs()
    return {"source": source, "qa_pairs": [p.model_dump() for p in pairs]}


@app.post("/papers/fetch")
def papers_fetch(
    paper_id: str | None = None,
    all_arxiv: bool = False,
    full_text: bool = False,
    force: bool = False,
):
    try:
        if paper_id:
            path = fetch_paper(paper_id, full_text=full_text, force=force)
            return {"results": [{"id": paper_id, "status": "ok", "path": str(path)}]}
        return {
            "results": fetch_papers(
                None,
                all_arxiv=all_arxiv,
                full_text=full_text,
                force=force,
            )
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/papers/chunk", response_model=ChunkResponse)
def papers_chunk(
    strategy: str = "recursive_baseline",
    paper_ids: str | None = None,
    fetch_if_missing: bool = True,
):
    try:
        ids = [s.strip() for s in paper_ids.split(",")] if paper_ids else None
        if ids and len(ids) == 1:
            doc = build_corpus_documents(ids, fetch_if_missing=fetch_if_missing)[0]
            text = doc.page_content
            title = doc.metadata.get("title", ids[0])
        else:
            text = load_corpus_text(ids, fetch_if_missing=fetch_if_missing)
            title = "references_corpus"
        result = run_chunk(text, strategy, title=title)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    data = chunks_to_response_dict(result, max_preview=10)
    return ChunkResponse(
        strategy=data["strategy"],
        category=data["category"],
        stats=data["stats"],
        elapsed_ms=data["elapsed_ms"],
        chunks=data["preview"],
    )


@app.post("/papers/eval", response_model=EvalResponse)
def papers_eval(
    strategy: str = "recursive_baseline",
    paper_ids: str | None = None,
    top_k: int | None = None,
    fetch_if_missing: bool = True,
    llm_judge: bool = False,
    eval_mode: str | None = None,
    qa_pairs: list | None = None,
    run_name: str = "",
    workspace: str = "default",
):
    from pathlib import Path

    import json

    from chunk_lab.types import EvalQAPair

    if qa_pairs is None:
        from chunk_lab.qa_store import load_builtin_qa_pairs

        qa_pairs = load_builtin_qa_pairs()
    else:
        qa_pairs = [
            EvalQAPair.from_dict(q) if isinstance(q, dict) else q for q in qa_pairs
        ]

    ids = [s.strip() for s in paper_ids.split(",")] if paper_ids else None
    try:
        docs = build_corpus_documents(ids, fetch_if_missing=fetch_if_missing)
        return evaluate_retrieval_on_documents(
            docs,
            strategy,
            qa_pairs,
            top_k=top_k,
            eval_mode=eval_mode,  # type: ignore[arg-type]
            llm_judge=llm_judge or None,
            run_name=run_name,
            workspace_id=workspace,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- Local directory corpus ---


@app.get("/local/files")
def local_files(path: str, recursive: bool = True, file_types: str | None = None):
    """List supported files in a local directory or describe a single file."""
    from pathlib import Path as _Path

    from chunk_lab.datasources.local_dir import SUPPORTED_EXTENSIONS, list_local_files

    types = [s.strip() for s in file_types.split(",")] if file_types else None
    try:
        p = _Path(path).resolve()
        if p.is_file():
            ext = p.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"不支持的文件格式: {ext}")
            files = [{"path": str(p), "name": p.name, "ext": ext, "size": p.stat().st_size}]
        else:
            files = list_local_files(path, recursive=recursive, file_types=types)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "path": path,
        "files": files,
        "count": len(files),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
    }


# --- Workspace management ---


@app.get("/workspaces")
def workspaces_list():
    from chunk_lab.db import list_workspaces
    return {"workspaces": list_workspaces()}


@app.post("/workspaces", response_model=WorkspaceResponse)
def workspaces_create(req: WorkspaceCreate):
    from chunk_lab.db import create_workspace, get_workspace
    if get_workspace(req.id):
        raise HTTPException(status_code=409, detail=f"Workspace '{req.id}' 已存在")
    ws = create_workspace(req.id, req.name)
    return WorkspaceResponse(**ws)


@app.delete("/workspaces/{workspace_id}")
def workspaces_delete(workspace_id: str):
    from chunk_lab.db import delete_workspace
    try:
        delete_workspace(workspace_id)
        return {"ok": True, "workspace_id": workspace_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# --- Eval preflight (config check before running eval) ---


@app.get("/eval/preflight")
def eval_preflight(workspace: str = "default", llm_judge: bool | None = None):
    """Return the effective config that will be used for evaluation, with connectivity check."""
    from chunk_lab.evaluator import resolve_eval_mode
    from chunk_lab.settings_store import get_effective_settings

    cfg = get_effective_settings(workspace)
    resolved_mode = resolve_eval_mode(cfg, llm_judge_flag=llm_judge)
    judge_active = resolved_mode in ("judge", "both")

    # Check embedding availability
    embedding_status = "ok"
    embedding_error = None
    try:
        from chunk_lab.models import get_embeddings
        emb = get_embeddings(cfg)
        emb.embed_query("test")
    except Exception as e:
        embedding_status = "error"
        embedding_error = str(e)

    return {
        "workspace": workspace,
        "llm": {
            "provider": cfg.chunk_llm_provider,
            "model": cfg.chunk_llm_model,
            "api_base": cfg.chunk_llm_api_base or "(default)",
            "has_api_key": bool(cfg.chunk_llm_api_key),
        },
        "embedding": {
            "provider": cfg.chunk_embedding_provider,
            "model": cfg.chunk_embedding_model,
            "api_base": cfg.chunk_embedding_api_base or "(local)" if cfg.chunk_embedding_provider == "huggingface" else cfg.chunk_embedding_api_base or "(default)",
            "device": cfg.embedding_device if cfg.chunk_embedding_provider == "huggingface" else None,
            "has_api_key": bool(cfg.chunk_embedding_api_key),
            "status": embedding_status,
            "error": embedding_error,
        },
        "vector_store": {
            "provider": cfg.vector_store_provider,
            "collection_prefix": cfg.vector_store_collection_prefix,
        },
        "eval": {
            "top_k": cfg.eval_top_k,
            "judge_enabled": judge_active,
            "judge_mode": resolved_mode if judge_active else None,
            "judge_llm": (cfg.eval_judge_llm_model or cfg.chunk_llm_model) if judge_active else None,
            "llm_judge_request": llm_judge,
        },
    }


# --- Eval history ---


@app.get("/eval/tasks")
def eval_tasks(workspace: str = "default"):
    """List eval tasks (grouped by run_name) with summary stats."""
    from chunk_lab.db import list_eval_run_names
    tasks = list_eval_run_names(workspace)
    return {"tasks": tasks, "workspace": workspace}


@app.get("/eval/history")
def eval_history(
    workspace: str = "default",
    strategy: str | None = None,
    run_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
):
    from chunk_lab.db import list_eval_runs
    runs = list_eval_runs(workspace, strategy=strategy, run_name=run_name, limit=limit, offset=offset)
    return {"runs": runs, "workspace": workspace}


@app.get("/eval/history/{run_id}")
def eval_history_detail(run_id: str):
    from chunk_lab.db import get_eval_run
    run = get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return run


@app.delete("/eval/history/{run_id}")
def eval_history_delete(run_id: str):
    from chunk_lab.db import delete_eval_run
    if not delete_eval_run(run_id):
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return {"ok": True, "run_id": run_id}


# --- Vector indexes ---


@app.get("/indexes")
def indexes_list(workspace: str = "default"):
    from chunk_lab.db import list_vector_indexes
    return {"indexes": list_vector_indexes(workspace), "workspace": workspace}


@app.delete("/indexes")
def indexes_clear(workspace: str = "default"):
    """Delete all vector indexes in a workspace."""
    from chunk_lab.db import clear_vector_indexes
    count = clear_vector_indexes(workspace)
    return {"ok": True, "deleted": count, "workspace": workspace}


@app.delete("/indexes/{index_id}")
def indexes_delete(index_id: str):
    from chunk_lab.db import delete_vector_index
    if not delete_vector_index(index_id):
        raise HTTPException(status_code=404, detail="索引不存在")
    return {"ok": True, "index_id": index_id}


# --- Web UI (rag_chunk_tui build output) ---
_TUI_DIST = Path(__file__).resolve().parent.parent / "rag_chunk_tui" / "dist"
if _TUI_DIST.is_dir() and (_TUI_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_TUI_DIST / "assets"), name="tui-assets")


@app.get("/")
def root():
    index = _TUI_DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {
        "message": "RAG Chunk Lab API",
        "docs": "/docs",
        "ui_dev": "cd rag_chunk_tui && npm run dev → http://localhost:5173",
        "ui_build": "cd rag_chunk_tui && npm run build, then reload this server",
    }

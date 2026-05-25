"""Parse experiment routes — mounted at /parse on chunk_api."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from parse_lab.pipeline import compare_to_dict, run_parse, run_parse_batch
from parse_lab.source_preview import (
    docx_preview_html,
    media_type_for_path,
    preview_meta,
    resolve_source_file,
)
from parse_lab.registry import list_parsers
from parse_lab.types import ParseBatchRequest, ParseRunRequest

router = APIRouter(prefix="/parse", tags=["parse"])


@router.get("/health")
def parse_health():
    return {"status": "ok", "service": "parse_lab"}


@router.get("/parsers")
def parsers_list():
    return {"parsers": list_parsers(include_unavailable=True)}


@router.get("/metrics-help")
def parse_metrics_help():
    """解析评测指标说明（与 parse_lab.metrics 及前端 parseMetricsHelp 一致）。"""
    return {
        "relative": True,
        "metrics": [
            {
                "id": "text_completeness",
                "label": "文本完整性",
                "formula": "parser_char_count / max(char_count among successful parsers)",
                "weight_in_rank": 0.5,
            },
            {
                "id": "table_structure_score",
                "label": "表格结构",
                "formula": "table_signal / baseline_tables (max count + markdown pipe rows)",
                "weight_in_rank": 0.25,
            },
            {
                "id": "heading_detection_score",
                "label": "标题检测",
                "formula": "heading_count / baseline_headings (max count + markdown # lines)",
                "weight_in_rank": 0.25,
            },
        ],
        "rank_formula": "0.5*text_completeness + 0.25*table_structure_score + 0.25*heading_detection_score",
        "doc_profile": "Inferred from filename keywords when not overridden; label only.",
    }


@router.get("/original")
def parse_original_file(path: str):
    """返回原文件流，供浏览器内嵌预览（目前 PDF 效果最佳）。"""
    try:
        file_path = resolve_source_file(path)
        return FileResponse(
            path=file_path,
            media_type=media_type_for_path(file_path),
            filename=file_path.name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/original-preview")
def parse_original_preview_meta(path: str):
    """原文件预览方式（PDF 内嵌 / docx HTML），不做解析抽取。"""
    try:
        return preview_meta(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/original-docx-html")
def parse_original_docx_html(path: str):
    """docx 转 HTML 页面，供 iframe 直接预览。"""
    try:
        html = docx_preview_html(path)
        return HTMLResponse(content=html)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/run")
def parse_run(req: ParseRunRequest):
    try:
        result = run_parse(req.file_path, req.parsers, doc_profile=req.doc_profile)
        return compare_to_dict(result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/ocr-status")
def parse_ocr_status():
    """Readiness for ocr_local and ocr_api parsers."""
    from parse_lab.ocr.config import ocr_api_available, ocr_local_available
    from chunk_lab.config import get_settings

    local_ok, local_hint = ocr_local_available()
    api_ok, api_hint = ocr_api_available()
    cfg = get_settings()
    return {
        "ocr_local": {
            "available": local_ok,
            "install_hint": local_hint,
        },
        "ocr_api": {
            "available": api_ok,
            "install_hint": api_hint,
            "provider": cfg.parse_ocr_api_provider,
            "model": cfg.parse_ocr_api_model,
            "has_api_key": bool(cfg.parse_ocr_api_key),
        },
        "max_pages": cfg.parse_ocr_max_pages,
        "dpi": cfg.parse_ocr_dpi,
        "settings_path": "系统配置 → OCR API",
    }


@router.get("/vlm-status")
def parse_vlm_status():
    """Whether vlm_pdf can run with current system settings."""
    from parse_lab.vlm.client import vlm_config_from_settings

    cfg = vlm_config_from_settings()
    configured = bool(
        cfg.model.strip()
        and (cfg.api_key or cfg.provider in ("ollama",))
    )
    return {
        "configured": configured,
        "provider": cfg.provider,
        "model": cfg.model,
        "has_api_key": bool(cfg.api_key),
        "max_pages": cfg.max_pages,
        "settings_path": "系统配置 → 视觉模型",
    }


@router.post("/batch")
def parse_batch(req: ParseBatchRequest):
    from chunk_lab.datasources.local_dir import list_local_files

    try:
        files = list_local_files(
            req.dir_path,
            recursive=req.recursive,
            file_types=req.file_types or [".pdf", ".docx"],
        )
        paths = [f["path"] for f in files]
        if not paths:
            raise ValueError("目录中无匹配的 PDF/docx 文件")
        results = run_parse_batch(paths, req.parsers)
        return {"count": len(results), "results": [compare_to_dict(r) for r in results]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

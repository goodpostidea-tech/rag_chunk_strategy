"""Parse experiment pipeline — run and compare parsers."""

from pathlib import Path

from parse_lab.metrics import (
    compute_parse_metrics,
    infer_doc_profile,
    rank_parsers,
)
from parse_lab.registry import get_parser, list_parsers, parsers_for_file
from parse_lab.types import ParseCompareResult, ParseRunResult
from parse_lab.vlm.client import vlm_config_from_settings
from parse_lab.vlm.context import vlm_config_scope


def _needs_vlm(names: list[str]) -> bool:
    return any(n.strip().lower() == "vlm_pdf" for n in names)


def run_parse(
    file_path: str | Path,
    parser_names: list[str] | None = None,
    *,
    doc_profile: str | None = None,
) -> ParseCompareResult:
    """Run all (or selected) parsers on one file and compute metrics."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = path.suffix.lower()
    candidates = parser_names or parsers_for_file(ext)
    if not candidates:
        raise ValueError(f"无可用解析器支持 {ext}")

    results = []
    errors: list[str] = []
    vlm_cfg = vlm_config_from_settings() if _needs_vlm(candidates) else None

    with vlm_config_scope(vlm_cfg):
        for name in candidates:
            try:
                parser = get_parser(name)
                if not parser.supports(path):
                    continue
                results.append(parser.parse(path))
            except (ImportError, ValueError) as exc:
                errors.append(f"{name}: {exc}")

    if not results:
        raise ValueError(f"所有解析器失败或未安装: {errors}")

    profile = doc_profile or infer_doc_profile(path)
    metrics = compute_parse_metrics(results, file_path=path, doc_profile=profile)
    return ParseCompareResult(
        file_path=str(path),
        doc_profile=profile,
        results=results,
        metrics=metrics,
        ranking=rank_parsers(metrics),
    )


def run_parse_batch(
    paths: list[str | Path],
    parser_names: list[str] | None = None,
) -> list[ParseCompareResult]:
    return [run_parse(p, parser_names) for p in paths]


def run_single_parser(file_path: str | Path, parser_name: str) -> ParseRunResult:
    path = Path(file_path).resolve()
    parser = get_parser(parser_name)
    result = parser.parse(path)
    metrics = compute_parse_metrics(
        [result],
        file_path=path,
        doc_profile=infer_doc_profile(path),
    )
    return ParseRunResult(parser=parser_name, results=[result], metrics=metrics)


# 细节对比弹窗用全文上限（字符），超出部分截断并标记
_COMPARE_TEXT_MAX = 200_000


def compare_to_dict(result: ParseCompareResult) -> dict:
    return {
        "file_path": result.file_path,
        "doc_profile": result.doc_profile,
        "ranking": result.ranking,
        "metrics": [
            {
                "parser": m.parser,
                "text_completeness": m.text_completeness,
                "table_structure_score": m.table_structure_score,
                "heading_detection_score": m.heading_detection_score,
                "char_count": m.char_count,
                "table_count": m.table_count,
                "heading_count": m.heading_count,
                "elapsed_ms": m.elapsed_ms,
                "error": m.error,
            }
            for m in result.metrics
        ],
        "previews": [
            {
                "parser": r.parser,
                "char_count": r.char_count,
                "table_count": r.table_count,
                "heading_count": r.heading_count,
                "elapsed_ms": r.elapsed_ms,
                "error": r.error,
                "text_preview": r.text[:500] if r.text else "",
                "text": (r.text[:_COMPARE_TEXT_MAX] if r.text else ""),
                "text_truncated": bool(r.text and len(r.text) > _COMPARE_TEXT_MAX),
                "extra": r.extra if r.extra else None,
            }
            for r in result.results
        ],
    }


def list_available_parsers() -> list[dict]:
    return list_parsers(include_unavailable=True)

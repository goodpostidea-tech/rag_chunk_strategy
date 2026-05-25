"""LXC Parse experiment metrics — completeness, tables, headings, speed.

指标均为**同文件多解析器横向对比**（非金标准 OCR）：
- text_completeness: char_count / max(char_count)
- table_structure_score: 表格量 / 基准（含 Markdown 管道表行）
- heading_detection_score: 标题数 / 基准（含 Markdown # 标题）
- rank 综合分: 0.5·完整性 + 0.25·表格 + 0.25·标题

前端说明见 rag_chunk_tui/src/config/parseMetricsHelp.ts
"""

import re
from pathlib import Path

from parse_lab.types import ParseMetrics, ParseResult

# Markdown heading: # .. ######
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Markdown pipe table row
_MD_TABLE_ROW = re.compile(r"^\|.+\|$", re.MULTILINE)


def infer_doc_profile(file_path: Path) -> str:
    """Heuristic document profile from filename (override via metadata in fixtures)."""
    name = file_path.stem.lower()
    if "scan" in name or "ocr" in name:
        return "scanned_pdf"
    if "academic" in name or "paper" in name or "arxiv" in name:
        return "academic_pdf"
    if "table" in name or "financial" in name or "report" in name:
        return "table_heavy_pdf"
    if "multi" in name and "col" in name:
        return "multicolumn_pdf"
    if file_path.suffix.lower() == ".docx":
        if "heading" in name or "outline" in name:
            return "docx_structured"
        return "docx_general"
    if file_path.suffix.lower() == ".pdf":
        return "plain_text_pdf"
    return "unknown"


def _count_markdown_headings(text: str) -> int:
    return len(_MD_HEADING.findall(text))


def _count_markdown_table_rows(text: str) -> int:
    return len(_MD_TABLE_ROW.findall(text))


def _table_structure_score(result: ParseResult, baseline_table_rows: int) -> float:
    if baseline_table_rows <= 0:
        return 1.0 if result.table_count == 0 else min(1.0, result.table_count / 3)
    extracted = result.table_count
    if result.tables:
        cells = sum(len(r.rows) * max(len(r.rows[0]), 1) for r in result.tables if r.rows)
        extracted = max(extracted, min(cells // 4, baseline_table_rows))
    return min(1.0, extracted / baseline_table_rows)


def _heading_score(result: ParseResult, baseline_headings: int) -> float:
    if baseline_headings <= 0:
        return 1.0 if result.heading_count == 0 else min(1.0, result.heading_count / 5)
    return min(1.0, result.heading_count / baseline_headings)


def compute_parse_metrics(
    results: list[ParseResult],
    *,
    file_path: Path,
    doc_profile: str | None = None,
) -> list[ParseMetrics]:
    """Compare parsers on the same file; scores are relative to the best run."""
    profile = doc_profile or infer_doc_profile(file_path)
    ok = [r for r in results if not r.error]
    if not ok:
        return [
            ParseMetrics(
                parser=r.parser,
                file_path=str(file_path),
                doc_profile=profile,
                text_completeness=0.0,
                table_structure_score=0.0,
                heading_detection_score=0.0,
                char_count=0,
                table_count=0,
                heading_count=0,
                elapsed_ms=r.elapsed_ms,
                error=r.error,
            )
            for r in results
        ]

    max_chars = max(r.char_count for r in ok)
    max_tables = max(r.table_count for r in ok)
    max_headings = max(r.heading_count for r in ok)

    # pdfplumber often best for tables — use max table rows in markdown as baseline
    baseline_table_rows = max(
        max_tables,
        max(_count_markdown_table_rows(r.text) for r in ok),
    )
    baseline_headings = max(
        max_headings,
        max(_count_markdown_headings(r.text) for r in ok),
    )

    metrics: list[ParseMetrics] = []
    for r in results:
        if r.error:
            metrics.append(
                ParseMetrics(
                    parser=r.parser,
                    file_path=str(file_path),
                    doc_profile=profile,
                    text_completeness=0.0,
                    table_structure_score=0.0,
                    heading_detection_score=0.0,
                    char_count=0,
                    table_count=0,
                    heading_count=0,
                    elapsed_ms=r.elapsed_ms,
                    error=r.error,
                )
            )
            continue

        completeness = (r.char_count / max_chars) if max_chars else 0.0
        metrics.append(
            ParseMetrics(
                parser=r.parser,
                file_path=str(file_path),
                doc_profile=profile,
                text_completeness=round(completeness, 4),
                table_structure_score=round(
                    _table_structure_score(r, baseline_table_rows), 4
                ),
                heading_detection_score=round(
                    _heading_score(r, baseline_headings), 4
                ),
                char_count=r.char_count,
                table_count=r.table_count,
                heading_count=r.heading_count,
                elapsed_ms=r.elapsed_ms,
            )
        )
    return metrics


def rank_parsers(metrics: list[ParseMetrics]) -> list[dict]:
    """Weighted score for quick comparison (completeness 50%, tables 25%, headings 25%)."""
    ranked = []
    for m in metrics:
        if m.error:
            ranked.append(
                {
                    "parser": m.parser,
                    "score": 0.0,
                    "error": m.error,
                    "elapsed_ms": m.elapsed_ms,
                }
            )
            continue
        score = (
            0.5 * m.text_completeness
            + 0.25 * m.table_structure_score
            + 0.25 * m.heading_detection_score
        )
        ranked.append(
            {
                "parser": m.parser,
                "score": round(score, 4),
                "text_completeness": m.text_completeness,
                "table_structure_score": m.table_structure_score,
                "heading_detection_score": m.heading_detection_score,
                "elapsed_ms": m.elapsed_ms,
            }
        )
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked

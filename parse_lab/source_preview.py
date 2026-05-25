"""原文件直接预览（不经过解析器 pipeline）。"""

from pathlib import Path

PARSE_SOURCE_TYPES = {".pdf", ".docx"}


def resolve_source_file(path: str) -> Path:
    p = Path(path).resolve()
    if not p.is_file():
        raise ValueError(f"文件不存在: {path}")
    ext = p.suffix.lower()
    if ext not in PARSE_SOURCE_TYPES:
        raise ValueError(f"仅支持原文件预览: {', '.join(sorted(PARSE_SOURCE_TYPES))}")
    return p


def media_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".docx":
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def preview_meta(path: str | Path) -> dict:
    """告知前端如何嵌入预览（不做文本抽取）。"""
    file_path = resolve_source_file(str(path))
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        mode = "pdf_embed"
    elif ext == ".docx":
        mode = "docx_html"
    else:
        mode = "none"
    return {
        "file_path": str(file_path),
        "file_type": ext.lstrip("."),
        "name": file_path.name,
        "preview_mode": mode,
    }


def docx_preview_html(path: str | Path) -> str:
    """docx → HTML（mammoth），仅用于浏览器预览，不参与评测指标。"""
    file_path = resolve_source_file(str(path))
    if file_path.suffix.lower() != ".docx":
        raise ValueError("仅 docx 支持 HTML 预览")

    import mammoth

    with open(file_path, "rb") as f:
        result = mammoth.convert_to_html(f)
    body = result.value or "<p>（空文档）</p>"
    warnings = [str(m) for m in result.messages[:5]]

    warn_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warn_html = f'<ul class="preview-warn">{items}</ul>'

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  body {{
    margin: 0;
    padding: 1rem 1.25rem;
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.6;
    color: #e8eaed;
    background: #1a1f28;
  }}
  .preview-warn {{
    margin: 0 0 1rem;
    padding: 0.5rem 0.75rem;
    background: rgba(251, 191, 36, 0.12);
    border-radius: 6px;
    font-size: 12px;
    color: #fcd34d;
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0; }}
  td, th {{ border: 1px solid #3d4a5c; padding: 4px 8px; }}
  img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
{warn_html}
{body}
</body>
</html>"""

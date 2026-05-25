# RAG Lab 三大实验模块

与 **分块实验** 同属一个 HTTP 服务：`chunk_api`（默认 **8765**）。业务逻辑仍在独立包 `parse_lab` / `retrieval_lab` / `vstore_lab`，路由挂载在：

| 模块 | 业务包 | API 前缀 |
|------|--------|----------|
| 文件解析 | `parse_lab/` | `/parse` |
| 检索策略 | `retrieval_lab/` | `/retrieval` |
| 向量库压测 | `vstore_lab/` | `/vstore` |
| 分块（已有） | `chunk_lab/` | `/`（根路径） |

实验顺序：**解析 → 分块 → 检索 → 向量库压测**。

---

## 本地开发（单进程）

```bash
cd rag_chunk_strategy
uv run uvicorn chunk_api:app --reload --port 8765
```

```bash
cd rag_chunk_tui && npm run dev
```

前端 Vite 仅代理 `/api` → `8765`。实验接口示例：

- `GET /parse/parsers`、`POST /parse/run`、`POST /parse/batch`
- `GET /retrieval/methods`、`POST /retrieval/eval`、`POST /retrieval/compare`
- `GET /vstore/backends`、`POST /vstore/benchmark`

`GET /health` 返回 `experiments` 字段列出上述前缀。

`parse_api.py` / `retrieval_api.py` / `vstore_api.py` 仅保留为兼容入口，均 re-export `chunk_api.app`，**无需再开多端口**。

---

## 1. 文件解析 (`parse_lab`)

解析器注册于 `parse_lab/registry.py`。Web「解析实验」可多选对比；CLI 与 API 等价。

| 解析器 | 说明 |
|--------|------|
| `pypdf` / `pymupdf` / `pdfplumber` / `pymupdf4llm` | PDF 文本 / Markdown |
| `docling` | 结构化（需 `parse-heavy`） |
| `docx2txt` / `python_docx` / `mammoth` | Word |
| `vlm_pdf` | 视觉模型按页解析；配置见 **系统设置 → 视觉模型** |
| `ocr_local` | 本地 RapidOCR 或 Tesseract；`uv sync --extra parse-ocr` |
| `ocr_api` | 在线 OCR；配置见 **系统设置 → OCR API** |

```bash
uv run parse-lab list
uv run parse-lab run -f path/to/file.pdf -p pymupdf,vlm_pdf
uv run python main_parse.py batch -d data/parse_samples   # 目录勿提交 git，见 README
```

HTTP：

- `GET /parse/parsers` — 可用解析器列表
- `GET /parse/vlm-status`、`GET /parse/ocr-status` — 可选能力探测
- `POST /parse/run`、`POST /parse/batch` — 单文件 / 批量（读当前 workspace 设置）

## 2. 检索 (`retrieval_lab`)

依赖 `chunk_lab` 做分块（固定 chunk 策略后对比检索方法）。

```bash
uv run retrieval-lab list
uv run retrieval-lab eval -i data/sample.txt --qa data/experiments/retrieval_qa.json -m hybrid
```

## 3. 向量库 (`vstore_lab`)

```bash
uv run vstore-lab list
uv run vstore-lab bench --sizes 1000,10000 --providers faiss,chroma
```

## 可选依赖

```bash
uv sync --extra parse-heavy   # Docling
```

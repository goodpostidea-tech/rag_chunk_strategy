# RAG Chunk Strategy Lab

基于 [RAG动态Chunk技术方案_2026.md](docs/RAG动态Chunk技术方案_2026.md) 的 LangChain 分块策略对比实验后端，并集成 **解析 / 检索 / 向量库** 三大实验模块。

**默认 LLM：DeepSeek**（`CHUNK_LLM_MODEL=deepseek-chat`），可通过环境变量或 Web「系统配置」切换。

## 仓库关系

本目录为 **独立 Git 仓库**。配套前端：[rag_chunk_tui](../rag_chunk_tui/README.md)。父目录 `rag_space` 仅作本地工作区聚合，无统一 Git 根。

## 已实现分块策略

| 名称 | 分类 | 说明 |
|------|------|------|
| `recursive_baseline` | baseline | 递归字符分割（512/128 默认） |
| `semantic` | A | 语义分块（`SemanticChunker`） |
| `parent_child` | C | 父子分块（`ParentDocumentRetriever`） |
| `metadata_enriched` | utility | 基线 + 元数据前缀 |
| `contextual` | B | LLM 上下文前缀 |
| `proposition` | B | LLM 原子命题分块 |
| `multi_granularity` | utility | 多粒度（256/512/1024） |

## 三大实验模块

业务包：`parse_lab` / `retrieval_lab` / `vstore_lab`。HTTP 与分块共用 **`chunk_api`（8765）**。详见 [docs/experiments.md](docs/experiments.md)。

| 模块 | 业务包 | API 前缀 | CLI |
|------|--------|----------|-----|
| 文件解析 | `parse_lab/` | `/parse` | `uv run python main_parse.py` |
| 检索策略 | `retrieval_lab/` | `/retrieval` | `uv run python main_retrieval.py` |
| 向量库压测 | `vstore_lab/` | `/vstore` | `uv run python main_vstore.py` |

### 解析实验解析器（`parse_lab`）

| 名称 | 说明 | 依赖 / 配置 |
|------|------|-------------|
| `pypdf` / `pymupdf` / `pdfplumber` | PDF 文本提取 | 默认可用 |
| `pymupdf4llm` | Markdown 友好 PDF | `uv sync --extra parse-heavy` |
| `docling` | 结构化文档 | `parse-heavy` |
| `docx2txt` / `python_docx` / `mammoth` | Word | `mammoth` 需额外安装 |
| **`vlm_pdf`** | 视觉模型按页解析 PDF | **系统配置 → 视觉模型**；需 `pymupdf` |
| **`ocr_local`** | 本地 OCR（RapidOCR 优先，否则 Tesseract） | 可选 `uv sync --extra parse-ocr` |
| **`ocr_api`** | 在线 OCR | **系统配置 → OCR API**（Key / 服务商） |

状态接口：`GET /parse/vlm-status`、`GET /parse/ocr-status`。

## 快速开始

```bash
cd rag_chunk_strategy
cp .env.example .env   # 填入 DEEPSEEK_API_KEY 等

uv sync
# 可选：Docling / 本地 OCR
# uv sync --extra parse-heavy
# uv sync --extra parse-ocr

# 单进程 API（推荐，供 TUI 代理）
uv run uvicorn chunk_api:app --reload --port 8765
```

```bash
# CLI 示例
uv run python main.py list
uv run python main.py compare -i data/sample.txt --json
uv run python main_parse.py list
uv run python main_retrieval.py compare -i data/sample.txt -m hybrid
```

## Web 界面（rag_chunk_tui）

```bash
# 终端 1
uv run uvicorn chunk_api:app --reload --port 8765

# 终端 2
cd ../rag_chunk_tui && npm install && npm run dev
# http://localhost:5173
```

生产托管：先 `npm run build`，再 `uvicorn chunk_api:app --port 8765` → http://localhost:8765

## 参考文献语料库

`data/references_catalog.json` 对应 `docs/RAG_Chunk_参考文献.md`：

```bash
uv run python main.py papers list
uv run python main.py papers fetch --id meta_chunking
uv run python main.py papers fetch --all-arxiv
```

抓取正文缓存于 `data/papers/{id}.txt`（**不提交 git**，见下方）。

## 模型与向量库（`.env`）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_LLM_MODEL` | `deepseek-chat` | 分块 / Judge LLM |
| `VECTOR_STORE_PROVIDER` | `faiss` | `faiss` / `qdrant` / `milvus` / `chroma` / `pinecone` |
| `CHUNK_EMBEDDING_PROVIDER` | `huggingface` | 嵌入提供商 |
| `CHUNK_EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | 嵌入模型 |

解析相关的 **视觉模型**、**OCR API** 凭证通过 `PUT /settings` 写入 SQLite 覆盖层（`data/chunk_lab.db`），勿写入 `.env` 后提交。

完整变量见 [.env.example](.env.example)。

## API 端点（节选）

| 前缀 | 说明 |
|------|------|
| `/strategies` `/chunk` `/compare` `/eval` | 分块实验 |
| `/parse/*` | 解析实验 |
| `/retrieval/*` | 检索实验 |
| `/vstore/*` | 向量库压测 |
| `/settings` `/settings/schema` | 系统配置（含 vision / ocr 组） |
| `/health` | 健康检查与实验模块列表 |

## Git：应提交 vs 忽略

| 建议提交 | 不要提交 |
|----------|----------|
| 源码、`pyproject.toml`、`uv.lock`、`.env.example` | `.env`、`.venv/` |
| `data/sample.txt`、`data/sample_qa.json`、`data/references_catalog.json`、`data/papers_qa.json`、`data/experiments/` | `data/chunk_lab.db*`（含 UI 保存的 API Key） |
| `data/papers/.gitkeep` | `data/papers/*.txt`、`*.meta.json`、PDF |
| `docs/` | `data/parse_samples/`（本地试跑目录）、`data/custom_qa.json`、`data/indexes/` |
| | `.chroma_chunk_lab/`、`__pycache__/`、`.claude/` |

首次克隆后执行 `cp .env.example .env`；论文语料需自行 `papers fetch`。

## 可观测性

默认关闭 LangSmith / Langfuse。开启方式见 `.env.example` 注释；`GET /health` 可查看当前状态。

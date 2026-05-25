"""PDF parser implementations."""

from pathlib import Path

from parse_lab.base import BaseDocumentParser
from parse_lab.parsers._utils import headings_from_markdown, run_timed
from parse_lab.types import ParsedTable


class PypdfParser(BaseDocumentParser):
    name = "pypdf"
    description = "PyPDF 基线（langchain PyPDFLoader，纯文本流）"
    file_types = (".pdf",)

    def parse(self, file_path: Path):
        def _do():
            from langchain_community.document_loaders import PyPDFLoader

            docs = PyPDFLoader(str(file_path)).load()
            text = "\n\n".join(d.page_content for d in docs)
            return text, [], [], {"pages": len(docs)}

        return run_timed(self.name, file_path, _do)


class PyMuPdfParser(BaseDocumentParser):
    name = "pymupdf"
    description = "PyMuPDF (fitz) 高速纯文本提取"
    file_types = (".pdf",)

    def parse(self, file_path: Path):
        def _do():
            import fitz

            doc = fitz.open(str(file_path))
            parts = [page.get_text() for page in doc]
            doc.close()
            text = "\n\n".join(parts)
            return text, [], [], {"pages": len(parts)}

        return run_timed(self.name, file_path, _do)


class PdfPlumberParser(BaseDocumentParser):
    name = "pdfplumber"
    description = "pdfplumber 表格结构保留较好"
    file_types = (".pdf",)

    def parse(self, file_path: Path):
        def _do():
            import pdfplumber

            text_parts: list[str] = []
            tables: list[ParsedTable] = []
            with pdfplumber.open(str(file_path)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        text_parts.append(t)
                    for tbl in page.extract_tables() or []:
                        rows = [[str(c or "") for c in row] for row in tbl if row]
                        if rows:
                            tables.append(ParsedTable(rows=rows, source="pdfplumber"))
            text = "\n\n".join(text_parts)
            return text, tables, [], {"pages": len(text_parts), "table_count": len(tables)}

        return run_timed(self.name, file_path, _do)


class PyMuPdf4LlmParser(BaseDocumentParser):
    name = "pymupdf4llm"
    description = "pymupdf4llm 输出 Markdown，保留标题层级"
    file_types = (".pdf",)
    optional = True

    def parse(self, file_path: Path):
        def _do():
            import pymupdf4llm

            text = pymupdf4llm.to_markdown(str(file_path))
            headings = headings_from_markdown(text)
            return text, [], headings, {"format": "markdown"}

        return run_timed(self.name, file_path, _do)


class DoclingParser(BaseDocumentParser):
    name = "docling"
    description = "IBM Docling 结构化 PDF/docx（依赖较重）"
    file_types = (".pdf", ".docx")
    optional = True

    def parse(self, file_path: Path):
        def _do():
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            text = result.document.export_to_markdown()
            headings = headings_from_markdown(text)
            return text, [], headings, {"format": "markdown", "engine": "docling"}

        return run_timed(self.name, file_path, _do)

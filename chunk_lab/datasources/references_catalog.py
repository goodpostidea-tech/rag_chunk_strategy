"""Load reference papers catalog from JSON."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "references_catalog.json"

SourceType = Literal["arxiv", "web", "benchmark", "tool", "engineering"]


class ReferencePaper(BaseModel):
    id: str
    ref_number: int
    title: str
    authors: list[str] = Field(default_factory=list)
    category: str
    source_type: SourceType
    arxiv_id: str | None = None
    url: str | None = None
    published: str | None = None
    related_strategies: list[str] = Field(default_factory=list)
    summary: str = ""

    @property
    def is_arxiv(self) -> bool:
        return bool(self.arxiv_id)

    @property
    def fetchable_arxiv(self) -> bool:
        return self.source_type == "arxiv" and bool(self.arxiv_id)


@lru_cache
def load_catalog() -> list[ReferencePaper]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [ReferencePaper.model_validate(item) for item in raw]


def get_by_id(paper_id: str) -> ReferencePaper:
    for paper in load_catalog():
        if paper.id == paper_id:
            return paper
    available = ", ".join(p.id for p in load_catalog())
    raise KeyError(f"Unknown paper id '{paper_id}'. Available: {available}")

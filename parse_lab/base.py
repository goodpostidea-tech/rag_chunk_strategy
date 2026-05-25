"""Base document parser interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from parse_lab.types import ParseResult


class BaseDocumentParser(ABC):
    """Parse a single file into structured text + optional tables/headings."""

    name: ClassVar[str]
    description: ClassVar[str]
    file_types: ClassVar[tuple[str, ...]]  # e.g. (".pdf",)
    optional: ClassVar[bool] = False  # requires extra pip package

    @abstractmethod
    def parse(self, file_path: Path) -> ParseResult:
        """Parse file and return structured result."""

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.file_types

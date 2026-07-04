"""Load documents and extract text page by page.

Each page keeps its page number so answers can cite exact sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


@dataclass
class Page:
    doc_name: str
    page_number: int  # 1-indexed
    text: str


def load_pdf(path: str | Path) -> list[Page]:
    """Extract text from every page of a PDF."""
    path = Path(path)
    pages: list[Page] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append(Page(doc_name=path.name, page_number=i, text=text))
    return pages


def load_text_file(path: str | Path) -> list[Page]:
    """Load a plain text / markdown file as a single page."""
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    return [Page(doc_name=path.name, page_number=1, text=text)] if text else []


def load_document(path: str | Path) -> list[Page]:
    """Route to the right loader based on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return load_text_file(path)
    raise ValueError(f"Unsupported file type: {suffix}")

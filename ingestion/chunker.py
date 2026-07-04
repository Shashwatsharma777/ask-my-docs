"""Split pages into token-based chunks that respect paragraph boundaries.

Chunks target ~600 tokens (500-800 range) with ~100 tokens of overlap when
a long paragraph must be split. Every chunk carries (doc_name, page_number,
chunk_id) so retrieval results can always be traced back to an exact
location in the source document.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from app import config
from ingestion.loader import Page

_encoding = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_number: int
    text: str


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_pages(
    pages: list[Page],
    chunk_tokens: int = config.CHUNK_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Greedily pack paragraphs into chunks of ~chunk_tokens tokens.

    Paragraphs longer than chunk_tokens are split with a sliding window
    over their token ids, keeping `overlap_tokens` of context between
    consecutive windows.
    """
    chunks: list[Chunk] = []
    for page in pages:
        buffer = ""
        buffer_tokens = 0
        for para in _split_paragraphs(page.text):
            para_tokens = count_tokens(para)
            if para_tokens > chunk_tokens:
                if buffer:
                    chunks.append(_make_chunk(page, buffer, len(chunks)))
                    buffer, buffer_tokens = "", 0
                token_ids = _encoding.encode(para)
                start = 0
                while start < len(token_ids):
                    window = _encoding.decode(token_ids[start : start + chunk_tokens])
                    chunks.append(_make_chunk(page, window, len(chunks)))
                    start += chunk_tokens - overlap_tokens
            elif buffer_tokens + para_tokens > chunk_tokens:
                chunks.append(_make_chunk(page, buffer, len(chunks)))
                buffer, buffer_tokens = para, para_tokens
            else:
                buffer = f"{buffer}\n{para}" if buffer else para
                buffer_tokens += para_tokens
        if buffer:
            chunks.append(_make_chunk(page, buffer, len(chunks)))
    return chunks


def _make_chunk(page: Page, text: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{page.doc_name}:p{page.page_number}:c{index}",
        doc_name=page.doc_name,
        page_number=page.page_number,
        text=text.strip(),
    )

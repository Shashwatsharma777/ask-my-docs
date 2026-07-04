"""Lightweight shared types — no heavy ML dependencies.

Keeping ScoredChunk here lets the generator and tests import it without
pulling in chromadb / sentence-transformers.
"""
from __future__ import annotations

from dataclasses import dataclass

from ingestion.chunker import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

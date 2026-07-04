"""Load versioned prompts from prompts.yaml.

All prompt text lives in the YAML file — never hardcode prompts in code.
The version number is recorded in eval reports for traceability.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.yaml"

_cache: dict | None = None


def load() -> dict:
    global _cache
    if _cache is None:
        _cache = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
    return _cache


def version() -> int:
    return load()["version"]


def not_found_message() -> str:
    return load()["not_found_message"]


def template(name: str) -> str:
    return load()[name]["template"]

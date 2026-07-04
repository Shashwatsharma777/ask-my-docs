"""Faithfulness scoring via LLM-as-judge.

Asks: are the claims in the generated answer actually supported by the
retrieved chunks? Returns supported_claims / total_claims in [0, 1].
"""
from __future__ import annotations

import json
import re

from google import genai

from app import config, prompts
from retrieval.types import ScoredChunk

_JSON_PATTERN = re.compile(r"\{[^{}]*\}")


def score_faithfulness(answer: str, results: list[ScoredChunk]) -> float:
    """Score one answer against its retrieved context. 1.0 = fully faithful."""
    context = "\n\n---\n\n".join(r.chunk.text for r in results)
    prompt = prompts.template("faithfulness_judge").format(
        context=context, answer=answer
    )
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(model=config.LLM_MODEL, contents=prompt)
    text = (response.text or "").strip()

    match = _JSON_PATTERN.search(text)
    if not match:
        raise ValueError(f"Judge returned unparseable output: {text[:200]}")
    verdict = json.loads(match.group())
    total = int(verdict["total_claims"])
    supported = int(verdict["supported_claims"])
    if total == 0:
        return 1.0
    return max(0.0, min(1.0, supported / total))

"""Generate answers from retrieved chunks with enforced citations.

Enforcement is two-layer:
1. The answer must contain citations in [doc_name, page N] form.
2. Every cited (doc, page) must actually be among the retrieved chunks —
   a citation pointing at a source the system never retrieved is treated
   as a hallucination and the answer is refused.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from google import genai

from app import config, prompts
from retrieval.types import ScoredChunk

CITATION_PATTERN = re.compile(r"\[([^\[\],]+),\s*page\s+(\d+)\]", re.IGNORECASE)


@dataclass
class Answer:
    text: str
    sources: list[dict]
    grounded: bool  # True when citations verify against retrieved chunks, or honest "not found"


def _format_context(results: list[ScoredChunk]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[{r.chunk.doc_name}, page {r.chunk.page_number}]\n{r.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def verify_citations(text: str, results: list[ScoredChunk]) -> bool:
    """Every citation in the answer must point at a retrieved (doc, page)."""
    citations = CITATION_PATTERN.findall(text)
    if not citations:
        return False
    retrieved = {(r.chunk.doc_name.lower(), r.chunk.page_number) for r in results}
    return all(
        (doc.strip().lower(), int(page)) in retrieved for doc, page in citations
    )


def generate_answer(question: str, results: list[ScoredChunk]) -> Answer:
    not_found = prompts.not_found_message()
    if not results:
        return Answer(text=not_found, sources=[], grounded=True)

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    prompt = prompts.template("qa").format(
        not_found=not_found,
        context=_format_context(results),
        question=question,
    )
    response = client.models.generate_content(model=config.LLM_MODEL, contents=prompt)
    text = (response.text or "").strip()

    is_not_found = not_found.lower() in text.lower()

    # Citation enforcement: refuse rather than return an ungrounded claim.
    if not is_not_found and not verify_citations(text, results):
        return Answer(text=not_found, sources=[], grounded=False)

    sources = [
        {
            "doc_name": r.chunk.doc_name,
            "page_number": r.chunk.page_number,
            "chunk_id": r.chunk.chunk_id,
            "rerank_score": round(r.score, 4),
        }
        for r in results
    ]
    return Answer(text=text, sources=[] if is_not_found else sources, grounded=True)

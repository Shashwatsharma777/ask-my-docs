"""Generate DRAFT Q&A pairs for the golden dataset from eval/corpus/ docs.

IMPORTANT: output goes to eval/draft_dataset.jsonl — every pair MUST be
manually verified for correctness before being moved into eval/dataset.jsonl.
A golden dataset is only golden if a human checked it.

Usage:
    python eval/generate_dataset.py [--per-chunk 1] [--max-chunks 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402

from app import config, prompts  # noqa: E402
from ingestion.chunker import chunk_pages  # noqa: E402
from ingestion.loader import load_document  # noqa: E402

CORPUS_DIR = Path(__file__).parent / "corpus"
DRAFT_PATH = Path(__file__).parent / "draft_dataset.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-chunk", type=int, default=1)
    parser.add_argument("--max-chunks", type=int, default=50)
    args = parser.parse_args()

    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is required.")
        sys.exit(2)
    if not CORPUS_DIR.exists():
        print(f"ERROR: {CORPUS_DIR} does not exist. Add documents there first.")
        sys.exit(2)

    chunks = []
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix.lower() in {".pdf", ".txt", ".md"}:
            chunks.extend(chunk_pages(load_document(path)))
    if not chunks:
        print("ERROR: no chunks produced from eval/corpus/")
        sys.exit(2)

    # Spread sampling evenly across the corpus instead of taking the head.
    step = max(1, len(chunks) // args.max_chunks)
    sampled = chunks[::step][: args.max_chunks]

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    template = prompts.template("dataset_generator")
    drafts = []

    for chunk in sampled:
        prompt = template.format(
            doc_name=chunk.doc_name,
            page_number=chunk.page_number,
            chunk_text=chunk.text,
            n_questions=args.per_chunk,
        )
        try:
            response = client.models.generate_content(
                model=config.LLM_MODEL, contents=prompt
            )
            text = (response.text or "").strip()
            start, end = text.find("["), text.rfind("]")
            pairs = json.loads(text[start : end + 1])
        except Exception as e:
            print(f"  skip {chunk.chunk_id}: {e}")
            continue

        for pair in pairs:
            drafts.append(
                {
                    "question": pair["question"],
                    "ground_truth": pair["ground_truth"],
                    "expected_doc": chunk.doc_name,
                    "expected_page": chunk.page_number,
                    "verified": False,  # flip to true after human review
                }
            )
        print(f"  {chunk.chunk_id}: {len(pairs)} pair(s)")

    with DRAFT_PATH.open("w", encoding="utf-8") as f:
        for row in drafts:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{len(drafts)} draft pairs written to {DRAFT_PATH}")
    print("NEXT: manually verify each pair, set verified=true, then move the")
    print("verified rows (without the 'verified' field) into eval/dataset.jsonl")


if __name__ == "__main__":
    main()

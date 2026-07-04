"""Evaluation pipeline with a CI quality gate.

Measures against the golden dataset (eval/dataset.jsonl):
  - Retrieval hit rate: did the expected page appear in retrieved chunks?
  - Grounding rate: did answers carry verified citations (or honest refusals)?
  - Faithfulness: are the claims in each answer supported by the retrieved
    chunks? (LLM-as-judge, skipped for "not found" answers)

Exits non-zero when a metric falls below its threshold, so CI blocks
merges that degrade quality. The report records the prompt version so
every metric is traceable to the exact prompts that produced it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, prompts  # noqa: E402
from app.generator import generate_answer  # noqa: E402
from eval.faithfulness import score_faithfulness  # noqa: E402
from retrieval.store import HybridStore  # noqa: E402

DATASET = Path(__file__).parent / "dataset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

# CI gate thresholds — a PR that drops below these fails the pipeline.
THRESHOLDS = {
    "retrieval_hit_rate": 0.80,
    "grounding_rate": 0.90,
    "faithfulness": 0.85,
}


def load_dataset() -> list[dict]:
    rows = []
    for line in DATASET.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def evaluate() -> dict:
    if not config.GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is required to run the evaluation.")
        sys.exit(2)

    dataset = load_dataset()
    store = HybridStore()
    if store.is_empty():
        print("ERROR: index is empty — run `python eval/ingest_corpus.py` first.")
        sys.exit(2)

    not_found = prompts.not_found_message()
    retrieval_hits = 0
    grounded_count = 0
    faithfulness_scores: list[float] = []
    records = []

    for row in dataset:
        results = store.search(row["question"])
        retrieved_pages = {(r.chunk.doc_name, r.chunk.page_number) for r in results}
        hit = (row["expected_doc"], row["expected_page"]) in retrieved_pages
        retrieval_hits += hit

        answer = generate_answer(row["question"], results)
        grounded_count += answer.grounded

        faithfulness = None
        if answer.grounded and not_found.lower() not in answer.text.lower():
            faithfulness = score_faithfulness(answer.text, results)
            faithfulness_scores.append(faithfulness)

        records.append(
            {
                "question": row["question"],
                "retrieval_hit": hit,
                "grounded": answer.grounded,
                "faithfulness": faithfulness,
                "answer": answer.text,
            }
        )

    n = len(dataset)
    metrics = {
        "prompt_version": prompts.version(),
        "n_questions": n,
        "retrieval_hit_rate": retrieval_hits / n,
        "grounding_rate": grounded_count / n,
        "faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores)
            if faithfulness_scores
            else 1.0
        ),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "report.json"
    report_path.write_text(json.dumps({"metrics": metrics, "records": records}, indent=2))
    print(f"Report written to {report_path}")
    return metrics


def main() -> None:
    metrics = evaluate()
    print("\n=== Evaluation Report ===")
    for name, value in metrics.items():
        print(f"  {name}: {value:.2%}" if isinstance(value, float) else f"  {name}: {value}")

    failures = [
        f"{name} {metrics[name]:.2%} < threshold {threshold:.0%}"
        for name, threshold in THRESHOLDS.items()
        if metrics[name] < threshold
    ]
    if failures:
        print("\nQUALITY GATE FAILED:")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(1)
    print("\nQuality gate passed ✓")


if __name__ == "__main__":
    main()

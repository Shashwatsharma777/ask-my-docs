"""Index every document in eval/corpus/ so run_eval.py has data to search.

Used locally and by CI before running the evaluation gate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import chunk_pages  # noqa: E402
from ingestion.loader import load_document  # noqa: E402
from retrieval.store import HybridStore  # noqa: E402

CORPUS_DIR = Path(__file__).parent / "corpus"


def main() -> None:
    if not CORPUS_DIR.exists():
        print(f"ERROR: {CORPUS_DIR} does not exist. Add eval documents there.")
        sys.exit(2)

    store = HybridStore()
    total_chunks = 0
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        pages = load_document(path)
        chunks = chunk_pages(pages)
        store.add_chunks(chunks)
        total_chunks += len(chunks)
        print(f"Indexed {path.name}: {len(pages)} pages, {len(chunks)} chunks")

    if total_chunks == 0:
        print("ERROR: no documents indexed from eval/corpus/")
        sys.exit(2)
    print(f"\nDone — {total_chunks} chunks indexed.")


if __name__ == "__main__":
    main()

# Ask My Docs 📄

A production-grade RAG (Retrieval-Augmented Generation) system that answers questions from your documents with **page-level citations** — built with hybrid retrieval, cross-encoder reranking, citation enforcement, and a **CI-gated evaluation pipeline**.

## Why this isn't another toy RAG demo

| Feature | Naive RAG | Ask My Docs |
|---|---|---|
| Chunking | Arbitrary splits | **Token-based (600 tokens, 100 overlap), paragraph-aware, page-tracked** |
| Retrieval | Vector search only | **BM25 + vector search fused with RRF** |
| Precision | Top-k as-is | **Cross-encoder reranking** |
| Trust | Answers may hallucinate | **Two-layer citation enforcement** — uncited or wrongly-cited answers are refused |
| Prompts | Hardcoded strings | **Versioned prompts.yaml** — eval reports record the prompt version |
| Quality | "Looks fine" | **Golden dataset + faithfulness (LLM-as-judge) metrics** |
| Regressions | Silent | **CI gate blocks merges that degrade accuracy** |

## Architecture

```
                ┌─────────────┐
 PDF/TXT/MD ──> │  Ingestion  │ ──> page-aware chunks
                └─────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ┌──────────┐      ┌──────────┐
        │  BM25    │      │  Chroma  │
        │ (lexical)│      │ (vector) │
        └────┬─────┘      └────┬─────┘
             └───── RRF ───────┘
                     │
              ┌──────▼───────┐
              │ Cross-encoder │
              │   reranker    │
              └──────┬───────┘
                     │  top-5 chunks
              ┌──────▼───────┐
              │  LLM (Gemini) │ ──> answer + [doc, page N] citations
              └──────────────┘
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY

# Start the API
uvicorn app.main:app --reload

# Upload a document
curl -X POST http://localhost:8000/ingest -F "file=@your_document.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total premium amount?"}'
```

Example response:

```json
{
  "question": "What is the total premium amount?",
  "answer": "The total premium is $50,000 [policy.pdf, page 3].",
  "grounded": true,
  "sources": [
    {"doc_name": "policy.pdf", "page_number": 3, "chunk_id": "policy.pdf:p3:c12", "rerank_score": 8.4123}
  ]
}
```

## How retrieval works

1. **BM25** catches exact terms (policy numbers, names, codes) that embeddings miss.
2. **Vector search** catches paraphrases ("how much do I pay" → "premium amount").
3. **Reciprocal Rank Fusion** merges both rankings without score normalization headaches.
4. **Cross-encoder reranking** re-scores each (query, chunk) pair jointly — far more precise than bi-encoder similarity — and keeps only the top 5.

## Citation enforcement (two layers)

1. The LLM must cite every claim as `[doc_name, page N]` — a factual answer without citations is refused.
2. Every cited `(doc, page)` is **verified against the actually-retrieved chunks** — a citation pointing at a source the system never retrieved is treated as a hallucination and the answer is replaced with an honest refusal.

No verified citation, no answer.

## Versioned prompts

All prompts live in [prompts.yaml](prompts.yaml) — never in code. The version number is bumped on every prompt change and recorded in each eval report, so quality metrics are always traceable to the exact prompts that produced them.

## Evaluation & CI gate

```bash
python eval/ingest_corpus.py      # index eval documents (eval/corpus/)
python eval/generate_dataset.py   # draft Q&A pairs (MUST be human-verified)
python eval/run_eval.py           # run metrics + quality gate
```

Metrics measured against the golden dataset `eval/dataset.jsonl` (manually verified question + ground truth + expected source page):

- **Retrieval hit rate** — did the expected page appear in retrieved chunks? (gate: ≥ 80%)
- **Grounding rate** — did answers carry verified citations or honest refusals? (gate: ≥ 90%)
- **Faithfulness** — are the claims in each answer actually supported by the retrieved chunks? Scored by an LLM judge as supported_claims / total_claims. (gate: ≥ 85%)

The GitHub Actions workflow ([.github/workflows/eval.yml](.github/workflows/eval.yml)) runs unit tests **and the evaluation gate on every PR** — a change that drops any metric below threshold fails CI and cannot merge. This is how production AI teams operate.

## Project structure

```
app/          FastAPI app, config, prompt loader, LLM answer generation
ingestion/    Document loading (PyMuPDF) + token-based page-aware chunking
retrieval/    Hybrid store: Chroma + BM25 + RRF + cross-encoder
eval/         Golden dataset, corpus ingestion, faithfulness judge, gated eval runner
tests/        Unit tests
prompts.yaml  Versioned prompt configuration
```

## Tech stack

Python 3.11 · FastAPI · ChromaDB · rank-bm25 · sentence-transformers · tiktoken · Gemini · GitHub Actions

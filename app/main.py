"""Ask My Docs — FastAPI application.

Endpoints:
  POST /ingest  — upload a PDF/TXT/MD file and index it
  POST /ask     — ask a question, get a cited answer
  GET  /health  — service status
"""
from __future__ import annotations

import shutil

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from app import config
from app.generator import generate_answer
from ingestion.chunker import chunk_pages
from ingestion.loader import load_document
from retrieval.store import HybridStore

app = FastAPI(title="Ask My Docs", version="0.1.0")
store: HybridStore | None = None


def get_store() -> HybridStore:
    global store
    if store is None:
        store = HybridStore()
    return store


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "indexed": not get_store().is_empty()}


@app.post("/ingest")
async def ingest(file: UploadFile):
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        pages = load_document(dest)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_pages(pages)
    get_store().add_chunks(chunks)
    return {
        "doc_name": file.filename,
        "pages": len(pages),
        "chunks_indexed": len(chunks),
    }


@app.post("/ask")
def ask(request: AskRequest):
    s = get_store()
    if s.is_empty():
        raise HTTPException(status_code=400, detail="No documents indexed yet. Use /ingest first.")
    results = s.search(request.question)
    answer = generate_answer(request.question, results)
    return {
        "question": request.question,
        "answer": answer.text,
        "grounded": answer.grounded,
        "sources": answer.sources,
    }

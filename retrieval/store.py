"""Index and search chunks: vector store (Chroma) + BM25, fused with RRF,
then reranked with a cross-encoder.

Pipeline: query -> BM25 top-k + vector top-k -> Reciprocal Rank Fusion
          -> cross-encoder rerank -> top-n chunks for the LLM.
"""
from __future__ import annotations

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from app import config
from ingestion.chunker import Chunk
from retrieval.types import ScoredChunk


class HybridStore:
    def __init__(self, collection_name: str = "ask_my_docs"):
        self._embedder = SentenceTransformer(config.EMBEDDING_MODEL)
        self._reranker = CrossEncoder(config.RERANKER_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._collection = client.get_or_create_collection(collection_name)
        self._chunks: dict[str, Chunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Rebuild in-memory chunk map and BM25 index from Chroma on startup."""
        existing = self._collection.get(include=["documents", "metadatas"])
        for chunk_id, text, meta in zip(
            existing["ids"], existing["documents"], existing["metadatas"]
        ):
            self._chunks[chunk_id] = Chunk(
                chunk_id=chunk_id,
                doc_name=meta["doc_name"],
                page_number=meta["page_number"],
                text=text,
            )
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._bm25_ids = list(self._chunks.keys())
        if self._bm25_ids:
            tokenized = [self._chunks[cid].text.lower().split() for cid in self._bm25_ids]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self._embedder.encode([c.text for c in chunks]).tolist()
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {"doc_name": c.doc_name, "page_number": c.page_number} for c in chunks
            ],
        )
        for c in chunks:
            self._chunks[c.chunk_id] = c
        self._rebuild_bm25()

    def _vector_search(self, query: str, k: int) -> list[str]:
        if not self._chunks:
            return []
        embedding = self._embedder.encode([query]).tolist()
        result = self._collection.query(
            query_embeddings=embedding, n_results=min(k, len(self._chunks))
        )
        return result["ids"][0]

    def _bm25_search(self, query: str, k: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._bm25_ids, scores), key=lambda x: x[1], reverse=True)
        return [cid for cid, score in ranked[:k] if score > 0]

    def _fuse_rrf(self, rankings: list[list[str]]) -> list[str]:
        """Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank))."""
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (config.RRF_K + rank)
        return sorted(fused, key=fused.get, reverse=True)

    def search(
        self,
        query: str,
        top_k_retrieve: int = config.TOP_K_RETRIEVE,
        top_k_rerank: int = config.TOP_K_RERANK,
    ) -> list[ScoredChunk]:
        vector_ids = self._vector_search(query, top_k_retrieve)
        bm25_ids = self._bm25_search(query, top_k_retrieve)
        candidate_ids = self._fuse_rrf([vector_ids, bm25_ids])[:top_k_retrieve]
        if not candidate_ids:
            return []

        candidates = [self._chunks[cid] for cid in candidate_ids]
        scores = self._reranker.predict([(query, c.text) for c in candidates])
        reranked = sorted(
            zip(candidates, scores), key=lambda x: float(x[1]), reverse=True
        )
        return [
            ScoredChunk(chunk=c, score=float(s)) for c, s in reranked[:top_k_rerank]
        ]

    def is_empty(self) -> bool:
        return not self._chunks

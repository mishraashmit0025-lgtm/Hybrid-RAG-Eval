"""The retrieval pipeline: BM25 + dense candidates, fused, optionally reranked."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25
from .fusion import Ranking, minmax_fusion, rrf
from .text import Chunk

METHODS = ("bm25", "dense", "hybrid_minmax", "hybrid_rrf", "hybrid_rrf_rerank")


@dataclass
class Hit:
    chunk: Chunk
    score: float


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], dense_model: str | None = "BAAI/bge-small-en-v1.5",
                 reranker_model: str | None = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 cache_dir: str | Path | None = None, candidates: int = 100, rerank_depth: int = 30):
        self.chunks = chunks
        self.texts = [f"{c.title}\n{c.text}" if c.title else c.text for c in chunks]
        self.candidates = candidates
        self.rerank_depth = rerank_depth
        self.bm25 = BM25().fit(self.texts)
        self.dense = None
        self.reranker = None
        if dense_model:
            from .dense import DenseIndex

            self.dense = DenseIndex(dense_model, cache_dir=cache_dir).fit(self.texts, show_progress=len(chunks) > 2000)
        if reranker_model:
            from .rerank import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(reranker_model)

    def rank(self, query: str, method: str, k: int, qvec=None) -> Ranking:
        """Ranking of chunk indexes. qvec lets callers batch-encode queries."""
        if method == "bm25":
            return self.bm25.search(query, k)
        if self.dense is None:
            raise ValueError(f"method {method!r} needs a dense model")
        from .bm25 import top_k

        dense = top_k(self.dense.scores_for(qvec if qvec is not None else self.dense.encode_queries([query])[0]),
                      max(k, self.candidates))
        if method == "dense":
            return dense[:k]
        sparse_ = self.bm25.search(query, self.candidates)
        if method == "hybrid_minmax":
            return minmax_fusion([sparse_, dense[: self.candidates]], limit=k)
        fused = rrf([sparse_, dense[: self.candidates]])
        if method == "hybrid_rrf":
            return fused[:k]
        if method == "hybrid_rrf_rerank":
            if self.reranker is None:
                raise ValueError("hybrid_rrf_rerank needs a reranker model")
            head = self.reranker.rerank(query, fused[: self.rerank_depth], self.texts)
            # keep the fused order below the reranked head so recall@100 is unaffected
            tail = [h for h in fused[self.rerank_depth:]]
            return (head + tail)[:k]
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")

    def search(self, query: str, k: int = 5, method: str = "hybrid_rrf_rerank") -> list[Hit]:
        if method == "hybrid_rrf_rerank" and self.reranker is None:
            method = "hybrid_rrf" if self.dense is not None else "bm25"
        return [Hit(self.chunks[i], s) for i, s in self.rank(query, method, k)]

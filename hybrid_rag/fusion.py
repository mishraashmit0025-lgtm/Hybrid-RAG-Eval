"""Combining ranked lists from different retrievers."""

from __future__ import annotations

from collections import defaultdict

Ranking = list[tuple[int, float]]  # (doc index, score), best first


def rrf(rankings: list[Ranking], k: int = 60, weights: list[float] | None = None, limit: int | None = None) -> Ranking:
    """Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
    score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1.

    It only looks at ranks, so it needs no score calibration between BM25 (unbounded)
    and cosine similarity (in [-1, 1]). k=60 is the value from the paper."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        for rank, (doc, _) in enumerate(ranking, start=1):
            fused[doc] += w / (k + rank)
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:limit] if limit else out


def minmax_fusion(rankings: list[Ranking], weights: list[float] | None = None, limit: int | None = None) -> Ranking:
    """Convex combination of min-max normalized scores; a document missing from a
    list gets 0 from it. Kept as a baseline to compare with RRF."""
    weights = weights or [1.0 / len(rankings)] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for w, ranking in zip(weights, rankings):
        if not ranking:
            continue
        scores = [s for _, s in ranking]
        lo, hi = min(scores), max(scores)
        span = hi - lo or 1.0
        for doc, s in ranking:
            fused[doc] += w * (s - lo) / span
    out = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:limit] if limit else out

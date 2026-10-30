"""Standard IR metrics. qrels map doc id -> graded relevance (> 0 is relevant)."""

from __future__ import annotations

import math


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    rel = {d for d, g in qrels.items() if g > 0}
    if not rel:
        return 0.0
    return len(rel.intersection(ranked[:k])) / len(rel)


def mrr_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if qrels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """nDCG with linear gains (rel / log2(rank + 1)), as computed by trec_eval and BEIR."""
    dcg = sum(qrels.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1))
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def dedupe_docs(ranked_chunks: list[str]) -> list[str]:
    """Collapse a ranking of chunk ids ("doc#n") into a ranking of documents,
    keeping each document at the rank of its best chunk."""
    seen: set[str] = set()
    out = []
    for cid in ranked_chunks:
        doc = cid.split("#", 1)[0]
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def paired_bootstrap(a: list[float], b: list[float], iters: int = 10000, seed: int = 0) -> float:
    """Two-sided p-value that systems a and b have the same mean, by resampling queries."""
    import random

    rng = random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    observed = abs(mean(diffs))
    centered = [d - mean(diffs) for d in diffs]
    hits = 0
    for _ in range(iters):
        sample = [centered[rng.randrange(n)] for _ in range(n)]
        if abs(mean(sample)) >= observed:
            hits += 1
    return (hits + 1) / (iters + 1)

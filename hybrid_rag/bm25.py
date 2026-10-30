"""Okapi BM25 over a sparse term-document matrix."""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from .text import tokenize


class BM25:
    """score(q, d) = sum over q's terms t of
         idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    with the Lucene idf, log(1 + (N - df + 0.5) / (df + 0.5)), which is never negative.

    The per-(term, doc) weights are precomputed into a CSC matrix at index time,
    so a query is one sparse column sum."""

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        # k1=0.9, b=0.4 are the Anserini defaults used for the BEIR baselines
        self.k1 = k1
        self.b = b

    def fit(self, texts: list[str]) -> "BM25":
        docs = [Counter(tokenize(t)) for t in texts]
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        for j, d in enumerate(docs):
            for term, tf in d.items():
                rows.append(vocab.setdefault(term, len(vocab)))
                cols.append(j)
                tfs.append(tf)
        n_docs = len(docs)
        lengths = np.array([sum(d.values()) for d in docs], dtype=np.float64)
        avgdl = lengths.mean() if n_docs else 1.0
        tf = np.array(tfs, dtype=np.float64)
        cols_a = np.array(cols)
        df = np.bincount(np.array(rows), minlength=len(vocab)).astype(np.float64)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        norm = self.k1 * (1 - self.b + self.b * lengths[cols_a] / avgdl)
        w = idf[np.array(rows)] * tf * (self.k1 + 1) / (tf + norm)
        self.vocab = vocab
        self.idf = idf
        self.matrix = sparse.csr_matrix((w, (rows, cols)), shape=(len(vocab), n_docs))
        self.n_docs = n_docs
        return self

    def scores(self, query: str) -> np.ndarray:
        ids = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not ids:
            return np.zeros(self.n_docs)
        # repeated query terms count once per occurrence, as in Lucene
        return np.asarray(self.matrix[ids].sum(axis=0)).ravel()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        # documents sharing no term with the query are not results
        return [(i, s) for i, s in top_k(self.scores(query), k) if s > 0]


def top_k(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    k = min(k, len(scores))
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [(int(i), float(scores[i])) for i in idx]

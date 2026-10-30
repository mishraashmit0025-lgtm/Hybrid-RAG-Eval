"""Retrieval evaluation on BEIR-format datasets."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import dedupe_docs, mean, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from .retriever import HybridRetriever
from .text import Chunk, chunk_document


@dataclass
class Dataset:
    name: str
    docs: dict[str, tuple[str, str]]  # id -> (title, text)
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]


def load_beir(path: str | Path, split: str = "test") -> Dataset:
    path = Path(path)
    docs = {}
    with open(path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["_id"]] = (d.get("title", ""), d.get("text", ""))
    queries = {}
    with open(path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    queries = {q: queries[q] for q in qrels if q in queries}
    return Dataset(path.name, docs, queries, qrels)


def chunk_corpus(ds: Dataset, max_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, (title, text) in ds.docs.items():
        chunks.extend(chunk_document(doc_id, text, title, max_words=max_words))
    return chunks


@dataclass
class MethodResult:
    method: str
    ndcg10: list[float] = field(default_factory=list)
    recall10: list[float] = field(default_factory=list)
    recall100: list[float] = field(default_factory=list)
    mrr10: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def row(self) -> dict:
        n = len(self.ndcg10)
        return {"method": self.method, "nDCG@10": mean(self.ndcg10), "Recall@10": mean(self.recall10),
                "Recall@100": mean(self.recall100), "MRR@10": mean(self.mrr10),
                "ms/query": 1000 * self.seconds / n if n else 0.0}


def evaluate(ds: Dataset, retriever: HybridRetriever, methods: list[str], depth: int = 100) -> list[MethodResult]:
    qids = list(ds.queries)
    texts = [ds.queries[q] for q in qids]
    t0 = time.perf_counter()
    qvecs = retriever.dense.encode_queries(texts) if retriever.dense is not None else None
    encode_s = time.perf_counter() - t0
    results = []
    # chunks per doc can be > 1, so ask for more chunks than docs needed at depth
    chunk_depth = depth * 3
    for m in methods:
        r = MethodResult(m)
        t0 = time.perf_counter()
        for i, qid in enumerate(qids):
            ranking = retriever.rank(texts[i], m, chunk_depth, qvec=qvecs[i] if qvecs is not None else None)
            docs = dedupe_docs([retriever.chunks[j].chunk_id for j, _ in ranking])[:depth]
            qrel = ds.qrels[qid]
            r.ndcg10.append(ndcg_at_k(docs, qrel, 10))
            r.recall10.append(recall_at_k(docs, qrel, 10))
            r.recall100.append(recall_at_k(docs, qrel, 100))
            r.mrr10.append(mrr_at_k(docs, qrel, 10))
        r.seconds = time.perf_counter() - t0 + (encode_s if m != "bm25" else 0.0)
        results.append(r)
    return results


def markdown_table(ds_name: str, results: list[MethodResult], n_chunks: int) -> str:
    lines = [f"### {ds_name} ({len(results[0].ndcg10)} queries, {n_chunks} chunks)", "",
             "| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | ms/query |", "|---|---|---|---|---|---|"]
    for r in results:
        row = r.row()
        lines.append(f"| {row['method']} | {row['nDCG@10']:.3f} | {row['Recall@10']:.3f} | {row['Recall@100']:.3f} "
                     f"| {row['MRR@10']:.3f} | {row['ms/query']:.1f} |")
    by = {r.method: r for r in results}
    notes = []
    for a, b in [("hybrid_rrf", "bm25"), ("hybrid_rrf", "dense"), ("hybrid_rrf_rerank", "hybrid_rrf")]:
        if a in by and b in by:
            p = paired_bootstrap(by[a].ndcg10, by[b].ndcg10)
            d = mean(by[a].ndcg10) - mean(by[b].ndcg10)
            notes.append(f"- {a} vs {b}: nDCG@10 {d:+.3f} (paired bootstrap p = {p:.3f})")
    if notes:
        lines += ["", *notes]
    return "\n".join(lines)

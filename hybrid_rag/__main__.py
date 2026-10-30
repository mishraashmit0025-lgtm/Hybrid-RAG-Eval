"""Command line.

    python -m hybrid_rag eval --data data/scifact
    python -m hybrid_rag ask --docs ./my_notes "How does the scheduler pick a worker?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .retriever import METHODS, HybridRetriever
from .text import chunk_document


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluate import chunk_corpus, evaluate, load_beir, markdown_table

    ds = load_beir(args.data, args.split)
    chunks = chunk_corpus(ds, args.chunk_words)
    print(f"{ds.name}: {len(ds.docs)} docs -> {len(chunks)} chunks, {len(ds.queries)} queries", file=sys.stderr)
    methods = args.methods.split(",")
    needs_dense = any(m != "bm25" for m in methods)
    retriever = HybridRetriever(chunks, dense_model=args.dense_model if needs_dense else None,
                                reranker_model=args.reranker if "hybrid_rrf_rerank" in methods else None,
                                cache_dir=args.cache, rerank_depth=args.rerank_depth)
    results = evaluate(ds, retriever, methods)
    table = markdown_table(ds.name, results, len(chunks))
    print(table)
    if args.out:
        Path(args.out).write_text(json.dumps({"dataset": ds.name, "dense_model": args.dense_model,
                                              "reranker": args.reranker, "results": [r.row() for r in results]},
                                             indent=2), encoding="utf-8")


def cmd_ask(args: argparse.Namespace) -> None:
    from .answer import answer

    chunks = []
    for p in sorted(Path(args.docs).rglob("*")):
        if p.suffix.lower() in (".txt", ".md", ".rst") and p.is_file():
            chunks.extend(chunk_document(str(p.relative_to(args.docs)), p.read_text(encoding="utf-8", errors="ignore"),
                                         title=p.name, max_words=args.chunk_words))
    if not chunks:
        sys.exit(f"no .txt/.md/.rst files under {args.docs}")
    retriever = HybridRetriever(chunks, cache_dir=args.cache)
    hits = retriever.search(args.question, k=args.k)
    if args.retrieve_only:
        for h in hits:
            print(f"{h.score:8.3f}  {h.chunk.chunk_id}  {h.chunk.text[:120]!r}")
        return
    print(answer(args.question, hits, model=args.model).render())


def main() -> None:
    ap = argparse.ArgumentParser(prog="hybrid_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="evaluate retrieval on a BEIR-format dataset")
    e.add_argument("--data", required=True, help="dataset folder with corpus.jsonl, queries.jsonl, qrels/")
    e.add_argument("--split", default="test")
    e.add_argument("--methods", default=",".join(METHODS))
    e.add_argument("--dense-model", default="BAAI/bge-small-en-v1.5")
    e.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    e.add_argument("--rerank-depth", type=int, default=30)
    e.add_argument("--chunk-words", type=int, default=200)
    e.add_argument("--cache", default=".cache", help="embedding cache directory")
    e.add_argument("--out", help="write results JSON here")
    e.set_defaults(fn=cmd_eval)

    a = sub.add_parser("ask", help="answer a question over a folder of text files, with citations")
    a.add_argument("--docs", required=True)
    a.add_argument("question")
    a.add_argument("-k", type=int, default=6)
    a.add_argument("--model", default="claude-opus-5-5")
    a.add_argument("--chunk-words", type=int, default=200)
    a.add_argument("--cache", default=".cache")
    a.add_argument("--retrieve-only", action="store_true", help="print retrieved chunks, no LLM call")
    a.set_defaults(fn=cmd_ask)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

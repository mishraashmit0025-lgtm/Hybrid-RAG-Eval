import math
from types import SimpleNamespace

import numpy as np
import pytest

from hybrid_rag.answer import build_request, parse_response
from hybrid_rag.bm25 import BM25
from hybrid_rag.fusion import minmax_fusion, rrf
from hybrid_rag.metrics import dedupe_docs, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from hybrid_rag.retriever import Hit, HybridRetriever
from hybrid_rag.text import Chunk, chunk_document, tokenize


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The running dogs are barking at THE cats") == ["run", "dog", "bark", "cat"]
    assert tokenize("COVID-19 vaccines") == ["covid-19", "vaccin"]


def test_bm25_matches_hand_computation():
    docs = ["apple banana", "apple apple cherry", "durian"]
    bm = BM25(k1=1.2, b=0.75).fit(docs)
    s = bm.scores("apple")
    n, df, avgdl = 3, 2, (2 + 3 + 1) / 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def w(tf, dl):
        return idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))

    assert s[0] == pytest.approx(w(1, 2))
    assert s[1] == pytest.approx(w(2, 3))
    assert s[2] == 0
    assert [i for i, _ in bm.search("apple", 10)] == [1, 0]  # no zero-score results


def test_bm25_rare_terms_weigh_more():
    bm = BM25().fit(["common rare", "common", "common", "common"])
    assert bm.search("common rare", 1)[0][0] == 0


def test_rrf_rewards_agreement():
    a = [(1, 9.0), (2, 8.0), (3, 7.0)]
    b = [(3, 0.9), (1, 0.8), (4, 0.7)]
    fused = rrf([a, b], k=60)
    assert fused[0][0] == 1  # ranks 1 and 2
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert {d for d, _ in fused} == {1, 2, 3, 4}


def test_minmax_fusion():
    fused = minmax_fusion([[(1, 10.0), (2, 0.0)], [(2, 1.0), (1, 0.0)]], weights=[0.7, 0.3])
    assert fused[0] == (1, pytest.approx(0.7))


def test_metrics():
    ranked = ["d3", "d1", "d9", "d2"]
    qrels = {"d1": 1, "d2": 2}
    assert recall_at_k(ranked, qrels, 2) == 0.5
    assert recall_at_k(ranked, qrels, 4) == 1.0
    assert mrr_at_k(ranked, qrels, 10) == 0.5
    dcg = 1 / math.log2(3) + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(ranked, qrels, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["d2", "d1"], qrels, 10) == pytest.approx(1.0)
    assert dedupe_docs(["a#0", "b#1", "a#2", "c#0"]) == ["a", "b", "c"]


def test_bootstrap_detects_real_difference_only():
    rng = np.random.default_rng(0)
    base = rng.random(200).tolist()
    better = [x + 0.1 for x in base]
    noisy = [x + rng.normal(0, 0.05) for x in base]
    assert paired_bootstrap(better, base, iters=2000) < 0.01
    assert paired_bootstrap(noisy, base, iters=2000) > 0.05


def test_chunker_respects_limit_and_overlaps():
    sents = [f"Sentence number {i} has exactly seven words." for i in range(30)]
    chunks = chunk_document("doc", " ".join(sents), "T", max_words=40, overlap_sents=1)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 40 for c in chunks)
    assert [c.chunk_id for c in chunks[:2]] == ["doc#0", "doc#1"]
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split(". ")[-1].rstrip(".") in b.text  # last sentence repeated
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sents)


def test_short_document_is_one_chunk():
    assert len(chunk_document("d", "One short sentence.", max_words=200)) == 1


def test_bm25_only_retriever():
    chunks = [Chunk("a", "a#0", "raft leader election uses randomized timeouts"),
              Chunk("b", "b#0", "bloom filters give false positives but no false negatives"),
              Chunk("c", "c#0", "leaders send heartbeats to followers")]
    r = HybridRetriever(chunks, dense_model=None, reranker_model=None)
    hits = r.search("how is a leader elected in raft", k=2)
    assert hits[0].chunk.doc_id == "a"


def test_citation_request_and_parsing():
    hits = [Hit(Chunk("guide.md", "guide.md#0", "Workers lease jobs for 30 seconds.", "guide.md"), 1.0),
            Hit(Chunk("faq.md", "faq.md#3", "Leases can be extended by heartbeats.", "faq.md"), 0.5)]
    req = build_request("How long is a lease?", hits, "claude-opus-5-5")
    content = req["messages"][0]["content"]
    assert [b["type"] for b in content] == ["document", "document", "text"]
    assert all(b["citations"] == {"enabled": True} for b in content[:2])
    assert content[0]["source"]["data"] == "Workers lease jobs for 30 seconds."

    msg = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="A lease lasts 30 seconds", citations=[
                SimpleNamespace(document_index=0, cited_text="Workers lease jobs for 30 seconds.")]),
            SimpleNamespace(type="text", text=" and can be extended.", citations=[
                SimpleNamespace(document_index=1, cited_text="Leases can be extended by heartbeats.")]),
        ],
        stop_reason="end_turn", usage=SimpleNamespace(input_tokens=100, output_tokens=12))
    ans = parse_response(msg, hits)
    assert ans.text == "A lease lasts 30 seconds and can be extended."
    assert [c.chunk_id for c in ans.citations] == ["guide.md#0", "faq.md#3"]
    assert "guide.md#0" in ans.render()

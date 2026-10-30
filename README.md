# Hybrid RAG Eval

Retrieval for RAG, built from the parts and measured. It covers BM25, dense embeddings, two ways of fusing them, and cross-encoder reranking, all evaluated on standard BEIR benchmarks with significance tests, plus a question-answering front end that uses Claude's citations feature so every sentence of an answer points to the exact source text it came from.

Most RAG tutorials pick a retrieval setup and move on. This project asks which choices actually help, by how much, and whether the difference is real or noise.

## Results

Documents are split on sentence boundaries into chunks of at most 200 words, with one sentence of overlap, and the title is prepended to each chunk. A ranking of chunks is collapsed into a ranking of documents (each document takes the rank of its best chunk) before scoring against the BEIR relevance judgments. Embeddings come from `BAAI/bge-small-en-v1.5` and the reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Everything ran on a laptop CPU.

**SciFact** (scientific claim verification, 5,183 abstracts, 300 test queries)

| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.668 | 0.805 | 0.906 | 0.629 | 0.9 |
| Dense | 0.718 | 0.834 | 0.947 | 0.689 | 9.9 |
| Hybrid, min-max fusion | **0.730** | **0.845** | **0.976** | **0.698** | 10.7 |
| Hybrid, RRF | 0.714 | 0.837 | 0.969 | 0.681 | 10.7 |
| Hybrid, RRF + rerank top 30 | 0.705 | 0.833 | 0.969 | 0.675 | 1530 |

**NFCorpus** (medical questions over nutrition research, 3,633 documents, 323 test queries)

| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| BM25 | 0.315 | 0.147 | 0.240 | 0.515 | 0.7 |
| Dense | 0.346 | 0.163 | 0.310 | 0.537 | 5.0 |
| Hybrid, min-max fusion | 0.358 | 0.170 | 0.313 | 0.565 | 5.7 |
| Hybrid, RRF | 0.355 | **0.172** | **0.314** | 0.563 | 5.7 |
| Hybrid, RRF + rerank top 30 | **0.359** | 0.170 | **0.314** | **0.577** | 1540 |

Paired bootstrap tests on per-query nDCG@10 (10,000 resamples):

| Comparison | SciFact | NFCorpus |
|---|---|---|
| Hybrid RRF vs BM25 | +0.046, p < 0.001 | +0.040, p < 0.001 |
| Hybrid RRF vs dense | -0.004, p = 0.70 | +0.008, p = 0.33 |
| Rerank vs hybrid RRF | -0.009, p = 0.53 | +0.004, p = 0.51 |

What the numbers say:

- **Hybrid clearly beats keyword search.** On both datasets the gain over BM25 is about 4 to 5 nDCG points, and it is far outside the noise. Recall@100 improves the most (0.906 to 0.976 on SciFact), which is the number that matters when retrieval feeds a reranker or a long-context model.
- **Hybrid vs dense alone is not significant at this size.** With a strong embedding model, fusion adds about 1 point and the bootstrap can't tell it from noise on 300 queries. It is still a cheap safety net for exact terms (drug names, gene IDs, error codes) that embeddings blur.
- **The off-the-shelf reranker does not pay for itself here.** A MiniLM cross-encoder trained on MS MARCO web queries adds about 1.5 seconds per query on CPU and gives no significant gain on scientific and medical text. That is a domain-shift result, not a claim that reranking never helps, and it is exactly the kind of thing worth measuring before shipping a reranker.
- **BM25 is implemented faithfully.** 0.668 on SciFact and 0.315 on NFCorpus are close to the published BEIR BM25 baselines (0.665 and 0.325), which is a useful check that the evaluation code is right.

Reproduce with `python -m hybrid_rag eval --data data/scifact --out results/scifact.json`. Raw numbers are in `results/`.

## How it works

```
documents ─> chunk (sentences, max 200 words, 1 sentence overlap, title prepended)
               ├─> BM25 index (Snowball stemming, stopwords, k1=0.9, b=0.4)
               └─> bge-small embeddings (normalized, cached on disk by content hash)

query ─> BM25 top 100 ─┐
      └> dense top 100 ┴─> fusion (RRF k=60, or min-max score blend) ─> [cross-encoder rerank] ─> top k
                                                                                                 │
                                         Claude, one document block per chunk, citations on <───┘
                                                 │
                                    answer text + exact cited spans per sentence
```

| File | Role |
|---|---|
| `text.py` | Tokenizer (Snowball stemmer via PyStemmer, with a fallback), sentence-aware overlapping chunker |
| `bm25.py` | Sparse BM25 using a SciPy CSR matrix, so scoring is one sparse dot product |
| `dense.py` | Sentence-transformers encoder with the bge query prefix and an on-disk embedding cache |
| `fusion.py` | Reciprocal rank fusion and min-max score fusion |
| `rerank.py` | Cross-encoder reranking of the fused top N |
| `retriever.py` | `HybridRetriever` with all five methods behind one `search(query, k, method)` |
| `metrics.py` | nDCG, recall, MRR, chunk-to-document collapsing, paired bootstrap |
| `evaluate.py` | BEIR loader, evaluation loop, results table |
| `answer.py` | Grounded answers with Claude's document citations |

### Answers with real citations

`answer.py` sends each retrieved chunk as its own `document` content block with `citations: {"enabled": true}`. The response comes back as text blocks, each carrying the document index and the exact cited text it relies on. Citations therefore point to real source spans, instead of `[1]` markers the model typed and could get wrong. The system prompt tells the model to say when the sources don't contain the answer, rather than answering from memory.

```bash
export ANTHROPIC_API_KEY=...
python -m hybrid_rag ask --docs ./handbook "How many days of parental leave do new parents get?"
python -m hybrid_rag ask --docs ./handbook --retrieve-only "parental leave"   # no API call
```

## Setup

```bash
pip install -e ".[dense,llm]"
python scripts/download_beir.py scifact nfcorpus
python -m hybrid_rag eval --data data/scifact
python -m hybrid_rag eval --data data/nfcorpus --methods bm25,dense,hybrid_rrf
```

## Tests

`pytest -q` runs offline in a few seconds: BM25 against a hand-computed score, IDF behaviour, RRF and min-max fusion, every metric against worked examples, the bootstrap (it must find a real difference and must not find one between identical systems), chunking limits and overlap, and building the citations request and parsing a citations response.

## License

MIT

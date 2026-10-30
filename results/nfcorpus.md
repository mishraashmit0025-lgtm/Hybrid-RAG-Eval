### nfcorpus (323 queries, 6106 chunks)

| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| bm25 | 0.315 | 0.147 | 0.240 | 0.515 | 0.7 |
| dense | 0.346 | 0.163 | 0.310 | 0.537 | 5.0 |
| hybrid_minmax | 0.358 | 0.170 | 0.313 | 0.565 | 5.7 |
| hybrid_rrf | 0.355 | 0.172 | 0.314 | 0.563 | 5.7 |
| hybrid_rrf_rerank | 0.359 | 0.170 | 0.314 | 0.577 | 1539.7 |

- hybrid_rrf vs bm25: nDCG@10 +0.040 (paired bootstrap p = 0.000)
- hybrid_rrf vs dense: nDCG@10 +0.008 (paired bootstrap p = 0.330)
- hybrid_rrf_rerank vs hybrid_rrf: nDCG@10 +0.004 (paired bootstrap p = 0.508)

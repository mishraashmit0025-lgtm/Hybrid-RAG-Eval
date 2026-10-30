### scifact (300 queries, 7755 chunks)

| Method | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| bm25 | 0.668 | 0.805 | 0.906 | 0.629 | 0.9 |
| dense | 0.718 | 0.834 | 0.947 | 0.689 | 9.9 |
| hybrid_minmax | 0.730 | 0.845 | 0.976 | 0.698 | 10.7 |
| hybrid_rrf | 0.714 | 0.837 | 0.969 | 0.681 | 10.7 |
| hybrid_rrf_rerank | 0.705 | 0.833 | 0.969 | 0.675 | 1529.5 |

- hybrid_rrf vs bm25: nDCG@10 +0.046 (paired bootstrap p = 0.000)
- hybrid_rrf vs dense: nDCG@10 -0.004 (paired bootstrap p = 0.703)
- hybrid_rrf_rerank vs hybrid_rrf: nDCG@10 -0.009 (paired bootstrap p = 0.529)

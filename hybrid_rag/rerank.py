"""Second-stage reranking with a cross-encoder.

A bi-encoder embeds query and passage separately, which is fast but loses
word-level interaction. A cross-encoder reads them together and scores the pair,
which is much more accurate and much slower, so it only sees the top candidates."""

from __future__ import annotations

from .fusion import Ranking


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32,
                 device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Ranking, texts: list[str], top_n: int | None = None) -> Ranking:
        if not candidates:
            return []
        pairs = [(query, texts[i]) for i, _ in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        out = sorted(((i, float(s)) for (i, _), s in zip(candidates, scores)), key=lambda kv: -kv[1])
        return out[:top_n] if top_n else out

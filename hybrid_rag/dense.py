"""Dense retrieval with sentence-transformers embeddings and exact cosine search."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .bm25 import top_k

# bge models are trained with an instruction prefix on the query side only
QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class DenseIndex:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: str | Path | None = None,
                 batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.prefix = QUERY_PREFIX.get(model_name, "")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.vectors: np.ndarray | None = None

    def _cache_path(self, texts: list[str]) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha256(self.model_name.encode())
        for t in texts:
            h.update(t.encode())
            h.update(b"\x00")
        return self.cache_dir / f"emb-{h.hexdigest()[:16]}.npy"

    def fit(self, texts: list[str], show_progress: bool = False) -> "DenseIndex":
        path = self._cache_path(texts)
        if path is not None and path.exists():
            self.vectors = np.load(path)
            return self
        self.vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                         convert_to_numpy=True, show_progress_bar=show_progress).astype(np.float32)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.vectors)
        return self

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.model.encode([self.prefix + q for q in queries], batch_size=self.batch_size,
                                 normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def scores_for(self, qvec: np.ndarray) -> np.ndarray:
        assert self.vectors is not None, "call fit() first"
        return self.vectors @ qvec  # vectors are unit length, so this is cosine similarity

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        return top_k(self.scores_for(self.encode_queries([query])[0]), k)

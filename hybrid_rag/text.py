"""Tokenization and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

# a small English stopword list; BM25 works noticeably better without these
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_SUFFIXES = ("ational", "ization", "fulness", "iveness", "ations", "ation", "ments", "ment", "ness", "ings", "ing",
             "edly", "ies", "ied", "es", "ed", "ly", "s")


def stem(word: str) -> str:
    """Fallback light suffix stripper, used only when PyStemmer is not installed.
    It conflates common plural and verb forms and leaves short words alone."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            return base + "y" if suf in ("ies", "ied") else base
    return word


try:  # Snowball (Porter 2) via PyStemmer when available; it is what Lucene-style BM25 baselines use
    import Stemmer

    _snowball = Stemmer.Stemmer("english")

    def _stem_all(words: list[str]) -> list[str]:
        return _snowball.stemWords(words)
except ImportError:  # pragma: no cover
    def _stem_all(words: list[str]) -> list[str]:
        return [stem(w) for w in words]


def tokenize(text: str) -> list[str]:
    return _stem_all([t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS])


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    title: str = ""


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def chunk_document(doc_id: str, text: str, title: str = "", max_words: int = 200, overlap_sents: int = 1) -> list[Chunk]:
    """Split on sentence boundaries into chunks of at most max_words words,
    repeating the last overlap_sents sentences at the start of the next chunk so
    a fact that straddles a boundary is still retrievable. Short documents stay whole."""
    sents = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    if not sents:
        return [Chunk(doc_id, f"{doc_id}#0", title, title)] if title else []
    chunks: list[Chunk] = []
    cur: list[str] = []
    words = 0
    i = 0
    while i < len(sents):
        n = len(sents[i].split())
        if cur and words + n > max_words:
            chunks.append(Chunk(doc_id, f"{doc_id}#{len(chunks)}", " ".join(cur), title))
            keep = cur[-overlap_sents:] if overlap_sents else []
            cur, words = list(keep), sum(len(s.split()) for s in keep)
            if words + n > max_words:  # overlap alone would overflow; drop it
                cur, words = [], 0
        cur.append(sents[i])
        words += n
        i += 1
    if cur:
        chunks.append(Chunk(doc_id, f"{doc_id}#{len(chunks)}", " ".join(cur), title))
    return chunks

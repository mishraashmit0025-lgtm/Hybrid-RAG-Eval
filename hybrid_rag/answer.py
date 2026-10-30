"""Grounded answers with citations, using Claude's document citations feature.

Each retrieved chunk is passed as its own `document` block with citations
enabled, so the response comes back split into text blocks that carry the exact
source spans they rely on. That makes "where did this sentence come from"
checkable, instead of trusting bracketed [1] markers the model typed itself."""

from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import Hit

SYSTEM = (
    "Answer the user's question using only the provided documents. "
    "If the documents do not contain the answer, say that you could not find it in the sources "
    "rather than answering from general knowledge. Be concise."
)


@dataclass
class Citation:
    chunk_id: str
    title: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def render(self) -> str:
        lines = [self.text.strip(), ""]
        seen = set()
        for c in self.citations:
            if (c.chunk_id, c.cited_text) in seen:
                continue
            seen.add((c.chunk_id, c.cited_text))
            lines.append(f"  [{c.chunk_id}] {c.title}: \"{c.cited_text.strip()[:160]}\"")
        return "\n".join(lines)


def build_request(question: str, hits: list[Hit], model: str, max_tokens: int = 4096) -> dict:
    docs = [{
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": h.chunk.text},
        "title": h.chunk.title or h.chunk.chunk_id,
        "context": f"chunk id {h.chunk.chunk_id}",
        "citations": {"enabled": True},
    } for h in hits]
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": [*docs, {"type": "text", "text": question}]}],
    }


def parse_response(message, hits: list[Hit]) -> Answer:
    parts: list[str] = []
    cites: list[Citation] = []
    for block in message.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for c in getattr(block, "citations", None) or []:
            i = getattr(c, "document_index", None)
            if i is not None and 0 <= i < len(hits):
                cites.append(Citation(hits[i].chunk.chunk_id, hits[i].chunk.title, c.cited_text))
    u = message.usage
    return Answer("".join(parts), cites, message.stop_reason, u.input_tokens, u.output_tokens)


def answer(question: str, hits: list[Hit], model: str = "claude-opus-5-5", client=None) -> Answer:
    import anthropic

    client = client or anthropic.Anthropic()
    msg = client.messages.create(**build_request(question, hits, model))
    if msg.stop_reason == "refusal":
        return Answer("The model declined to answer this question.", [], "refusal")
    return parse_response(msg, hits)

"""Orchestrates retrieval and validates every evidence reference before returning text."""

import json
import os
import threading
import time
from pathlib import Path

from rag_system.models import Answer
from rag_system.providers import (
    AnswerProvider,
    CrossEncoderReranker,
    Embedder,
    ExtractiveProvider,
    Reranker,
    SentenceEmbedder,
    configured_provider,
)
from rag_system.retrieval import retrieve
from rag_system.store import Store, digest


class KnowledgeBase:
    def __init__(
        self,
        path: Path,
        provider: AnswerProvider | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ):
        self.store = Store(path)
        self.provider = provider or ExtractiveProvider()
        self.embedder, self.reranker = embedder, reranker
        self.lock = threading.RLock()

    def ingest(self, name: str, data: bytes) -> dict:
        with self.lock:
            return self.store.ingest(name, data, self.embedder)

    def delete(self, document_id: str) -> bool:
        with self.lock:
            return self.store.delete(document_id)

    def ask(self, query: str, top_k: int = 5) -> Answer:
        query = query.strip()
        if not query or len(query) > 2000 or not 1 <= top_k <= 8:
            raise ValueError("Question must contain 1–2000 characters and top_k must be 1–8.")
        started = time.perf_counter()
        with self.lock:
            key = digest(
                json.dumps(
                    [
                        query,
                        top_k,
                        self.provider.identity,
                        self.embedder.identity if self.embedder else "none",
                        self.reranker.identity if self.reranker else "none",
                        "retrieval-v1",
                        [(d["id"], d["digest"]) for d in self.store.documents()],
                    ]
                )
            )
            with self.store.connect() as db:
                cached = db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            if cached:
                result = Answer.model_validate_json(cached[0])
                result.cached = True
            else:
                sources = retrieve(self.store, query, top_k, self.embedder, self.reranker)
                result = Answer(
                    status="abstained",
                    mode=self.provider.identity.split(":")[0],
                    sources=sources,
                    retrieval="hybrid-rrf" if self.embedder else "bm25",
                    reranker=self.reranker.identity if self.reranker else "none",
                )
                if not sources:
                    result.reason = (
                        "No matching evidence. Add a relevant document or rephrase the question."
                    )
                else:
                    draft = self.provider.generate(query, sources)
                    source_map = {s.id: s for s in sources}
                    valid = all(
                        e.source_id in source_map and e.quote in source_map[e.source_id].text
                        for c in draft.claims
                        for e in c.evidence
                    )
                    if draft.claims and valid:
                        result.status, result.claims = "answered", draft.claims
                    else:
                        result.reason = (
                            "The model did not provide sufficient verifiable evidence."
                            if valid
                            else "An evidence reference failed validation; the answer was withheld."
                        )
                with self.store.connect() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO cache VALUES(?,?)", (key, result.model_dump_json())
                    )
                    db.execute(
                        "DELETE FROM cache WHERE rowid NOT IN (SELECT rowid FROM cache ORDER BY rowid DESC LIMIT 128)"
                    )
            result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return result


def configured_kb() -> KnowledgeBase:
    embedding = os.getenv("RAG_EMBEDDING_MODEL", "")
    reranking = os.getenv("RAG_RERANK_MODEL", "")
    return KnowledgeBase(
        Path(os.getenv("RAG_DATA_DIR", ".data")) / "knowledge.sqlite3",
        configured_provider(),
        SentenceEmbedder(embedding) if embedding else None,
        CrossEncoderReranker(reranking) if reranking else None,
    )

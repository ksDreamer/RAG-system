"""BM25 candidates, optional dense RRF fusion, reranking and overlapping-chunk suppression."""

import json
import math

from rag_system.ingestion import tokens
from rag_system.models import Source
from rag_system.providers import Embedder, Reranker
from rag_system.store import Store


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a or not all(math.isfinite(x) for x in a + b):
        raise ValueError(
            "Embedding dimension or numeric value mismatch; use a versioned model identity."
        )
    norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0


def retrieve(
    store: Store,
    query: str,
    top_k: int = 5,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    min_similarity: float = 0.35,
) -> list[Source]:
    terms = list(dict.fromkeys(tokens(query)))[:64]
    if not terms:
        return []
    with store.connect() as db:
        lexical = [
            r[0]
            for r in db.execute(
                "SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 30",
                (" OR ".join('"' + t.replace('"', '""') + '"' for t in terms),),
            )
        ]
        rankings = [lexical]
        if embedder:
            rows = db.execute("SELECT id,content_hash,text FROM chunks").fetchall()
            store.ensure_embeddings(db, [(r[1], r[2]) for r in rows], embedder)
            if rows:
                q = embedder.encode([query])[0]
                vectors = {
                    r[0]: json.loads(r[1])
                    for r in db.execute(
                        "SELECT content_hash,vector FROM embeddings WHERE model=?",
                        (embedder.identity,),
                    )
                }
                dense = sorted(
                    [(r[0], cosine(q, vectors[r[1]])) for r in rows], key=lambda x: -x[1]
                )
                rankings.append([key for key, score in dense[:30] if score >= min_similarity])
        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, key in enumerate(ranking, 1):
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
        sources = []
        for key in sorted(scores, key=lambda k: (-scores[k], k))[:30]:
            row = dict(
                db.execute(
                    "SELECT c.*,d.name FROM chunks c JOIN documents d ON c.document_id=d.id WHERE c.id=?",
                    (key,),
                ).fetchone()
            )
            sources.append(Source(**row, score=scores[key]))
    if reranker and sources:
        sources = reranker.rank(query, sources)
    selected: list[Source] = []
    for s in sources:
        if len(s.text) < 8:
            continue
        if any(
            s.text == p.text
            or (
                s.document_id == p.document_id
                and s.page == p.page
                and max(0, min(s.end, p.end) - max(s.start, p.start))
                > 0.5 * min(s.end - s.start, p.end - p.start)
            )
            for p in selected
        ):
            continue
        selected.append(s)
        if len(selected) >= top_k:
            break
    return selected

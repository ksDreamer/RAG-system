"""SQLite is the authoritative index; text, FTS rows and cache updates are atomic."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rag_system.ingestion import CHUNKER_VERSION, chunk_pages, extract, tokens, validate_name
from rag_system.providers import Embedder


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL, digest TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER, section TEXT, start INTEGER, end INTEGER, text TEXT, content_hash TEXT
                );
                CREATE INDEX IF NOT EXISTS chunks_document ON chunks(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, terms, tokenize='porter unicode61');
                CREATE TRIGGER IF NOT EXISTS chunks_delete AFTER DELETE ON chunks BEGIN
                    DELETE FROM chunks_fts WHERE id=old.id;
                END;
                CREATE TABLE IF NOT EXISTS embeddings (
                    content_hash TEXT, model TEXT, vector TEXT NOT NULL,
                    PRIMARY KEY(content_hash, model)
                );
                CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def documents(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute("""
                SELECT d.*, count(c.id) AS chunks FROM documents d
                LEFT JOIN chunks c ON c.document_id=d.id GROUP BY d.id ORDER BY d.name
            """)
            ]

    def ingest(self, name: str, data: bytes, embedder: Embedder | None = None) -> dict:
        validate_name(name)
        fingerprint = digest(data + CHUNKER_VERSION.encode())
        doc_id = digest(name)[:24]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT digest FROM documents WHERE id=?", (doc_id,)).fetchone()
            if old and old[0] == fingerprint:
                return {"id": doc_id, "name": name, "status": "unchanged"}
            chunks = chunk_pages(extract(name, data))
            rows = [
                (
                    digest(f"{doc_id}:{fingerprint}:{i}"),
                    doc_id,
                    c.page,
                    c.section,
                    c.start,
                    c.end,
                    c.text,
                    digest(c.text),
                )
                for i, c in enumerate(chunks)
            ]
            if embedder:
                self.ensure_embeddings(db, [(r[7], r[6]) for r in rows], embedder)
            db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            db.execute(
                "INSERT INTO documents(id,name,digest) VALUES(?,?,?)", (doc_id, name, fingerprint)
            )
            db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", rows)
            db.executemany(
                "INSERT INTO chunks_fts(id,terms) VALUES(?,?)",
                [(r[0], " ".join(tokens(r[6]))) for r in rows],
            )
            self.invalidate(db)
        return {
            "id": doc_id,
            "name": name,
            "status": "updated" if old else "created",
            "chunks": len(rows),
        }

    def delete(self, doc_id: str) -> bool:
        with self.connect() as db:
            removed = db.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount > 0
            if removed:
                self.invalidate(db)
            return removed

    @staticmethod
    def invalidate(db):
        db.execute("DELETE FROM cache")
        db.execute(
            "DELETE FROM embeddings WHERE content_hash NOT IN (SELECT content_hash FROM chunks)"
        )

    @staticmethod
    def ensure_embeddings(db, items: list[tuple[str, str]], embedder: Embedder):
        missing = {
            key: text
            for key, text in items
            if not db.execute(
                "SELECT 1 FROM embeddings WHERE content_hash=? AND model=?",
                (key, embedder.identity),
            ).fetchone()
        }
        if missing:
            import math

            vectors = embedder.encode(list(missing.values()))
            if len(vectors) != len(missing) or not vectors or not vectors[0]:
                raise ValueError("Embedding provider returned the wrong number of vectors.")
            dim = len(vectors[0])
            if any(
                len(v) != dim or not all(math.isfinite(x) for x in v) or sum(x * x for x in v) == 0
                for v in vectors
            ):
                raise ValueError("Embedding provider returned invalid vectors.")
            db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES(?,?,?)",
                [
                    (key, embedder.identity, json.dumps(v))
                    for key, v in zip(missing, vectors, strict=True)
                ],
            )

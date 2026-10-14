"""SQLite metadata + FTS5/BM25 + stored dense vectors, for a small local corpus."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.models import Chunk, DocumentInfo


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL, page_count INTEGER NOT NULL,
                    embedding_tag TEXT NOT NULL, pipeline_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id)
                    ON DELETE CASCADE, source TEXT NOT NULL, page INTEGER NOT NULL,
                    section TEXT NOT NULL, text TEXT NOT NULL, search_text TEXT NOT NULL,
                    vector TEXT
                );
                CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(document_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED, search_text, tokenize='porter unicode61'
                );
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def documents(self) -> list[DocumentInfo]:
        with self.connect() as conn:
            return [
                DocumentInfo(**dict(row))
                for row in conn.execute("SELECT * FROM documents ORDER BY created_at DESC, id")
            ]

    def document(self, doc_id: str) -> DocumentInfo | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return DocumentInfo(**dict(row)) if row else None

    def insert(self, document: DocumentInfo, chunks: list[Chunk], vectors) -> bool:
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match chunk count")
        with self.connect() as conn:
            inserted = conn.execute(
                "INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?)",
                tuple(document.model_dump().values()),
            ).rowcount
            if not inserted:
                return False
            for i, chunk in enumerate(chunks):
                conn.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.source,
                        chunk.page,
                        chunk.section,
                        chunk.text,
                        chunk.search_text,
                        json.dumps(vectors[i]) if vectors else None,
                    ),
                )
                conn.execute("INSERT INTO chunks_fts VALUES (?,?)", (chunk.id, chunk.search_text))
            return True

    def delete(self, doc_id: str) -> bool:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN "
                "(SELECT id FROM chunks WHERE document_id=?)",
                (doc_id,),
            )
            return bool(conn.execute("DELETE FROM documents WHERE id=?", (doc_id,)).rowcount)

    @staticmethod
    def chunk(row):
        return Chunk(**{field: row[field] for field in Chunk.model_fields})

    def lexical(self, query: str, document_ids: list[str], limit: int) -> list[Chunk]:
        if not query:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.* FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.chunk_id "
                f"WHERE chunks_fts MATCH ? AND c.document_id IN ({placeholders}) "
                "ORDER BY bm25(chunks_fts), c.id LIMIT ?",
                (query, *document_ids, limit),
            ).fetchall()
        return [self.chunk(row) for row in rows]

    def dense_rows(self, document_ids: list[str]):
        placeholders = ",".join("?" for _ in document_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE document_id IN ({placeholders}) "
                "AND vector IS NOT NULL",
                document_ids,
            ).fetchall()
        return [(self.chunk(row), json.loads(row["vector"])) for row in rows]

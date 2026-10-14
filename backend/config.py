"""Explicit configuration; no provider is created at import time."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "clause-pages-v1"


@dataclass(frozen=True)
class Settings:
    db_path: Path = ROOT / "runtime" / "contracts.sqlite3"
    mode: str = "demo"
    openai_api_key: str = ""
    chat_model: str = "gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    reranker: str = "none"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_words: int = 220
    overlap_words: int = 30
    candidate_k: int = 20
    top_k: int = 5
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pages: int = 300
    max_chunks: int = 3000
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")

    def __post_init__(self):
        if self.mode not in {"demo", "openai"}:
            raise ValueError("RAG_MODE must be demo or openai")
        if self.reranker not in {"none", "cross-encoder"}:
            raise ValueError("RAG_RERANKER must be none or cross-encoder")
        if not 0 <= self.overlap_words < self.chunk_words <= 500:
            raise ValueError("Chunk size must be 1..500 words; overlap must be smaller")
        if not 1 <= self.top_k <= self.candidate_k <= 100:
            raise ValueError("Require 1 <= top_k <= candidate_k <= 100")
        if self.mode == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required in openai mode")

    @property
    def embedding_tag(self) -> str:
        return f"openai:{self.embedding_model}" if self.mode == "openai" else "none"

    @classmethod
    def from_env(cls):
        load_dotenv(ROOT / ".env", override=False)
        return cls(
            db_path=Path(os.getenv("RAG_DB_PATH", str(ROOT / "runtime/contracts.sqlite3"))),
            mode=os.getenv("RAG_MODE", "demo"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            chat_model=os.getenv("RAG_CHAT_MODEL", "gpt-5-mini"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            reranker=os.getenv("RAG_RERANKER", "none"),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            chunk_words=int(os.getenv("RAG_CHUNK_WORDS", "220")),
            overlap_words=int(os.getenv("RAG_OVERLAP_WORDS", "30")),
            candidate_k=int(os.getenv("RAG_CANDIDATE_K", "20")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            cors_origins=tuple(
                origin.strip()
                for origin in os.getenv(
                    "RAG_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
                ).split(",")
                if origin.strip()
            ),
        )

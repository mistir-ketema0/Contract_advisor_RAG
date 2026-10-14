"""Run: uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import Settings
from backend.errors import InputError
from backend.models import AnswerResponse, DocumentInfo, IngestResponse, QuestionRequest
from backend.service import RAGService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, service: RAGService | None = None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        app.state.rag = service or RAGService(settings)
        yield

    app = FastAPI(title="Contract Advisor RAG", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(Exception)
    async def provider_error(request, exc):
        logger.error("Request failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Request failed. Check provider configuration and server availability."
            },
        )

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": settings.mode,
            "version": "0.2.0",
            "embedding_tag": settings.embedding_tag,
        }

    @app.get("/documents", response_model=list[DocumentInfo])
    def documents(request: Request):
        return request.app.state.rag.store.documents()

    @app.post("/documents", response_model=IngestResponse)
    def upload(request: Request, file: UploadFile):
        raw = file.file.read(settings.max_upload_bytes + 1)
        filename = Path((file.filename or "contract.pdf").replace("\\", "/")).name
        if len(filename) > 200:
            raise HTTPException(400, "Filename is too long")
        try:
            return request.app.state.rag.ingest(filename, raw)
        except InputError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/documents/{document_id}", status_code=204)
    def delete(document_id: str, request: Request):
        if not request.app.state.rag.store.delete(document_id):
            raise HTTPException(404, "Document not found")

    @app.post("/chat", response_model=AnswerResponse)
    def chat(body: QuestionRequest, request: Request):
        try:
            return request.app.state.rag.answer(body.question, body.document_ids)
        except InputError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app

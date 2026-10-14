"""Page-aware, clause-oriented chunks. Only source text enters the evidence index."""

import hashlib
import io
import re

from pypdf import PdfReader

from backend.config import PIPELINE_VERSION, Settings
from backend.errors import InputError
from backend.models import Chunk

HEADING = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+|ARTICLE\s+\w+|SECTION\s+\w+)", re.I)


def parse_document(filename: str, raw: bytes, settings: Settings) -> list[tuple[int, str]]:
    if not raw:
        raise InputError("The document is empty")
    if len(raw) > settings.max_upload_bytes:
        raise InputError("Document exceeds the upload size limit")
    suffix = filename.rsplit(".", 1)[-1].lower()
    if suffix == "pdf":
        if not raw.startswith(b"%PDF-"):
            raise InputError("File is not a PDF")
        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                raise InputError("Encrypted PDFs are not supported")
            if len(reader.pages) > settings.max_pages:
                raise InputError("PDF exceeds the page limit")
            pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except InputError:
            raise
        except Exception as exc:
            raise InputError("Unable to parse this PDF") from exc
    elif suffix in {"txt", "md"}:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InputError("Text files must use UTF-8") from exc
        pages = list(enumerate(text.split("\f"), start=1))
        if len(pages) > settings.max_pages:
            raise InputError("Document exceeds the page limit")
    else:
        raise InputError("Supported formats: PDF, TXT and Markdown")
    if not any(text.strip() for _, text in pages):
        raise InputError("No extractable text found. Scanned PDFs need OCR before ingestion")
    return pages


def document_id(raw: bytes, settings: Settings) -> tuple[str, str]:
    content_hash = hashlib.sha256(raw).hexdigest()
    fingerprint = (
        f"{PIPELINE_VERSION}:{settings.chunk_words}:{settings.overlap_words}:"
        f"{settings.embedding_tag}"
    )
    return hashlib.sha256(f"{content_hash}:{fingerprint}".encode()).hexdigest()[:24], content_hash


def chunk_pages(pages, doc_id: str, filename: str, settings: Settings) -> list[Chunk]:
    chunks = []
    section = "Preamble"

    def emit(page, label, lines):
        words = " ".join(lines).split()
        start = 0
        while start < len(words):
            end = min(start + settings.chunk_words, len(words))
            text = " ".join(words[start:end])
            chunks.append(
                Chunk(
                    id=f"{doc_id}:{len(chunks)}",
                    document_id=doc_id,
                    source=filename,
                    page=page,
                    section=label,
                    text=text,
                    search_text=f"Document: {filename}\nSection: {label}\nPage: {page}\n{text}",
                )
            )
            if len(chunks) > settings.max_chunks:
                raise InputError("Document produces too many chunks")
            if end == len(words):
                break
            start = end - settings.overlap_words

    for page, text in pages:
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip().lstrip("# ")
            if not line:
                continue
            if len(line.split()) <= 14 and HEADING.match(line):
                if lines:
                    emit(page, section, lines)
                section, lines = line, [line]
            else:
                lines.append(line)
        if lines:
            emit(page, section, lines)
    if not chunks:
        raise InputError("No text chunks could be produced")
    return chunks

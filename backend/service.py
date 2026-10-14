import logging
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from backend.config import PIPELINE_VERSION
from backend.ingestion import chunk_pages, document_id, parse_document
from backend.models import AnswerResponse, Citation, DocumentInfo, IngestResponse
from backend.providers import DemoProvider, OpenAIProvider
from backend.retrieval import Retriever
from backend.store import Store

logger = logging.getLogger(__name__)
ABSTENTION = "I couldn't find sufficient verified evidence in the selected contracts to answer."


def normalize(text):
    return " ".join(text.split())


def validate_answer(draft, hits):
    """Fail closed on unknown IDs, missing citations or invented quotes.

    Quote matching is structural validation, not proof of semantic entailment.
    """
    if draft.insufficient_evidence or not draft.statements:
        return ABSTENTION, [], True
    available = {hit.chunk.id: hit.chunk for hit in hits}
    citations, sentences = [], []
    for statement in draft.statements:
        if not statement.text.strip() or not statement.evidence:
            return ABSTENTION, [], True
        numbers = []
        for ref in statement.evidence:
            chunk = available.get(ref.chunk_id)
            quote = normalize(ref.quote)
            if not chunk or len(quote) < 8 or quote not in normalize(chunk.text):
                return ABSTENTION, [], True
            existing = next(
                (c for c in citations if (c.chunk_id, c.quote) == (chunk.id, quote)), None
            )
            if existing:
                numbers.append(existing.number)
            else:
                number = len(citations) + 1
                citations.append(
                    Citation(
                        number=number,
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        source=chunk.source,
                        page=chunk.page,
                        section=chunk.section,
                        quote=quote,
                    )
                )
                numbers.append(number)
        sentences.append(
            statement.text.strip()
            + " "
            + "".join(f"[{number}]" for number in dict.fromkeys(numbers))
        )
    return "\n\n".join(sentences), citations, False


class RAGService:
    def __init__(self, settings, embeddings=None, generator=None):
        self.settings = settings
        self.store = Store(settings.db_path)
        provider = None
        if settings.mode == "openai" and (embeddings is None or generator is None):
            provider = OpenAIProvider(settings)
        self.embeddings = embeddings or provider
        self.generator = generator or provider or DemoProvider()
        self.retriever = Retriever(self.store, settings, self.embeddings)

    def ingest(self, filename, raw):
        doc_id, content_hash = document_id(raw, self.settings)
        if document := self.store.document(doc_id):
            return IngestResponse(document=document, reused=True)
        pages = parse_document(filename, raw, self.settings)
        chunks = chunk_pages(pages, doc_id, filename, self.settings)
        vectors = (
            self.embeddings.embed([chunk.search_text for chunk in chunks])
            if self.embeddings
            else None
        )
        document = DocumentInfo(
            id=doc_id,
            filename=filename,
            content_hash=content_hash,
            chunk_count=len(chunks),
            page_count=len(pages),
            embedding_tag=self.settings.embedding_tag,
            pipeline_version=PIPELINE_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        inserted = self.store.insert(document, chunks, vectors)
        return IngestResponse(document=self.store.document(doc_id), reused=not inserted)

    def answer(self, question, document_ids):
        started = perf_counter()
        request_id = uuid4().hex
        hits = self.retriever.search(question, document_ids)
        retrieved = perf_counter()
        if hits:
            draft, usage = self.generator.generate(question, hits)
            answer, citations, insufficient = validate_answer(draft, hits)
        else:
            answer, citations, insufficient, usage = ABSTENTION, [], True, {}
        finished = perf_counter()
        timings = {
            "retrieval": round((retrieved - started) * 1000, 2),
            "generation": round((finished - retrieved) * 1000, 2),
            "total": round((finished - started) * 1000, 2),
        }
        logger.info(
            "rag_request id=%s mode=%s hits=%d abstained=%s total_ms=%.2f",
            request_id,
            self.settings.mode,
            len(hits),
            insufficient,
            timings["total"],
        )
        return AnswerResponse(
            answer=answer,
            insufficient_evidence=insufficient,
            citations=citations,
            mode=self.settings.mode,
            retrieval_method=self.retriever.method,
            timings_ms=timings,
            request_id=request_id,
            usage=usage,
        )

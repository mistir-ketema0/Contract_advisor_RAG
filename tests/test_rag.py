import io
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from backend.app import create_app
from backend.config import ROOT, Settings
from backend.ingestion import chunk_pages, parse_document
from backend.models import AnswerDraft, EvidenceRef, Statement
from backend.providers import DemoProvider, OpenAIProvider
from backend.retrieval import cosine, reciprocal_rank_fusion
from backend.service import RAGService, validate_answer

SAMPLE = ROOT / "examples/sample-contract.txt"
DISTRACTOR = ROOT / "examples/distractor-contract.txt"


@pytest.fixture
def settings(tmp_path):
    return Settings(db_path=tmp_path / "contracts.sqlite3")


@pytest.fixture
def service(settings):
    return RAGService(settings)


def test_api_full_flow_upload_answer_abstain_delete(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["mode"] == "demo"
        uploaded = client.post("/documents", files={"file": (SAMPLE.name, SAMPLE.read_bytes())})
        assert uploaded.status_code == 200
        doc_id = uploaded.json()["document"]["id"]
        assert client.get("/documents").json()[0]["id"] == doc_id
        answer = client.post(
            "/chat",
            json={"question": "What is the termination notice period?", "document_ids": [doc_id]},
        )
        assert answer.status_code == 200
        result = answer.json()
        assert "60 days" in result["answer"]
        assert result["citations"][0]["section"] == "5. Termination"
        assert result["citations"][0]["page"] == 1
        assert result["request_id"] and result["timings_ms"]["total"] >= 0
        missing = client.post(
            "/chat",
            json={"question": "What is the annual insurance premium?", "document_ids": [doc_id]},
        )
        assert missing.json()["insufficient_evidence"] is True
        assert missing.json()["citations"] == []
        assert client.delete(f"/documents/{doc_id}").content == b""
        assert client.get("/documents").json() == []
        assert (
            client.post(
                "/chat", json={"question": "termination", "document_ids": [doc_id]}
            ).status_code
            == 400
        )


@pytest.mark.parametrize(
    "question,ids", [(" ", ["missing"]), ("termination", []), ("x" * 2001, ["a"])]
)
def test_api_validates_requests(settings, question, ids):
    with TestClient(create_app(settings)) as client:
        assert (
            client.post("/chat", json={"question": question, "document_ids": ids}).status_code
            == 422
        )


def test_persistence_and_idempotent_ingestion(service, settings):
    original = service.ingest(SAMPLE.name, SAMPLE.read_bytes())
    duplicate = service.ingest(SAMPLE.name, SAMPLE.read_bytes())
    assert not original.reused and duplicate.reused
    reopened = RAGService(settings)
    assert len(reopened.store.documents()) == 1
    assert "60 days" in reopened.answer("termination notice", [original.document.id]).answer


def test_document_filter_prevents_conflicting_answers(service):
    target = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    other = service.ingest(DISTRACTOR.name, DISTRACTOR.read_bytes()).document.id
    first = service.answer("termination notice", [target])
    second = service.answer("termination notice", [other])
    assert "60 days" in first.answer and "7 days" not in first.answer
    assert "7 days" in second.answer and "60 days" not in second.answer
    assert all(c.document_id == target for c in first.citations)
    assert all(c.document_id == other for c in second.citations)


def test_chunking_preserves_pages_and_bounds(settings):
    chunks = chunk_pages(
        [(1, "1. Fees\n" + "payment " * 70), (2, "2. Termination\n" + "notice " * 70)],
        "doc",
        "a.txt",
        replace(settings, chunk_words=25, overlap_words=5),
    )
    assert {c.page for c in chunks} == {1, 2}
    assert all(0 < len(c.text.split()) <= 25 for c in chunks)
    assert all(c.section == ("1. Fees" if c.page == 1 else "2. Termination") for c in chunks)
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]


def pdf_bytes(text=True):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
        )
        stream = DecodedStreamObject()
        stream.set_data(
            b"BT /F1 12 Tf 50 700 Td (1. Termination) Tj 0 -20 Td "
            b"(Either party may terminate with 60 days of written notice.) Tj ET"
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_real_pdf_parser_and_api(settings):
    with TestClient(create_app(settings)) as client:
        uploaded = client.post("/documents", files={"file": ("contract.pdf", pdf_bytes())})
        assert uploaded.status_code == 200
        answer = client.post(
            "/chat",
            json={
                "question": "termination notice",
                "document_ids": [uploaded.json()["document"]["id"]],
            },
        )
        assert "60 days" in answer.json()["answer"]
        assert answer.json()["citations"][0]["page"] == 1


@pytest.mark.parametrize(
    "filename,raw",
    [
        ("x.exe", b"text"),
        ("x.pdf", b"not a pdf"),
        ("x.txt", b""),
        ("x.txt", b"\xff"),
        ("x.pdf", pdf_bytes(False)),
    ],
)
def test_bad_uploads_are_rejected_without_indexing(settings, filename, raw):
    with TestClient(create_app(settings)) as client:
        assert client.post("/documents", files={"file": (filename, raw)}).status_code == 400
        assert client.get("/documents").json() == []


def test_oversize_document(settings):
    with pytest.raises(ValueError, match="size limit"):
        parse_document("x.txt", b"abcde", replace(settings, max_upload_bytes=4))


@pytest.mark.parametrize("kind", ["unknown_source", "fake_quote", "no_citation", "partial_bad"])
def test_invalid_citations_fail_closed(service, kind):
    doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    hits = service.retriever.search("termination notice", [doc])
    ref = EvidenceRef(chunk_id=hits[0].chunk.id, quote=hits[0].chunk.text)
    if kind == "unknown_source":
        ref.chunk_id = "invented"
    elif kind == "fake_quote":
        ref.quote = "The contract provides free termination with no notice."
    statements = [
        Statement(
            text="The notice period is 60 days.", evidence=[] if kind == "no_citation" else [ref]
        )
    ]
    if kind == "partial_bad":
        statements.append(Statement(text="No liability cap applies.", evidence=[]))
    answer, citations, insufficient = validate_answer(
        AnswerDraft(insufficient_evidence=False, statements=statements), hits
    )
    assert insufficient and not citations and "60 days" not in answer


def test_no_query_or_hypothetical_answer_is_indexed(service):
    doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    with service.store.connect() as conn:
        count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    service.answer("Which bananas are intergalactic?", [doc])
    with service.store.connect() as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == count
        assert not conn.execute("SELECT 1 FROM chunks WHERE text LIKE '%intergalactic%'").fetchone()


class FakeEmbeddings:
    def __init__(self):
        self.ingestion_calls = 0
        self.query_calls = 0

    def embed(self, texts):
        self.ingestion_calls += 1
        return [[1.0, 0.0] if "invoice" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_query(self, question):
        self.query_calls += 1
        return [1.0, 0.0] if "bill" in question else [0.0, 1.0]


def test_hybrid_uses_dense_and_lexical_without_reembedding(settings):
    embeddings = FakeEmbeddings()
    configured = replace(settings, mode="openai", openai_api_key="test-not-a-secret")
    service = RAGService(configured, embeddings=embeddings, generator=DemoProvider())
    doc = service.ingest("a.txt", b"1. Payment\nInvoices are payable in thirty days.").document.id
    service.ingest("a.txt", b"1. Payment\nInvoices are payable in thirty days.")
    hits = service.retriever.search("How soon must I settle my bill?", [doc])
    assert hits[0].dense_rank == 1
    hits2 = service.retriever.search("Invoices", [doc])
    assert hits2[0].lexical_rank == 1 and hits2[0].dense_rank == 1
    assert embeddings.ingestion_calls == 1 and embeddings.query_calls == 2


def test_embedding_mode_mismatch_is_rejected(service, settings):
    doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    other = RAGService(
        replace(settings, mode="openai", openai_api_key="test"),
        embeddings=FakeEmbeddings(),
        generator=DemoProvider(),
    )
    with pytest.raises(ValueError, match="different embedding mode"):
        other.retriever.search("termination", [doc])


def test_rank_fusion_boosts_consensus(service):
    doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    chunks = service.store.lexical('"agreement"', [doc], 3)
    a, b, c = chunks[:3]
    fused = reciprocal_rank_fusion({"lexical": [b, a], "dense": [c, a]}, 3)
    assert fused[0].chunk.id == a.id
    assert fused[0].lexical_rank == 2 and fused[0].dense_rank == 2
    assert cosine([1, 0], [1, 0]) == 1
    with pytest.raises(ValueError, match="dimensions"):
        cosine([1, 0], [1])


def test_provider_failure_does_not_leak_details(settings):
    class BrokenGenerator:
        def generate(self, question, hits):
            raise ValueError("sensitive-contract-text API_KEY=example")

    service = RAGService(settings, generator=BrokenGenerator())
    doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    with TestClient(create_app(settings, service), raise_server_exceptions=False) as client:
        response = client.post("/chat", json={"question": "termination", "document_ids": [doc]})
        assert response.status_code == 503
        assert "sensitive" not in response.text and "API_KEY" not in response.text


def test_cors_is_restricted(settings):
    with TestClient(create_app(settings)) as client:
        response = client.options(
            "/chat",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers


def test_openai_adapter_contract(monkeypatch, settings, service):
    calls = []

    def embedding_create(**kwargs):
        calls.append(("embedding", kwargs))
        return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])])

    draft = AnswerDraft(insufficient_evidence=True, statements=[])

    def responses_parse(**kwargs):
        calls.append(("response", kwargs))
        return SimpleNamespace(
            output_parsed=draft, usage=SimpleNamespace(input_tokens=10, output_tokens=2)
        )

    monkeypatch.setattr(
        "openai.OpenAI",
        lambda **kwargs: SimpleNamespace(
            embeddings=SimpleNamespace(create=embedding_create),
            responses=SimpleNamespace(parse=responses_parse),
        ),
    )
    provider = OpenAIProvider(replace(settings, mode="openai", openai_api_key="test"))
    provider.embed_query("question")
    provider.embed_query("question")
    assert len(calls) == 1  # Identical queries use the bounded in-process cache.
    doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    provider.generate("question", service.retriever.search("termination", [doc]))
    response_call = calls[-1][1]
    assert response_call["text_format"] is AnswerDraft
    assert response_call["store"] is False
    assert response_call["model"] == settings.chat_model
    assert "untrusted" in response_call["input"][0]["content"]


def test_optional_reranker_scores_candidates_before_top_k(monkeypatch, settings):
    import sys

    class FakeCrossEncoder:
        def __init__(self, model):
            assert model == settings.reranker_model

        def predict(self, pairs):
            assert len(pairs) > 1
            return [10.0 if "1. Parties" in text else -10.0 for _, text in pairs]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    )
    service = RAGService(replace(settings, reranker="cross-encoder", top_k=1))
    doc = service.ingest(SAMPLE.name, SAMPLE.read_bytes()).document.id
    hits = service.retriever.search("party", [doc])
    assert len(hits) == 1
    assert hits[0].chunk.section == "1. Parties" and hits[0].rerank_score == 10.0
    assert service.retriever.method == "bm25+cross-encoder"

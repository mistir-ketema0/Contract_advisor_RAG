from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[str] = Field(min_length=1, max_length=50)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank")
        return value


class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_hash: str
    chunk_count: int
    page_count: int
    embedding_tag: str
    pipeline_version: str
    created_at: str


class IngestResponse(BaseModel):
    document: DocumentInfo
    reused: bool


class Chunk(BaseModel):
    id: str
    document_id: str
    source: str
    page: int
    section: str
    text: str
    search_text: str


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    lexical_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class EvidenceRef(BaseModel):
    chunk_id: str
    quote: str


class Statement(BaseModel):
    text: str
    evidence: list[EvidenceRef]


class AnswerDraft(BaseModel):
    """Provider output; citations are checked again outside the model."""

    insufficient_evidence: bool
    statements: list[Statement]


class Citation(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    source: str
    page: int
    section: str
    quote: str


class AnswerResponse(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: list[Citation]
    mode: Literal["demo", "openai"]
    retrieval_method: str
    timings_ms: dict[str, float]
    request_id: str
    usage: dict[str, int] = Field(default_factory=dict)

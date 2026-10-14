# Design decisions and next steps

## Scope

A reproducible local contract Q&A application with an inspectable evidence path. The default demo uses no cloud service. AI mode performs hybrid retrieval and structured generation with replaceable provider adapters.

## Decisions

| Decision | Reason | Tradeoff |
| --- | --- | --- |
| Separate ingestion and querying | Avoid re-extracting/re-embedding contracts and polluting evidence with queries | Configuration changes need re-ingestion |
| Content hash + pipeline/model fingerprint | Deduplicate and keep model-specific indexes distinct | Renamed identical content retains its first ingested filename |
| Clause/page-aware splitting | Preserve context and source locations useful for reviewing contracts | Heuristics cannot reconstruct complex PDF layout |
| Metadata prefixes during embedding/search | Retain document and clause context without invented summaries | Does not implement full LLM contextual retrieval |
| BM25 + dense retrieval with RRF | Cover exact identifiers and semantic paraphrases without mixing raw score scales | Candidate counts/fusion still need empirical tuning |
| Optional cross-encoder after candidate retrieval | Spend joint query/passage scoring on a bounded candidate set | Model download, CPU cost and general-domain bias |
| Typed model output + per-statement evidence | Make source validation possible and the result predictable for the UI | Schema validity and quote presence do not prove correctness |
| Fail closed on invalid evidence | Reject made-up references or partially uncited answers | One invalid statement rejects the whole answer |
| Explicit demo/OpenAI modes | Permit a low-friction local demonstration; avoid silent quality changes | Demo extraction cannot resolve paraphrases or semantic answerability |
| SQLite + exact vector scan | Install and test locally with minimal infrastructure | Single-machine corpus; no ANN index, access control or distributed scaling |
| Fixed public synthetic fixture | Make both successes and failures reproducible | Too small/simple to support legal quality or market performance claims |

The earlier HyDE implementation inserted the question plus hypothetical answer into the evidence index. HyDE, when used, should guide retrieval through a query representation; it should not make model-generated text an authoritative contract source. The updated pipeline indexes source text only.

## Implementation boundaries

- Parsing/chunking: `backend/ingestion.py`.
- Storage and atomic insert/delete: `backend/store.py`.
- Hybrid fusion, document filters and optional reranking: `backend/retrieval.py`.
- External model calls and offline extraction: `backend/providers.py`.
- Citation validation, ingestion reuse and request timing: `backend/service.py`.
- HTTP validation and API contract: `backend/app.py` / `backend/models.py`.
- Reproducible retrieval/generation experiments: `backend/evaluation.py`.

Embedding calls use bounded batches. Query embeddings use a bounded per-process cache. Model timeouts/retries are explicit. SQLite connections are scoped per operation, use WAL and foreign keys, and insert document/chunk/FTS records in one transaction. Repeated concurrent ingestion may still duplicate provider work before the database deduplicates insertion; a job-level lock would avoid that at scale.

## Priority roadmap

1. **Measure real retrieval/generation:** run lexical/dense/hybrid/reranked ablations with the same embeddings, fixed corpus and held-out legal QA labels. Track clause/span recall, MRR/nDCG, citation correctness, unsupported-answer rate, latency and token cost. Record model snapshots and exact packages; include failures and confidence intervals.
2. **Improve parsing:** integrate a layout-aware parser/OCR, retain section hierarchies and table units, and add parser fixtures for multi-page clauses, tables, scanned PDFs and multilingual text.
3. **Improve answerability:** use annotated negative questions, semantic citation/entailment review and calibrated evidence checks. Test prompt injection and conflicting clauses. Do not present a retrieval similarity or reranker score as answer confidence.
4. **Scale ingestion/storage:** move ingestion to durable background jobs, add object storage for authorized source files, and use Qdrant or a PostgreSQL vector index behind the retrieval boundary when exact scans no longer meet measured latency needs. Preserve document filters, model versioning and deletion semantics.
5. **Prepare for multi-user use:** authenticate, authorize document access before retrieval, add tenant filters in every storage/search operation, rate limits, request-body limits at the proxy, malicious-PDF isolation, encryption/retention policy and audit trails without plaintext contracts.
6. **Extend workflows selectively:** add bounded tools for comparing selected versions or assembling a clause checklist, only after measuring their benefit against this fixed retrieval pipeline. Each tool must return traceable sources. Broader autonomous legal decisions are outside the project scope.

## Portfolio claims supported by the implementation

- Built a React/FastAPI contract Q&A application with file ingestion, document selection and source inspection.
- Separated ingestion from retrieval using persistent storage, content/configuration hashes and model-specific indexes.
- Implemented scoped keyword/dense retrieval with rank fusion and optional cross-encoder reranking.
- Added typed model output, statement-level citations and server-side source/quote validation.
- Created a public synthetic regression benchmark, measured an offline baseline, and added API/browser tests and CI.

Avoid claims of production readiness, legal reliability, proven hybrid accuracy improvement, or live model performance until those systems have been run and evaluated with supporting artifacts. [Demo results](../reports/demo-evaluation.json) show the current offline baseline, including failures.

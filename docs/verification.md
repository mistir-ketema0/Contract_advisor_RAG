# Verification

Story: a user uploads a contract, selects its index, asks a question, and sees the answer with an original page/clause quote.

## Verification boundaries

- Backend regression tests cover real PDF parsing, persistence, repeated ingestion, document filters, invalid uploads, citation rejection, model/index incompatibility, provider failure handling and restricted CORS.
- Dense retrieval and rank fusion are tested using controlled vectors. The OpenAI adapter and optional reranker wiring use test doubles, not external API calls or a downloaded neural model.
- Browser tests cover the actual React → Vite proxy → FastAPI → SQLite → response → rendered citation path and mobile missing-evidence state.
- `reports/demo-evaluation.json` contains a measured offline regression baseline on synthetic public fixtures.

Live OpenAI generation, embedding quality and downloaded cross-encoder quality require explicit model-enabled runs. The public synthetic benchmark is not a validated legal benchmark.

## Local verification result

- Python 3.12: **26 backend tests passed**; Ruff lint and format checks passed.
- Frontend: Vite production build passed.
- Browser: **one end-to-end test passed**, exercising desktop upload/question/citation and the mobile missing-evidence state; no page errors were observed.
- Screenshot: `docs/demo.png` captures the real local app with the synthetic contract.
- Optional cross-encoder: wiring tested with a double; neural inference not run.
- OpenAI: SDK method signature and structured schema checked, adapter tested with a double; paid requests not run.
- CI is configured for Python 3.11/3.12/3.13; this local run used 3.12 only.

The container's standard browser download was unavailable. Local browser verification used a bundled Chromium executable with a single-process launch override. The regular Playwright setup remains the documented/CI path.

"""Reproducible synthetic benchmark. Live mode makes explicit paid provider calls."""

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from backend.config import ROOT, Settings
from backend.service import RAGService


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 3)


def run(settings, benchmark, strategy="hybrid", generate=False):
    rows = []
    with tempfile.TemporaryDirectory(prefix="contract-rag-eval-") as directory:
        service = RAGService(replace(settings, db_path=Path(directory) / "evaluation.sqlite3"))
        sample = ROOT / "examples/sample-contract.txt"
        primary = service.ingest(sample.name, sample.read_bytes()).document.id
        # A distractor with conflicting terms tests whether selection remains scoped.
        distractor = ROOT / "examples/distractor-contract.txt"
        service.ingest(distractor.name, distractor.read_bytes())
        for item in benchmark:
            start = perf_counter()
            hits = service.retriever.search(item["question"], [primary], strategy=strategy)
            latency = (perf_counter() - start) * 1000
            expected = item["expected_sections"]
            found = [hit.chunk.section for hit in hits]
            ranks = [i + 1 for i, section in enumerate(found) if section in expected]
            covered = len(set(expected) & set(found))
            row = {
                "id": item["id"],
                "question": item["question"],
                "answerable": item["answerable"],
                "expected_sections": expected,
                "retrieved_sections": found,
                "retrieval_ms": round(latency, 3),
                "hit_at_k": bool(ranks) if expected else None,
                "reciprocal_rank": 1 / min(ranks) if ranks else (0 if expected else None),
                "section_recall_at_k": covered / len(expected) if expected else None,
            }
            if generate:
                # Reuse precisely the evaluated retrieval strategy/candidates, avoiding a
                # second retrieval with a different configuration.
                from backend.models import AnswerDraft
                from backend.service import validate_answer

                draft, usage = (
                    service.generator.generate(item["question"], hits)
                    if hits
                    else (AnswerDraft(insufficient_evidence=True, statements=[]), {})
                )
                answer, citations, insufficient = validate_answer(draft, hits)
                row.update(
                    answer=answer,
                    insufficient_evidence=insufficient,
                    citations=[citation.model_dump() for citation in citations],
                    usage=usage,
                    expected_fragment_present=bool(
                        item["expected_fragment"]
                        and item["expected_fragment"].lower() in answer.lower()
                    ),
                )
            rows.append(row)
    positives = [row for row in rows if row["answerable"]]
    summary = {
        "total_questions": len(rows),
        "answerable_questions": len(positives),
        "unanswerable_questions": len(rows) - len(positives),
        "hit_at_k": statistics.mean(row["hit_at_k"] for row in positives),
        "mrr_at_k": statistics.mean(row["reciprocal_rank"] for row in positives),
        "section_recall_at_k": statistics.mean(row["section_recall_at_k"] for row in positives),
        "retrieval_p50_ms": percentile([row["retrieval_ms"] for row in rows], 0.5),
        "retrieval_p95_ms": percentile([row["retrieval_ms"] for row in rows], 0.95),
    }
    if generate:
        negatives = [row for row in rows if not row["answerable"]]
        summary["answerable_fragment_match_rate"] = statistics.mean(
            row["expected_fragment_present"] for row in positives
        )
        summary["unanswerable_abstention_rate"] = (
            statistics.mean(row["insufficient_evidence"] for row in negatives)
            if negatives
            else None
        )
        summary["non_abstained_answers"] = sum(not row["insufficient_evidence"] for row in rows)
    return {"summary": summary, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["demo", "openai"], default="demo")
    parser.add_argument("--strategy", choices=["lexical", "dense", "hybrid"], default="lexical")
    parser.add_argument(
        "--generate", action="store_true", help="Also generate answers (paid in openai mode)"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="Use optional cross-encoder; downloads a model"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evaluation.json")
    args = parser.parse_args()
    if args.mode == "demo" and args.strategy == "dense":
        parser.error("Dense retrieval requires --mode openai")
    base = Settings.from_env()
    settings = replace(base, mode=args.mode, reranker="cross-encoder" if args.rerank else "none")
    data = ROOT / "benchmarks/synthetic-qa.json"
    raw = data.read_bytes()
    result = run(settings, json.loads(raw), args.strategy, args.generate)
    result["run"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "mode": args.mode,
        "strategy": args.strategy,
        "generation": args.generate,
        "chat_model": settings.chat_model if args.mode == "openai" else None,
        "embedding_tag": settings.embedding_tag,
        "reranker": settings.reranker,
        "candidate_k": settings.candidate_k,
        "top_k": settings.top_k,
        "chunk_words": settings.chunk_words,
        "overlap_words": settings.overlap_words,
        "benchmark_sha256": hashlib.sha256(raw).hexdigest(),
    }
    result["limitations"] = [
        "Tiny synthetic fixture; not a held-out legal benchmark or proof of legal accuracy.",
        "Relevance labels are clause sections, not manually annotated spans.",
        "Fragment match is a regression check, not a semantic correctness metric.",
        "Citation validation checks source IDs and quotes, not semantic entailment.",
        "Do not compare demo extraction with LLM generation as if they were the same system.",
        "No historic improvement percentage: original inputs and environment were unavailable.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

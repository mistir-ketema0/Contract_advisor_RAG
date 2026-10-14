"""Scoped hybrid retrieval with reciprocal rank fusion and optional cross-encoder."""

import math
import re
from collections import defaultdict
from threading import Lock

from backend.errors import InputError
from backend.models import SearchHit

STOPWORDS = frozenset(
    """a an the is are was were be been being to of for in on at by
with and or as it its this that these those what which who when where how why do does
did can could would should will shall may please tell me about contract agreement
under according document you your i we they their from if then any there have has
had much many get""".split()
)


def terms(question):
    return list(
        dict.fromkeys(
            term
            for term in re.findall(r"\b[a-z0-9]+\b", question.lower())
            if term not in STOPWORDS and len(term) > 1
        )
    )


def fts_query(question):
    return " OR ".join(f'"{term}"' for term in terms(question)[:64])


def cosine(a, b):
    if len(a) != len(b):
        raise InputError("Embedding dimensions differ; re-ingest using the current model")
    norm = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def reciprocal_rank_fusion(rankings, limit, constant=60):
    scores = defaultdict(float)
    chunks, ranks = {}, defaultdict(dict)
    for label, ranking in rankings.items():
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] += 1 / (constant + rank)
            chunks[chunk.id] = chunk
            ranks[chunk.id][f"{label}_rank"] = rank
    return [
        SearchHit(chunk=chunks[key], score=scores[key], **ranks[key])
        for key in sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    ]


class Retriever:
    def __init__(self, store, settings, embeddings=None):
        self.store, self.settings, self.embeddings = store, settings, embeddings
        self._reranker = None
        self._model_lock = Lock()

    def search(self, question, document_ids, strategy="hybrid", rerank=True):
        if strategy not in {"hybrid", "lexical", "dense"}:
            raise InputError("Unknown retrieval strategy")
        for doc_id in document_ids:
            document = self.store.document(doc_id)
            if not document:
                raise InputError("Selected document does not exist")
            if document.embedding_tag != self.settings.embedding_tag:
                raise InputError("Selected document uses a different embedding mode; re-ingest it")
        rankings = {}
        if strategy != "dense":
            rankings["lexical"] = self.store.lexical(
                fts_query(question), document_ids, self.settings.candidate_k
            )
        if strategy != "lexical" and self.embeddings:
            vector = self.embeddings.embed_query(question)
            candidates = sorted(
                ((cosine(vector, v), c) for c, v in self.store.dense_rows(document_ids)),
                key=lambda item: (-item[0], item[1].id),
            )
            rankings["dense"] = [c for _, c in candidates[: self.settings.candidate_k]]
        elif strategy == "dense":
            raise InputError("Dense retrieval requires openai mode")
        hits = reciprocal_rank_fusion(rankings, self.settings.candidate_k)
        if rerank and hits and self.settings.reranker == "cross-encoder":
            with self._model_lock:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder

                    self._reranker = CrossEncoder(self.settings.reranker_model)
                scores = self._reranker.predict([(question, hit.chunk.search_text) for hit in hits])
            for hit, score in zip(hits, scores):
                hit.rerank_score = float(score)
            hits.sort(key=lambda hit: (-hit.rerank_score, hit.chunk.id))
        return hits[: self.settings.top_k]

    @property
    def method(self):
        base = "bm25+dense+rrf" if self.embeddings else "bm25"
        return f"{base}+cross-encoder" if self.settings.reranker == "cross-encoder" else base

from collections.abc import Mapping, Sequence
from typing import Any

from opensearchpy import OpenSearch
from pydantic import BaseModel, ConfigDict, Field

from haeindex.index import INDEX
from haeindex.ollama import Ollama

TOP_K = 5
CANDIDATE_K = 50
RRF_K = 60
DEDUPE_JACCARD = 0.7
BOOSTS = {
    "title": 3.0,
    "path": 1.5,
    "text": 1.0,
    "title_phrase": 4.0,
    "text_phrase": 2.0,
}
SOURCE_EXCLUDE = ["embedding"]


class LegHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int
    score: float


class Hit(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    fused: float
    source: dict[str, Any]
    legs: dict[str, LegHit] = Field(default_factory=dict)


class Result(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    hits: list[Hit] = Field(default_factory=list)
    candidates: dict[str, int] = Field(default_factory=dict)
    dropped: int = 0
    degraded: list[str] = Field(default_factory=list)

    def top_score(self, leg: str) -> float:
        return self.hits[0].legs[leg].score if self.hits and leg in self.hits[0].legs else 0.0


def doc_filter(doc_ids: Sequence[str]) -> list[dict[str, Any]]:
    return [{"terms": {"doc_id": list(doc_ids)}}] if doc_ids else []


def bm25_body(query: str, filters: Sequence[dict[str, Any]], size: int) -> dict[str, Any]:
    b = BOOSTS
    return {
        "size": size,
        "_source": {"excludes": SOURCE_EXCLUDE},
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "type": "best_fields",
                            "tie_breaker": 0.3,
                            "fields": [
                                f"title^{b['title']}",
                                f"path^{b['path']}",
                                f"text^{b['text']}",
                            ],
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "type": "phrase",
                            "slop": 2,
                            "fields": [
                                f"title^{b['title_phrase']}",
                                f"text^{b['text_phrase']}",
                            ],
                        }
                    },
                ],
                "filter": list(filters),
                "minimum_should_match": 1,
            }
        },
    }


def knn_body(
    vector: Sequence[float], filters: Sequence[dict[str, Any]], size: int
) -> dict[str, Any]:
    knn: dict[str, Any] = {"vector": list(vector), "k": size}
    if filters:
        knn["filter"] = {"bool": {"filter": list(filters)}}
    return {
        "size": size,
        "_source": {"excludes": SOURCE_EXCLUDE},
        "query": {"knn": {"embedding": knn}},
    }


def char_ngrams(text: str, n: int = 3) -> set[str]:
    t = "".join(text.split())
    return {t[i : i + n] for i in range(max(0, len(t) - n + 1))}


def dedupe(hits: Sequence[Hit], jaccard: float = DEDUPE_JACCARD) -> tuple[list[Hit], int]:
    kept: list[Hit] = []
    grams: list[set[str]] = []
    dropped = 0
    for h in hits:
        g = char_ngrams(h.source.get("body", ""))
        if any(g and k and len(g & k) / len(g | k) >= jaccard for k in grams):
            dropped += 1
            continue
        kept.append(h)
        grams.append(g)
    return (kept, dropped)


def rrf_fuse(results: Mapping[str, Sequence[dict[str, Any]]], rrf_k: int = RRF_K) -> list[Hit]:
    canon: dict[str, dict[str, Any]] = {}
    for rows in results.values():
        for h in rows:
            canon.setdefault(h["_source"]["chunk_id"], h["_source"])

    scores = dict.fromkeys(canon, 0.0)
    legs: dict[str, dict[str, LegHit]] = {cid: {} for cid in canon}
    for leg, rows in results.items():
        for rank, h in enumerate(rows, 1):
            cid = h["_source"]["chunk_id"]
            scores[cid] += 1.0 / (rrf_k + rank)
            legs[cid][leg] = LegHit(rank=rank, score=float(h.get("_score") or 0.0))

    order = sorted(canon, key=lambda c: (-scores[c], c))
    return [Hit(chunk_id=c, fused=scores[c], source=canon[c], legs=legs[c]) for c in order]


def search(
    os_client: OpenSearch,
    query: str,
    *,
    embedder: Ollama | None = None,
    doc_ids: Sequence[str] = (),
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    index: str = INDEX,
    extra_queries: Sequence[str] = (),
) -> Result:
    filters = doc_filter(doc_ids)
    bodies: dict[str, dict[str, Any]] = {"bm25": bm25_body(query, filters, candidate_k)}
    degraded: list[str] = []
    for i, extra in enumerate(extra_queries):
        bodies[f"bm25+{i}"] = bm25_body(extra, filters, candidate_k)

    if embedder is None:
        degraded.append("knn(임베더 없음)")
    else:
        try:
            vecs = embedder.embed([query, *extra_queries])
            bodies["knn"] = knn_body(vecs[0], filters, candidate_k)
            for i, vec in enumerate(vecs[1:]):
                bodies[f"knn+{i}"] = knn_body(vec, filters, candidate_k)
        except Exception as e:
            degraded.append(f"knn(임베딩 실패: {type(e).__name__})")

    results: dict[str, list[dict[str, Any]]] = {}
    for leg, body in bodies.items():
        results[leg] = os_client.search(index=index, body=body)["hits"]["hits"]

    fused = rrf_fuse(results)
    kept, dropped = dedupe(fused)
    return Result(
        query=query,
        hits=kept[:top_k],
        candidates={leg: len(rows) for leg, rows in results.items()},
        dropped=dropped,
        degraded=degraded,
    )

import json
import math
import re
import statistics
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch
from pydantic import BaseModel, ConfigDict, Field

from haeindex.bedrock import Bedrock
from haeindex.index import INDEX
from haeindex.search import CANDIDATE_K, RRF_K, Hit, char_ngrams
from haeindex.search import search as run_search

FEATURES = (
    "bm25_rr",
    "knn_rr",
    "bm25_z",
    "knn_cos",
    "both_legs",
    "title_gram",
    "body_gram",
    "depth_norm",
    "len_norm",
    "doc_share",
)
TRAIN_DOCS_HELD_OUT = 2
NEG_DEPTH = 20
HANGUL = re.compile(r"[가-힣]")
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def script_of(text: str) -> str:
    han, cjk, lat = (
        len(HANGUL.findall(text)),
        len(CJK.findall(text)),
        len(LATIN.findall(text)),
    )
    if cjk > han and cjk > 3:
        return "cjk"
    if han:
        return "han"
    if lat:
        return "latin"
    return "other"


def gram_overlap(query: str, text: str) -> float:
    q = char_ngrams(query)
    if not q:
        return 0.0
    return len(q & char_ngrams(text)) / len(q)


def featurize(query: str, hits: Sequence[Hit]) -> list[list[float]]:
    bm25 = [h.legs["bm25"].score for h in hits if "bm25" in h.legs]
    mean = statistics.fmean(bm25) if bm25 else 0.0
    sd = statistics.pstdev(bm25) if len(bm25) > 1 else 0.0
    share: dict[str, int] = {}
    for h in hits:
        d = str(h.source.get("doc_id", ""))
        share[d] = share.get(d, 0) + 1
    n = len(hits) or 1

    rows = []
    for h in hits:
        s = h.source
        b, k = h.legs.get("bm25"), h.legs.get("knn")
        rows.append(
            [
                1.0 / (RRF_K + b.rank) if b else 0.0,
                1.0 / (RRF_K + k.rank) if k else 0.0,
                (b.score - mean) / sd if b and sd else 0.0,
                k.score if k else 0.0,
                1.0 if b and k else 0.0,
                gram_overlap(query, str(s.get("title", ""))),
                gram_overlap(query, str(s.get("body", ""))[:1200]),
                min(int(s.get("depth", 0)), 4) / 4.0,
                min(int(s.get("token_len", 0)), 512) / 512.0,
                share.get(str(s.get("doc_id", "")), 0) / n,
            ]
        )
    return rows


class Row(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    query: str
    label: int
    feats: list[float]


class Model(BaseModel):
    model_config = ConfigDict(frozen=True)

    features: list[str] = Field(default_factory=lambda: list(FEATURES))
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]
    n_rows: int
    n_pos: int
    train_docs: list[str]
    held_out: list[str]
    lang_match: bool
    l2: float
    iters: int
    loss: float

    def score(self, feats: Sequence[float]) -> float:
        z = self.bias + sum(
            w * (x - m) / (s or 1.0)
            for w, x, m, s in zip(self.weights, feats, self.mean, self.std, strict=True)
        )
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def coefficients(self) -> list[tuple[str, float]]:
        return sorted(zip(self.features, self.weights, strict=True), key=lambda kv: -abs(kv[1]))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Model":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def chunks_with_queries(
    os_client: OpenSearch, *, doc_ids: Sequence[str] = (), index: str = INDEX
) -> Iterator[dict[str, Any]]:
    after: list[Any] | None = None
    filters: list[dict[str, Any]] = [{"exists": {"field": "queries"}}]
    if doc_ids:
        filters.append({"terms": {"doc_id": list(doc_ids)}})
    while True:
        body: dict[str, Any] = {
            "size": 200,
            "_source": ["chunk_id", "doc_id", "queries", "text"],
            "sort": [{"chunk_id": "asc"}],
            "query": {"bool": {"filter": filters}},
        }
        if after:
            body["search_after"] = after
        hits = os_client.search(index=index, body=body)["hits"]["hits"]
        if not hits:
            return
        for h in hits:
            yield h["_source"]
        after = hits[-1]["sort"]


class Harvest(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: list[Row] = Field(default_factory=list)
    n_queries: int = 0
    skipped_lang: int = 0
    skipped_missing: int = 0
    baseline_rr: list[float] = Field(default_factory=list)


def harvest(
    os_client: OpenSearch,
    ol: Bedrock,
    *,
    doc_ids: Sequence[str] = (),
    per_chunk: int = 2,
    lang_match: bool = True,
    neg_depth: int = NEG_DEPTH,
    candidate_k: int = CANDIDATE_K,
    index: str = INDEX,
) -> Harvest:
    rows: list[Row] = []
    n_queries = skipped_lang = skipped_missing = 0
    baseline: list[float] = []
    for src in chunks_with_queries(os_client, doc_ids=doc_ids, index=index):
        want = script_of(str(src.get("text", "")))
        queries = [q.strip() for q in str(src.get("queries", "")).splitlines() if q.strip()]
        used = 0
        for q in queries:
            if used >= per_chunk:
                break
            if lang_match and script_of(q) != want:
                skipped_lang += 1
                continue
            used += 1
            n_queries += 1
            res = run_search(
                os_client,
                q,
                embedder=ol,
                top_k=neg_depth,
                candidate_k=candidate_k,
                index=index,
            )
            ids = [h.chunk_id for h in res.hits]
            if src["chunk_id"] not in ids:
                skipped_missing += 1
                continue
            baseline.append(1.0 / (ids.index(src["chunk_id"]) + 1))
            for h, feats in zip(res.hits, featurize(q, res.hits), strict=True):
                rows.append(
                    Row(
                        doc_id=str(src["doc_id"]),
                        query=q,
                        label=int(h.chunk_id == src["chunk_id"]),
                        feats=feats,
                    )
                )
    return Harvest(
        rows=rows,
        n_queries=n_queries,
        skipped_lang=skipped_lang,
        skipped_missing=skipped_missing,
        baseline_rr=baseline,
    )


def fit(
    rows: Sequence[Row],
    *,
    train_docs: Sequence[str],
    held_out: Sequence[str],
    lang_match: bool,
    l2: float = 1.0,
    lr: float = 0.5,
    iters: int = 600,
) -> Model:
    if not rows:
        raise ValueError("학습 행이 없다")
    n = len(FEATURES)
    cols = [[r.feats[j] for r in rows] for j in range(n)]
    mean = [statistics.fmean(c) for c in cols]
    std = [statistics.pstdev(c) or 1.0 for c in cols]
    x = [[(r.feats[j] - mean[j]) / std[j] for j in range(n)] for r in rows]
    y = [float(r.label) for r in rows]

    n_pos = int(sum(y))
    n_neg = len(y) - n_pos
    pos_w = (n_neg / n_pos) if n_pos else 1.0
    weights = [pos_w if yi else 1.0 for yi in y]
    total = sum(weights)

    w = [0.0] * n
    bias = 0.0
    loss = 0.0
    for _ in range(iters):
        gw = [0.0] * n
        gb = 0.0
        loss = 0.0
        for xi, yi, wi in zip(x, y, weights, strict=True):
            z = bias + sum(w[j] * xi[j] for j in range(n))
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = (p - yi) * wi
            for j in range(n):
                gw[j] += err * xi[j]
            gb += err
            loss -= wi * (yi * math.log(max(p, 1e-12)) + (1 - yi) * math.log(max(1 - p, 1e-12)))
        loss = loss / total + 0.5 * l2 * sum(v * v for v in w) / total
        for j in range(n):
            w[j] -= lr * (gw[j] / total + l2 * w[j] / total)
        bias -= lr * gb / total

    return Model(
        weights=w,
        bias=bias,
        mean=mean,
        std=std,
        n_rows=len(rows),
        n_pos=n_pos,
        train_docs=list(train_docs),
        held_out=list(held_out),
        lang_match=lang_match,
        l2=l2,
        iters=iters,
        loss=loss,
    )


def rerank(model: Model, query: str, hits: Sequence[Hit]) -> list[Hit]:
    if not hits:
        return []
    scored = list(zip(hits, [model.score(f) for f in featurize(query, hits)], strict=True))
    order = sorted(range(len(scored)), key=lambda i: (-scored[i][1], i))
    return [scored[i][0] for i in order]


def mrr_of(rows: Sequence[Row], model: Model | None) -> float:
    groups: dict[str, list[Row]] = {}
    for r in rows:
        groups.setdefault(r.query, []).append(r)
    out = []
    for _, g in groups.items():
        if not any(r.label for r in g):
            continue
        if model is None:
            order = list(range(len(g)))
        else:
            order = sorted(range(len(g)), key=lambda i: (-model.score(g[i].feats), i))
        pos = next(i for i, idx in enumerate(order, 1) if g[idx].label)
        out.append(1.0 / pos)
    return statistics.fmean(out) if out else 0.0

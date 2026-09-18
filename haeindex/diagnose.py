from collections.abc import Sequence
from typing import Any

from opensearchpy import OpenSearch
from pydantic import BaseModel, ConfigDict, Field

from haeindex.evaluate import Question, covered_targets
from haeindex.index import INDEX, find_covering
from haeindex.search import CANDIDATE_K, TOP_K, Hit

CAUSES = ("성공", "순위", "dedupe", "후보", "청킹")
FIXES = {
    "성공": "정답이 top-k 안에 있다",
    "순위": "후보에는 있고 top-k 밖이다 — 리랭커·부스트가 고칠 수 있는 부류",
    "dedupe": "후보에 있었는데 근접중복으로 버려졌다 — jaccard 임계값 문제",
    "후보": "BM25 도 kNN 도 후보에 못 올렸다 — 리랭커로는 못 고친다. 색인·청킹·질의 쪽",
    "청킹": "정답 라벨을 덮는 청크가 인덱스에 아예 없다 — 라벨이나 청킹이 틀렸다",
}


class Located(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    page: int
    end_page: int
    label: str
    targets: list[str] = Field(default_factory=list)
    fused_rank: int | None = None
    legs: dict[str, int] = Field(default_factory=dict)
    deduped: bool = False


class Diagnosis(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    query: str
    doc_id: str
    bucket: str
    targets: list[str] = Field(default_factory=list)
    covering: list[Located] = Field(default_factory=list)
    top_k: int = TOP_K
    candidate_k: int = CANDIDATE_K

    @property
    def best(self) -> Located | None:
        ranked = [c for c in self.covering if c.fused_rank is not None]
        return min(ranked, key=lambda c: c.fused_rank or 0) if ranked else None

    @property
    def cause(self) -> str:
        if not self.covering:
            return "청킹"
        best = self.best
        if best is None:
            return "dedupe" if any(c.deduped for c in self.covering) else "후보"
        return "성공" if (best.fused_rank or 0) <= self.top_k else "순위"


def _label(source: dict[str, Any]) -> str:
    return str(source.get("path") or source.get("title") or "")


def classify(
    question: Question,
    covering: Sequence[dict[str, Any]],
    hits: Sequence[Hit],
    *,
    dropped_ids: Sequence[str] = (),
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
) -> Diagnosis:
    """정답을 덮는 청크가 검색의 어느 단계에서 사라졌는지 가른다."""
    ranks = {h.chunk_id: i + 1 for i, h in enumerate(hits)}
    legs = {h.chunk_id: {k: v.rank for k, v in h.legs.items()} for h in hits}
    dropped = set(dropped_ids)

    located = [
        Located(
            chunk_id=str(src["chunk_id"]),
            page=int(src.get("page", 0)),
            end_page=int(src.get("end_page", src.get("page", 0))),
            label=_label(src),
            targets=sorted(covered_targets(src, question)),
            fused_rank=ranks.get(str(src["chunk_id"])),
            legs=legs.get(str(src["chunk_id"]), {}),
            deduped=str(src["chunk_id"]) in dropped,
        )
        for src in covering
    ]
    return Diagnosis(
        id=question.id,
        query=question.query,
        doc_id=question.doc_id,
        bucket=question.bucket,
        targets=list(question.targets),
        covering=located,
        top_k=top_k,
        candidate_k=candidate_k,
    )


def diagnose(
    os_client: OpenSearch,
    question: Question,
    hits: Sequence[Hit],
    *,
    dropped_ids: Sequence[str] = (),
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    index: str = INDEX,
) -> Diagnosis:
    covering = find_covering(
        os_client,
        question.doc_id,
        sections=question.sections,
        pages=question.pages,
        name=index,
    )
    return classify(
        question,
        covering,
        hits,
        dropped_ids=dropped_ids,
        top_k=top_k,
        candidate_k=candidate_k,
    )


def tally(diags: Sequence[Diagnosis]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {c: [] for c in CAUSES}
    for d in diags:
        out[d.cause].append(d.id)
    return out

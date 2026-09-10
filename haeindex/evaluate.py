import math
import random
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

BUCKETS = ("A", "B", "C", "D", "E")


class Question(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    doc_id: str
    bucket: str
    query: str
    pages: list[int] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def expect_refusal(self) -> bool:
        return self.bucket == "E"

    @property
    def targets(self) -> list[str]:
        if self.sections:
            return list(self.sections)
        return [f"p{n}" for n in self.pages]

    @model_validator(mode="after")
    def _check(self) -> "Question":
        if self.bucket not in BUCKETS:
            raise ValueError(f"{self.id}: bucket 은 {BUCKETS} 중 하나 ({self.bucket!r})")
        if self.expect_refusal and (self.pages or self.sections):
            raise ValueError(f"{self.id}: 버킷 E 는 정답이 없어야 한다")
        if not self.expect_refusal and not (self.pages or self.sections):
            raise ValueError(
                f"{self.id}: 정답이 비었다. 비어 있으면 모든 시스템이 recall 0 을 받고 "
                "그걸 '검색이 나쁘다' 로 읽는다"
            )
        return self


class Goldset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "v1"
    questions: list[Question]

    @property
    def ranked(self) -> list[Question]:
        return [q for q in self.questions if not q.expect_refusal]

    @property
    def refusal(self) -> list[Question]:
        return [q for q in self.questions if q.expect_refusal]

    @classmethod
    def load(cls, path: Path) -> "Goldset":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        gold = cls.model_validate(data)
        dup = {q.id for q in gold.questions if sum(x.id == q.id for x in gold.questions) > 1}
        if dup:
            raise ValueError(f"질의 id 중복: {sorted(dup)}")
        return gold

    def verify(self, known_docs: Sequence[str]) -> None:
        bad = sorted({q.doc_id for q in self.questions} - set(known_docs))
        if bad:
            raise ValueError(
                f"골든셋이 색인에 없는 문서를 가리킨다: {bad}\n색인된 문서: {sorted(known_docs)}"
            )


def covered_targets(source: dict[str, Any], question: Question) -> set[str]:
    if question.sections:
        return set(source.get("section_ids", [])) & set(question.sections)
    start = int(source.get("page", 0))
    end = int(source.get("end_page", start))
    return {f"p{n}" for n in question.pages if start <= n <= end}


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(gains: Sequence[float], n_relevant: int, k: int) -> float:
    if n_relevant <= 0:
        return 0.0
    ideal = dcg([1.0] * min(n_relevant, k))
    return dcg(list(gains)[:k]) / ideal if ideal else 0.0


class QueryScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    bucket: str
    ndcg: float
    recall: float
    chunk_precision: float
    rr: float
    first_rank: int | None = None
    duplicate_hits: int = 0
    right_doc: bool = False


def score_query(
    question: Question,
    hits: Sequence[dict[str, Any]],
    k: int,
) -> QueryScore:
    targets = set(question.targets)
    top = list(hits)[:k]

    gains: list[float] = []
    covered: set[str] = set()
    dup = 0
    relevant_flags: list[float] = []
    for src in top:
        hit_targets = covered_targets(src, question)
        relevant_flags.append(1.0 if hit_targets else 0.0)
        new = hit_targets - covered
        if new:
            covered |= new
            gains.append(1.0)
        else:
            if hit_targets:
                dup += 1
            gains.append(0.0)

    first = next((i + 1 for i, g in enumerate(relevant_flags) if g > 0), None)
    return QueryScore(
        id=question.id,
        doc_id=question.doc_id,
        bucket=question.bucket,
        ndcg=ndcg_at_k(gains, len(targets), k),
        recall=len(covered) / len(targets) if targets else 0.0,
        chunk_precision=(sum(relevant_flags) / len(top)) if top else 0.0,
        rr=(1.0 / first) if first else 0.0,
        first_rank=first,
        duplicate_hits=dup,
        right_doc=bool(top) and str(top[0].get("doc_id", "")) == question.doc_id,
    )


class Summary(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    n: int
    ndcg: float
    recall: float
    chunk_precision: float
    mrr: float
    right_doc: float
    duplicate_hits: float
    zero_hit: list[str] = Field(default_factory=list)
    by_bucket: dict[str, dict[str, float]] = Field(default_factory=dict)
    by_doc: dict[str, dict[str, float]] = Field(default_factory=dict)


def _agg(scores: Sequence[QueryScore]) -> dict[str, float]:
    return {
        "n": float(len(scores)),
        "ndcg": statistics.fmean(s.ndcg for s in scores),
        "recall": statistics.fmean(s.recall for s in scores),
        "mrr": statistics.fmean(s.rr for s in scores),
    }


def summarize(label: str, scores: Sequence[QueryScore]) -> Summary:
    if not scores:
        return Summary(
            label=label,
            n=0,
            ndcg=0,
            recall=0,
            chunk_precision=0,
            mrr=0,
            right_doc=0,
            duplicate_hits=0,
        )
    buckets: dict[str, list[QueryScore]] = {}
    docs: dict[str, list[QueryScore]] = {}
    for s in scores:
        buckets.setdefault(s.bucket, []).append(s)
        docs.setdefault(s.doc_id, []).append(s)
    return Summary(
        label=label,
        n=len(scores),
        ndcg=statistics.fmean(s.ndcg for s in scores),
        recall=statistics.fmean(s.recall for s in scores),
        chunk_precision=statistics.fmean(s.chunk_precision for s in scores),
        mrr=statistics.fmean(s.rr for s in scores),
        right_doc=statistics.fmean(1.0 if s.right_doc else 0.0 for s in scores),
        duplicate_hits=statistics.fmean(s.duplicate_hits for s in scores),
        zero_hit=[s.id for s in scores if s.rr == 0.0],
        by_bucket={b: _agg(v) for b, v in sorted(buckets.items())},
        by_doc={d: _agg(v) for d, v in sorted(docs.items())},
    )


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str
    a: str
    b: str
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def paired_bootstrap(
    metric: str,
    a_label: str,
    b_label: str,
    a: Sequence[float],
    b: Sequence[float],
    *,
    iters: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Comparison:
    if len(a) != len(b):
        raise ValueError(f"짝이 맞지 않는다: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("질의가 0개다")

    diffs = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(iters))
    return Comparison(
        metric=metric,
        a=a_label,
        b=b_label,
        mean_a=statistics.fmean(a),
        mean_b=statistics.fmean(b),
        diff=statistics.fmean(diffs),
        ci_low=means[int(alpha / 2 * iters)],
        ci_high=means[min(iters - 1, int((1 - alpha / 2) * iters))],
        n=n,
    )

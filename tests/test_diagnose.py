from typing import Any

from haeindex.diagnose import classify, tally
from haeindex.evaluate import Question
from haeindex.search import Hit, LegHit


def q(**kw: Any) -> Question:
    base: dict[str, Any] = {
        "id": "t1",
        "doc_id": "d",
        "bucket": "B",
        "query": "질의",
        "pages": [9],
    }
    base.update(kw)
    return Question(**base)


def src(cid: str, page: int = 9, end: int | None = None) -> dict[str, Any]:
    return {
        "chunk_id": cid,
        "doc_id": "d",
        "page": page,
        "end_page": end if end is not None else page,
        "section_ids": [],
        "title": f"제{page}조",
        "body": "본문",
    }


def hit(cid: str, bm25: int | None = None) -> Hit:
    legs = {"bm25": LegHit(rank=bm25, score=1.0)} if bm25 else {}
    return Hit(chunk_id=cid, fused=1.0, source=src(cid), legs=legs)


def test_top_k_안에_있으면_성공이다() -> None:
    d = classify(q(), [src("c1")], [hit("c1", 1)], top_k=5)
    assert d.cause == "성공"
    assert d.best is not None and d.best.fused_rank == 1


def test_후보에_있고_top_k_밖이면_순위_문제다() -> None:
    hits = [hit(f"x{i}") for i in range(9)] + [hit("c1", 10)]
    d = classify(q(), [src("c1")], hits, top_k=5)
    assert d.cause == "순위"
    assert d.best is not None and d.best.fused_rank == 10


def test_후보에_아예_없으면_리랭커로는_못_고친다() -> None:
    d = classify(q(), [src("c1")], [hit("x0", 1)], top_k=5)
    assert d.cause == "후보"
    assert d.best is None


def test_dedupe_로_사라진_것을_후보_없음과_구별한다() -> None:
    d = classify(q(), [src("c1")], [hit("x0", 1)], dropped_ids=["c1"], top_k=5)
    assert d.cause == "dedupe"


def test_정답을_덮는_청크가_없으면_청킹_이나_라벨_문제다() -> None:
    d = classify(q(), [], [hit("x0", 1)], top_k=5)
    assert d.cause == "청킹"


def test_여러_청크가_정답을_덮으면_가장_높은_순위로_판정한다() -> None:
    hits = [hit("c2", 1)] + [hit(f"x{i}") for i in range(20)] + [hit("c1", 22)]
    d = classify(q(), [src("c1"), src("c2")], hits, top_k=5)
    assert d.cause == "성공"
    assert d.best is not None and d.best.chunk_id == "c2"


def test_원인별로_집계한다() -> None:
    a = classify(q(), [src("c1")], [hit("c1", 1)], top_k=5)
    b = classify(q(id="t2"), [], [hit("c1", 1)], top_k=5)
    counts = tally([a, b])
    assert counts["성공"] == ["t1"]
    assert counts["청킹"] == ["t2"]
    assert counts["후보"] == []

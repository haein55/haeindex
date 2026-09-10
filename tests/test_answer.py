from typing import Any

from haeindex.answer import (
    MIN_COS_TOP1,
    Refusal,
    assemble,
    est_tokens,
    keep_only_known,
    should_refuse,
)
from haeindex.search import Hit, LegHit, Result


def hit(cid: str, cos: float, body: str = "본문", **src: Any) -> Hit:
    return Hit(
        chunk_id=cid,
        fused=1.0,
        source={"chunk_id": cid, "body": body, "doc_id": "d", "page": 1, "path": "p", **src},
        legs={"knn": LegHit(rank=1, score=cos), "bm25": LegHit(rank=1, score=30.0)},
    )


def test_히트가_없으면_거부한다() -> None:
    assert should_refuse(Result(query="q")) is Refusal.NO_HITS


def test_벡터_검색이_빠지면_거부한다() -> None:
    res = Result(query="q", hits=[hit("a", 0.99)], degraded=["knn(임베더 없음)"])
    assert should_refuse(res) is Refusal.SEARCH_DEGRADED


def test_관련도가_낮으면_거부한다() -> None:
    assert should_refuse(Result(query="q", hits=[hit("a", 0.60)])) is Refusal.LOW_CONFIDENCE


def test_관련도가_높으면_통과한다() -> None:
    assert should_refuse(Result(query="q", hits=[hit("a", MIN_COS_TOP1)])) is None


def test_거부_이유마다_다른_문장을_쓴다() -> None:
    from haeindex.answer import MESSAGES

    assert len({MESSAGES[r] for r in Refusal}) == len(list(Refusal))
    assert "문서에 없다는 뜻이 아닙니다" in MESSAGES[Refusal.SEARCH_DEGRADED]


def test_없는_인용_번호를_지운다() -> None:
    text, used = keep_only_known("답이다 [cite:1] 또 [cite:9]", {1, 2})
    assert used == {1}
    assert "[cite:1]" in text
    assert "9" not in text


def test_부분만_무효면_유효한_것을_남긴다() -> None:
    text, used = keep_only_known("답이다 [cite:2,99]", {1, 2})
    assert used == {2}
    assert "[cite:2]" in text


def test_공백을_관대하게_받고_정식형으로_쓴다() -> None:
    text, used = keep_only_known("답 [cite: 1 , 2 ]", {1, 2})
    assert used == {1, 2}
    assert "[cite:1,2]" in text


def test_컨텍스트는_검색_순위_그대로다() -> None:
    res = Result(query="q", hits=[hit("a", 0.9, "가"), hit("b", 0.8, "나")])
    ctx = assemble(res)
    assert [b.n for b in ctx.blocks] == [1, 2]
    assert [b.text for b in ctx.blocks] == ["가", "나"]


def test_예산을_넘으면_자르지_않고_버린다() -> None:
    res = Result(query="q", hits=[hit("a", 0.9, "가" * 80), hit("b", 0.8, "나" * 80)])
    ctx = assemble(res, max_chars=100)
    assert len(ctx.blocks) == 1
    assert ctx.dropped == 1
    assert ctx.blocks[0].text == "가" * 80


def test_한국어_토큰_추정() -> None:
    assert est_tokens("가" * 160) == 100

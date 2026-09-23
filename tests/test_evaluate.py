from typing import Any

import pytest
from pydantic import ValidationError

from haeindex.evaluate import (
    Question,
    covered_targets,
    ndcg_at_k,
    score_query,
    summarize,
)


def q(bucket: str = "A", **kw: Any) -> Question:
    base: dict[str, Any] = {"id": "t1", "doc_id": "d", "bucket": bucket, "query": "질의"}
    base.update(kw)
    return Question(**base)


def chunk(page: int, end: int | None = None, sections: list[str] | None = None) -> dict[str, Any]:
    return {
        "doc_id": "d",
        "page": page,
        "end_page": end if end is not None else page,
        "section_ids": sections or [],
    }


def test_정답이_비면_거부한다() -> None:
    with pytest.raises(ValidationError, match="정답이 비었다"):
        q()


def test_버킷_E_는_정답이_있으면_모순이다() -> None:
    with pytest.raises(ValidationError, match="정답이 없어야"):
        q(bucket="E", pages=[1])


def test_절이_있으면_절로_없으면_페이지로_판정한다() -> None:
    by_page = q(pages=[3])
    by_sec = q(pages=[3], sections=["d#s0001"])
    assert covered_targets(chunk(3), by_page) == {"p3"}
    assert covered_targets(chunk(9), by_page) == set()
    assert covered_targets(chunk(9, sections=["d#s0001"]), by_sec) == {"d#s0001"}
    assert covered_targets(chunk(3), by_sec) == set()


def test_여러_쪽에_걸친_청크는_범위로_본다() -> None:
    assert covered_targets(chunk(3, 5), q(pages=[4])) == {"p4"}


@pytest.mark.parametrize("sections", [[], ["d#s0001"]])
def test_다른_문서의_같은_페이지나_절은_정답이_아니다(sections) -> None:
    question = q(pages=[3], sections=sections)
    wrong = {**chunk(3, sections=sections), "doc_id": "other"}
    score = score_query(question, [wrong, chunk(3, sections=sections)], 5)
    assert covered_targets(wrong, question) == set()
    assert score.first_rank == 2
    assert score.rr == 0.5


def test_정답을_1위에_놓으면_만점이다() -> None:
    s = score_query(q(pages=[3]), [chunk(3), chunk(9)], 5)
    assert s.ndcg == 1.0
    assert s.recall == 1.0
    assert s.rr == 1.0
    assert s.right_doc is True


def test_못_찾으면_0_이고_0히트로_잡힌다() -> None:
    s = score_query(q(pages=[3]), [chunk(9), chunk(10)], 5)
    assert (s.ndcg, s.recall, s.rr) == (0.0, 0.0, 0.0)
    assert summarize("x", [s]).zero_hit == ["t1"]


def test_신규성_이득이라_nDCG_가_1을_넘지_않는다() -> None:
    """★ 옛 프로젝트에서 nDCG 1.054 가 나온 버그를 고정한다.

    정답이 1개인데 그 정답을 덮는 청크가 3개 나오면, 맞을 때마다 이득을 주면
    DCG = 1 + 0.631 + 0.5 = 2.131 이고 이상 DCG 는 1.0 이다.
    """
    s = score_query(q(pages=[3]), [chunk(3), chunk(3), chunk(3)], 5)
    assert s.ndcg == 1.0
    assert s.duplicate_hits == 2


def test_chunk_precision_은_중복을_그대로_센다() -> None:
    s = score_query(q(pages=[3]), [chunk(3), chunk(3), chunk(9), chunk(9), chunk(9)], 5)
    assert s.ndcg == 1.0
    assert s.chunk_precision == pytest.approx(0.4)


def test_정답이_여럿이면_빨리_다_덮어야_만점이다() -> None:
    fast = score_query(q(pages=[3, 4]), [chunk(3), chunk(4), chunk(9)], 5)
    slow = score_query(q(pages=[3, 4]), [chunk(9), chunk(3), chunk(4)], 5)
    assert fast.ndcg == 1.0
    assert slow.ndcg < fast.ndcg
    assert slow.recall == fast.recall == 1.0


def test_ndcg_는_존재하는_정답_수로_정규화한다() -> None:
    assert ndcg_at_k([1.0], n_relevant=1, k=5) == 1.0
    assert ndcg_at_k([1.0], n_relevant=2, k=5) < 1.0


def test_버킷별과_문서별로_나눠_본다() -> None:
    scores = [
        score_query(q(bucket="A", pages=[3]), [chunk(3)], 5),
        score_query(q(bucket="B", pages=[3]), [chunk(9)], 5),
    ]
    s = summarize("x", scores)
    assert s.by_bucket["A"]["ndcg"] == 1.0
    assert s.by_bucket["B"]["ndcg"] == 0.0
    assert s.by_doc["d"]["n"] == 2.0


def test_짝지은_부트스트랩은_짝을_유지한다() -> None:
    from haeindex.evaluate import paired_bootstrap

    a = [0.9, 0.8, 0.7, 1.0, 0.6]
    b = [0.5, 0.4, 0.3, 0.6, 0.2]
    cmp = paired_bootstrap("nDCG", "a", "b", a, b, iters=2000)
    assert cmp.diff == pytest.approx(0.4)
    assert cmp.ci_low > 0.0
    assert cmp.significant


def test_차이가_없으면_구간이_0을_포함한다() -> None:
    from haeindex.evaluate import paired_bootstrap

    a = [0.5, 0.9, 0.1, 0.7, 0.3]
    b = [0.9, 0.5, 0.7, 0.1, 0.3]
    cmp = paired_bootstrap("nDCG", "a", "b", a, b, iters=2000)
    assert not cmp.significant


def test_seed_를_고정하면_같은_구간이_나온다() -> None:
    from haeindex.evaluate import paired_bootstrap

    args = ("nDCG", "a", "b", [0.9, 0.5, 0.7], [0.4, 0.3, 0.6])
    first = paired_bootstrap(*args, iters=1000)
    assert first.ci_low == paired_bootstrap(*args, iters=1000).ci_low


def test_짝이_안_맞으면_거부한다() -> None:
    from haeindex.evaluate import paired_bootstrap

    with pytest.raises(ValueError, match="짝이 맞지 않는다"):
        paired_bootstrap("nDCG", "a", "b", [0.5], [0.5, 0.5])

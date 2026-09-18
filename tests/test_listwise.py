from collections.abc import Sequence
from typing import Any

from haeindex.listwise import parse_order, passage, rerank, windows
from haeindex.search import Hit


class FakeLLM:
    """chat 만 흉내 낸다. 창마다 정해진 응답을 돌려준다."""

    def __init__(self, replies: Sequence[str], num_ctx: int = 8192) -> None:
        self.replies = list(replies)
        self.num_ctx = num_ctx
        self.calls = 0

    def chat(self, messages: Any, **kw: Any) -> Any:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return type("R", (), {"content": reply})()


def hit(cid: str) -> Hit:
    return Hit(
        chunk_id=cid,
        fused=1.0,
        source={"chunk_id": cid, "doc_id": "d", "page": 1, "title": cid, "body": "본문 " + cid},
    )


def ids(hits: Sequence[Hit]) -> list[str]:
    return [h.chunk_id for h in hits]


def test_부분_순열이면_나머지는_융합_순서를_지킨다() -> None:
    assert parse_order("[3, 1]", 4) == [2, 0, 1, 3]


def test_범위_밖과_중복을_버린다() -> None:
    assert parse_order("[2, 2, 99, 0, 1]", 3) == [1, 0, 2]


def test_배열이_없으면_실패로_본다() -> None:
    assert parse_order("첫 번째가 맞습니다", 3) is None
    assert parse_order("[]", 3) is None


def test_창을_뒤에서_앞으로_민다() -> None:
    assert windows(50, 20, 10) == [(30, 50), (20, 40), (10, 30), (0, 20)]
    assert windows(12, 20, 10) == [(0, 12)]


def test_재정렬이_적용된다() -> None:
    hits = [hit(c) for c in "abcde"]
    r = rerank(FakeLLM(["[5, 1]"]), "질의", hits, depth=5, window=5)
    assert ids(r.hits) == ["e", "a", "b", "c", "d"]
    assert r.calls == 1
    assert r.parse_fails == 0
    assert r.moved == 5


def test_파싱_실패하면_원래_순서를_지킨다() -> None:
    hits = [hit(c) for c in "abcde"]
    r = rerank(FakeLLM(["설명만 한다"]), "질의", hits, depth=5, window=5)
    assert ids(r.hits) == list("abcde")
    assert r.parse_fails == 1
    assert r.moved == 0


def test_depth_밖은_건드리지_않는다() -> None:
    hits = [hit(c) for c in "abcde"]
    r = rerank(FakeLLM(["[2, 1]"]), "질의", hits, depth=2, window=2)
    assert ids(r.hits) == ["b", "a", "c", "d", "e"]


def test_후보가_하나면_LLM_을_안_부른다() -> None:
    llm = FakeLLM(["[1]"])
    r = rerank(llm, "질의", [hit("a")], depth=20)
    assert llm.calls == 0
    assert r.calls == 0


def test_조각에_문서와_쪽과_경로가_들어간다() -> None:
    p = passage(hit("a"), 3)
    assert p.startswith("[3] (d · p.1 · a)")
    assert "본문 a" in p

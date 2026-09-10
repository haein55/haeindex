from typing import Any

from haeindex.search import (
    Hit,
    bm25_body,
    char_ngrams,
    dedupe,
    doc_filter,
    knn_body,
    rrf_fuse,
)


def row(chunk_id: str, score: float, body: str = "") -> dict[str, Any]:
    return {"_score": score, "_source": {"chunk_id": chunk_id, "body": body}}


def test_문서_필터는_비면_없다() -> None:
    assert doc_filter([]) == []
    assert doc_filter(["a", "b"]) == [{"terms": {"doc_id": ["a", "b"]}}]


def test_필터는_bool_filter_에_들어간다() -> None:
    body = bm25_body("질의", doc_filter(["a"]), 50)
    assert body["query"]["bool"]["filter"] == [{"terms": {"doc_id": ["a"]}}]


def test_knn_은_pre_filter_다() -> None:
    body = knn_body([0.1] * 4, doc_filter(["a"]), 50)
    knn = body["query"]["knn"]["embedding"]
    assert knn["k"] == 50
    assert knn["filter"] == {"bool": {"filter": [{"terms": {"doc_id": ["a"]}}]}}
    assert knn_body([0.1] * 4, [], 50)["query"]["knn"]["embedding"].get("filter") is None


def test_벡터는_source_로_안_돌려받는다() -> None:
    assert bm25_body("q", [], 5)["_source"]["excludes"] == ["embedding"]
    assert knn_body([0.1], [], 5)["_source"]["excludes"] == ["embedding"]


def test_rrf_는_점수가_아니라_순위를_쓴다() -> None:
    results = {
        "bm25": [row("a", 100.0), row("b", 1.0)],
        "knn": [row("b", 0.9), row("a", 0.8)],
    }
    hits = rrf_fuse(results)
    assert {h.chunk_id for h in hits} == {"a", "b"}
    assert abs(hits[0].fused - hits[1].fused) < 1e-9


def test_두_leg_이_합의한_문서가_올라간다() -> None:
    results = {
        "bm25": [row("x", 50.0), row("both", 40.0)],
        "knn": [row("y", 0.9), row("both", 0.8)],
    }
    hits = rrf_fuse(results)
    assert hits[0].chunk_id == "both"


def test_한_leg_만_있어도_돈다() -> None:
    hits = rrf_fuse({"bm25": [row("a", 10.0)]})
    assert [h.chunk_id for h in hits] == ["a"]
    assert hits[0].legs["bm25"].rank == 1


def test_근접_중복을_버린다() -> None:
    same = "메모리 카드를 포맷하면 데이터가 삭제됩니다" * 3
    hits = [
        Hit(chunk_id="a", fused=1.0, source={"body": same}),
        Hit(chunk_id="b", fused=0.9, source={"body": same + " 조금 더"}),
        Hit(chunk_id="c", fused=0.8, source={"body": "배터리 충전 시간은 얼마인가" * 3}),
    ]
    kept, dropped = dedupe(hits)
    assert [h.chunk_id for h in kept] == ["a", "c"]
    assert dropped == 1


def test_본문이_비면_중복으로_안_본다() -> None:
    hits = [
        Hit(chunk_id="a", fused=1.0, source={"body": ""}),
        Hit(chunk_id="b", fused=0.9, source={"body": ""}),
    ]
    kept, dropped = dedupe(hits)
    assert len(kept) == 2
    assert dropped == 0


def test_ngram_은_공백을_무시한다() -> None:
    assert char_ngrams("가 나 다") == char_ngrams("가나다")


def test_가상_질문은_검색_필드에서_빠져_있다() -> None:
    body = bm25_body("질의", [], 50)
    fields = body["query"]["bool"]["should"][0]["multi_match"]["fields"]
    assert not any(f.startswith("queries^") for f in fields)

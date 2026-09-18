import math

import pytest

from haeindex.rerank import (
    FEATURES,
    Model,
    Row,
    featurize,
    fit,
    gram_overlap,
    mrr_of,
    rerank,
    script_of,
)
from haeindex.search import Hit, LegHit


def hit(cid, doc="d1", *, bm25=None, knn=None, title="", body="", depth=0, tokens=100):
    legs = {}
    if bm25:
        legs["bm25"] = LegHit(rank=bm25[0], score=bm25[1])
    if knn:
        legs["knn"] = LegHit(rank=knn[0], score=knn[1])
    return Hit(
        chunk_id=cid,
        fused=0.0,
        legs=legs,
        source={
            "doc_id": doc,
            "title": title,
            "body": body,
            "depth": depth,
            "token_len": tokens,
        },
    )


def test_스크립트_판정():
    assert script_of("연차 휴가는 며칠인가요") == "han"
    assert script_of("What is Graph RAG") == "latin"
    assert script_of("RAG是什么技术？") == "cjk"


def test_중국어_질의가_영문_청크와_불일치로_잡힌다():
    chunk = "Retrieval-Augmented Generation represents a significant advancement"
    assert script_of(chunk) == "latin"
    assert script_of("RAG是如何工作的？") != script_of(chunk)


def test_그램_겹침():
    assert gram_overlap("연차휴가", "연차휴가 일수") == 1.0
    assert gram_overlap("연차휴가", "해고 예고") == 0.0
    assert gram_overlap("", "무엇이든") == 0.0


def test_피처_이름과_개수가_맞는다():
    rows = featurize("질의", [hit("a", bm25=(1, 30.0), knn=(2, 0.8))])
    assert len(FEATURES) == len(rows[0]) == 10


def test_한쪽_leg_만_잡은_후보는_0_이_들어간다():
    hits = [hit("a", bm25=(1, 30.0)), hit("b", knn=(1, 0.9))]
    a, b = featurize("질의", hits)
    assert a[FEATURES.index("knn_rr")] == 0.0
    assert a[FEATURES.index("knn_cos")] == 0.0
    assert b[FEATURES.index("bm25_rr")] == 0.0
    assert a[FEATURES.index("both_legs")] == b[FEATURES.index("both_legs")] == 0.0


def test_doc_share_는_질의_안에서만_의미가_있다():
    hits = [hit("a", "d1"), hit("b", "d1"), hit("c", "d1"), hit("d", "d2")]
    feats = featurize("질의", hits)
    j = FEATURES.index("doc_share")
    assert feats[0][j] == pytest.approx(0.75)
    assert feats[3][j] == pytest.approx(0.25)


def test_bm25_z_는_척도가_아니라_벌어짐을_본다():
    small = [hit("a", bm25=(1, 3.0)), hit("b", bm25=(2, 1.0))]
    big = [hit("a", bm25=(1, 300.0)), hit("b", bm25=(2, 100.0))]
    j = FEATURES.index("bm25_z")
    assert featurize("q", small)[0][j] == pytest.approx(featurize("q", big)[0][j])


def rows_for(n_query=12):
    rows = []
    for i in range(n_query):
        rows.append(Row(doc_id="d1", query=f"q{i}", label=1, feats=[1.0] * 10))
        for k in range(4):
            rows.append(Row(doc_id="d1", query=f"q{i}", label=0, feats=[0.1 * (k + 1)] * 10))
    return rows


def test_학습이_양성을_구분한다():
    m = fit(rows_for(), train_docs=["d1"], held_out=["d2"], lang_match=True, iters=200)
    assert m.score([1.0] * 10) > m.score([0.1] * 10)
    assert m.n_pos == 12
    assert math.isfinite(m.loss)


def test_계수는_절댓값_순으로_읽힌다():
    m = fit(rows_for(), train_docs=["d1"], held_out=["d2"], lang_match=True, iters=50)
    coefs = [abs(w) for _, w in m.coefficients()]
    assert coefs == sorted(coefs, reverse=True)


def test_행이_없으면_학습을_거부한다():
    with pytest.raises(ValueError):
        fit([], train_docs=[], held_out=[], lang_match=True)


def test_표준화가_상수_피처에서_0으로_나누지_않는다():
    rows = [
        Row(doc_id="d", query="q", label=1, feats=[5.0] * 10),
        Row(doc_id="d", query="q", label=0, feats=[5.0] * 10),
    ]
    m = fit(rows, train_docs=["d"], held_out=[], lang_match=True, iters=10)
    assert all(s != 0.0 for s in m.std)
    assert math.isfinite(m.score([5.0] * 10))


def test_저장하고_불러오면_같은_점수가_나온다(tmp_path):
    m = fit(rows_for(), train_docs=["d1"], held_out=["d2"], lang_match=True, iters=50)
    p = tmp_path / "m.json"
    m.save(p)
    again = Model.load(p)
    assert again.score([1.0] * 10) == pytest.approx(m.score([1.0] * 10))


def test_리랭커는_후보_집합을_바꾸지_않고_순서만_바꾼다():
    m = fit(rows_for(), train_docs=["d1"], held_out=["d2"], lang_match=True, iters=200)
    hits = [
        hit("a", bm25=(1, 10.0), title="무관"),
        hit("b", bm25=(9, 1.0), knn=(1, 0.95), title="연차 휴가", body="연차 휴가는 15일"),
    ]
    out = rerank(m, "연차 휴가", hits)
    assert {h.chunk_id for h in out} == {"a", "b"}
    assert len(out) == 2


def test_빈_후보는_빈_결과다():
    m = fit(rows_for(), train_docs=["d1"], held_out=["d2"], lang_match=True, iters=10)
    assert rerank(m, "질의", []) == []


def test_mrr_은_정답_없는_질의를_세지_않는다():
    rows = [Row(doc_id="d", query="q", label=0, feats=[0.0] * 10)]
    assert mrr_of(rows, None) == 0.0


def test_mrr_기준선은_주어진_순서를_그대로_쓴다():
    rows = [
        Row(doc_id="d", query="q", label=0, feats=[0.0] * 10),
        Row(doc_id="d", query="q", label=1, feats=[1.0] * 10),
    ]
    assert mrr_of(rows, None) == pytest.approx(0.5)

import pytest

from haeindex.agent import (
    Step,
    Trace,
    match_doc,
    merge_hits,
    parse_action,
    render_docs,
    render_obs,
    summarize_traces,
)
from haeindex.answer import Answer, Refusal
from haeindex.search import Hit, LegHit

DOCS = ["취업규칙-260417", "3-홍천양수발전소-1-2호기-토건공사-입찰안내서", "hxr-mc88-manual-kr1"]


def hit(cid, fused, doc="d", page=1, body="본문", title="제1조"):
    return Hit(
        chunk_id=cid,
        fused=fused,
        legs={"knn": LegHit(rank=1, score=0.8)},
        source={"doc_id": doc, "page": page, "body": body, "title": title, "path": title},
    )


def test_한_줄_JSON_을_읽는다():
    assert parse_action('{"action":"search","query":"연차"}') == {
        "action": "search",
        "query": "연차",
    }


def test_설명이_붙어도_JSON_만_건진다():
    raw = '다음으로 검색하겠습니다.\n{"action":"search","query":"정년"}\n이유는...'
    assert parse_action(raw) == {"action": "search", "query": "정년"}


def test_코드블록에_싸여도_읽는다():
    raw = '```json\n{"action":"answer"}\n```'
    assert parse_action(raw) == {"action": "answer"}


def test_action_이_없는_JSON_은_거부한다():
    assert parse_action('{"query":"연차"}') is None


def test_JSON_이_아니면_None_이다():
    assert parse_action("연차 휴가를 검색해야 합니다") is None


def test_문서_이름을_부분_문자열로_찾는다():
    assert match_doc("취업규칙", DOCS) == ["취업규칙-260417"]
    assert match_doc("입찰안내서", DOCS) == [DOCS[1]]


def test_문서_이름이_토막나도_찾는다():
    assert match_doc("HXR MC88", DOCS) == ["hxr-mc88-manual-kr1"]


def test_없는_문서_이름은_빈_목록이다():
    assert match_doc("존재하지않는문서", DOCS) == []
    assert match_doc("", DOCS) == []


def test_문서_지정이_없으면_전체_검색이_된다():
    assert match_doc("전체", DOCS) == []


def test_병합은_같은_조각을_두_번_넣지_않는다():
    merged = merge_hits([hit("a", 0.1)], [hit("a", 0.3), hit("b", 0.2)])
    assert [h.chunk_id for h in merged] == ["a", "b"]
    assert merged[0].fused == 0.3


def test_병합은_점수_높은_쪽을_남긴다():
    merged = merge_hits([hit("a", 0.5)], [hit("a", 0.2)])
    assert merged[0].fused == 0.5


def test_병합_순서는_점수_내림차순이다():
    merged = merge_hits([hit("a", 0.1)], [hit("b", 0.9)])
    assert [h.chunk_id for h in merged] == ["b", "a"]


def test_관측은_코사인과_문서를_같이_보여준다():
    step = Step(n=1, action="search", query="연차", n_hits=1)
    obs = render_obs(step, [hit("a", 0.1, doc="취업규칙-260417", page=9)])
    assert "취업규칙" in obs and "p9" in obs and "0.80" in obs


def test_문서_목록은_조각_수를_같이_준다():
    assert "취업규칙-260417 (101조각)" in render_docs({"취업규칙-260417": 101})


def test_trace_는_top_k_까지만_답변에_쓴다():
    tr = Trace(question="q", hits=[hit(str(i), 1.0 - i / 10) for i in range(8)])
    assert len(tr.result(5).hits) == 5


def test_비용_요약이_질문당_평균을_낸다():
    traces = [
        Trace(question="a", calls=4, searches=2, seconds=10.0, parse_fails=1),
        Trace(question="b", calls=2, searches=1, seconds=6.0),
    ]
    got = summarize_traces(traces)
    assert got["LLM호출/질문"] == pytest.approx(3.0)
    assert got["검색/질문"] == pytest.approx(1.5)
    assert got["파싱실패"] == 1
    assert got["재검색"] == 1


def test_거부한_추적도_요약에_들어간다():
    tr = Trace(question="q", answer=Answer(refusal=Refusal.LOW_CONFIDENCE), calls=2)
    assert summarize_traces([tr])["질문"] == 1


def test_doc_later_규칙이_프롬프트에_붙는다():
    from haeindex.agent import DOC_LATER, SYSTEM

    assert "첫 검색에는" in DOC_LATER
    assert "첫 검색에는" not in SYSTEM


def test_문서지정_무시가_기록된다():
    step = Step(n=1, action="search", query="q", doc="취업규칙", doc_ignored=True)
    tr = Trace(question="q", steps=[step])
    assert summarize_traces([tr])["문서지정무시"] == 1

from typing import Any

from haeindex.answer import (
    MIN_COS_TOP1,
    Refusal,
    assemble,
    claims_have_citations,
    est_tokens,
    keep_only_known,
    language_matches,
    scope_single_citation,
    should_refuse,
)
from haeindex.answer import answer as run_answer
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


def test_질문과_답변_언어가_같은지_검사한다() -> None:
    assert language_matches("메모리 카드를 포맷합니다 [cite:1]", "ko")
    assert not language_matches("Format the memory card [cite:1]", "ko")
    assert language_matches("Format the memory card [cite:1]", "en")
    assert not language_matches("햇빛이나 강한 光线에 노출하지 마세요 [cite:1]", "ko")
    assert not language_matches("Use strong 光线 carefully [cite:1]", "en")


def test_모든_문장에_인용이_있어야_한다() -> None:
    assert claims_have_citations("첫 문장입니다. [cite:1] 둘째 문장입니다. [cite:2]")
    assert not claims_have_citations("첫 문장입니다. [cite:1] 둘째 문장에는 근거가 없습니다.")


def test_근거가_하나면_답변_전체_인용으로_정리한다() -> None:
    text = "[1][cite:1] 첫 문장입니다. 둘째 문장입니다."
    assert scope_single_citation(text, {1}) == "첫 문장입니다. 둘째 문장입니다. [cite:1]"


class FakeLLM:
    num_ctx = 8192

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls = 0

    def chat(self, messages, **kwargs):
        reply = self.replies[self.calls]
        self.calls += 1
        return type("Reply", (), {"content": reply})()


def test_한국어_질문에_영어로_답하면_한번_다시_생성한다() -> None:
    llm = FakeLLM(["The answer is ten days. [cite:1]", "기간은 10일입니다. [cite:1]"])
    result = Result(query="기간은?", hits=[hit("a", 0.9, "기간은 10일입니다")])
    got = run_answer(llm, result, "기간은 며칠인가요?")
    assert got.refusal is None
    assert got.language == "ko"
    assert got.language_retries == 1
    assert llm.calls == 2


def test_재생성해도_언어가_다르면_거부한다() -> None:
    llm = FakeLLM(
        [
            "It is ten days. [cite:1]",
            "Still ten days. [cite:1]",
            "Still not Korean. [cite:1]",
        ]
    )
    result = Result(query="기간은?", hits=[hit("a", 0.9, "기간은 10일입니다")])
    got = run_answer(llm, result, "기간은 며칠인가요?")
    assert got.refusal is Refusal.LANGUAGE_MISMATCH


def test_한자가_섞인_재생성은_다시_깨끗하게_생성한다() -> None:
    llm = FakeLLM(
        [
            "Use no direct sunlight. [cite:1]",
            "햇빛과 강한 光线을 피하세요. [cite:1]",
            "햇빛과 강한 빛을 피하세요. [cite:1]",
        ]
    )
    result = Result(query="주의사항", hits=[hit("a", 0.9, "햇빛과 강한 빛을 피하세요")])
    got = run_answer(llm, result, "햇빛에 대한 주의사항은?")
    assert got.refusal is None
    assert got.language_retries == 2
    assert "光" not in got.text


def test_근거가_하나면_인용을_답변_전체에_적용한다() -> None:
    llm = FakeLLM(["기간은 10일입니다. [cite:1] 추가 조건은 없습니다."])
    result = Result(query="기간은?", hits=[hit("a", 0.9, "기간은 10일입니다")])
    got = run_answer(llm, result, "기간은 며칠인가요?")
    assert got.refusal is None
    assert got.citation_retries == 0
    assert got.citations_complete
    assert got.text.endswith("[cite:1]")


def test_여러_인용의_교정이_실패해도_원래_답은_폐기하지_않는다() -> None:
    original = "기간은 10일입니다. [cite:1] 추가 조건은 없습니다. [cite:2] 끝 문장입니다."
    llm = FakeLLM([original, "인용을 고치지 못했습니다"])
    result = Result(
        query="기간은?",
        hits=[hit("a", 0.9, "기간은 10일입니다"), hit("b", 0.89, "추가 조건")],
    )
    got = run_answer(llm, result, "기간과 조건은 무엇인가요?")
    assert got.refusal is None
    assert got.text == original
    assert got.cited == [1, 2]
    assert not got.citations_complete

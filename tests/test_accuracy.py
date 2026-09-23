import json
from types import SimpleNamespace

import pytest

from haeindex import pipeline, reasoning
from haeindex.answer import Refusal
from haeindex.augmentation import ChunkContext, ContextBatch, Quality, augment_batch
from haeindex.bedrock import ChatResult, Truncated
from haeindex.llm_tasks import BudgetExceeded, TaskFailure, TaskRunner
from haeindex.reasoning import (
    AnswerPatch,
    AnswerReview,
    ClaimReview,
    Draft,
    EvidenceCheck,
    EvidenceGap,
    QuestionPlan,
    Quote,
    SentenceEdit,
    Support,
    check_evidence,
    review_answer,
)
from haeindex.routing import Route
from haeindex.search import Hit, Result


def hit(cid="a#1", body="신청 기한은 2024년 6월 1일까지입니다.", doc="a"):
    return Hit(chunk_id=cid, fused=1, source={"body": body, "doc_id": doc, "page": 1})


class Runner:
    def __init__(self, *results):
        self.results = iter(results)

    def run(self, *args, **kwargs):
        return next(self.results)


class LLM:
    chat_model = "test-model"
    num_ctx = 16384

    def __init__(self, text='{"text":"ok"}', digest="v1"):
        self.text = text
        self.digest = digest
        self.calls = 0

    def model_digest(self):
        return self.digest

    def chat(self, *args, **kwargs):
        self.calls += 1
        return ChatResult(content=self.text)


def test_cache_is_bound_to_model_digest_and_input(tmp_path):
    llm = LLM()
    runner = TaskRunner(cache=tmp_path)
    for _ in range(2):
        assert runner.run(llm, "draft", "instruction", {"q": "x"}, Draft).text == "ok"
    assert llm.calls == 1
    llm.digest = "v2"
    runner.run(llm, "draft", "instruction", {"q": "x"}, Draft)
    runner.run(llm, "draft", "instruction", {"q": "y"}, Draft)
    assert llm.calls == 3


def test_invalid_json_uses_bounded_retry_without_caching(tmp_path):
    llm = LLM('{"unexpected": 1}')
    with pytest.raises(TaskFailure):
        TaskRunner(cache=tmp_path).run(llm, "draft", "i", {}, Draft)
    assert llm.calls == 2
    assert not list(tmp_path.glob("*.json"))


def test_budget_and_context_guard_prevent_extra_calls():
    llm = LLM()
    with pytest.raises(BudgetExceeded):
        TaskRunner(max_calls=0, cache=None).run(llm, "draft", "i", {}, Draft)
    with pytest.raises(TaskFailure, match="컨텍스트"):
        TaskRunner(cache=None).run(llm, "draft", "i", {"body": "x" * 50000}, Draft)
    assert llm.calls == 0


class TruncatingLLM(LLM):
    def __init__(self, text='{"text":"ok"}', *, threshold=3600):
        super().__init__(text)
        self.threshold = threshold
        self.limits = []

    def chat(self, messages, **kwargs):
        self.limits.append(kwargs["num_predict"])
        self.calls += 1
        if kwargs["num_predict"] < self.threshold:
            raise Truncated("max_output_tokens")
        return ChatResult(content=self.text)


def test_truncated_output_expands_once_and_caches_only_valid_result(tmp_path):
    llm = TruncatingLLM()
    runner = TaskRunner(cache=tmp_path)
    for _ in range(2):
        result = runner.run(llm, "draft", "i", {}, Draft, num_predict=1800, max_predict=3600)
        assert result.text == "ok"
    assert llm.limits == [1800, 3600]
    assert runner.calls == 2
    assert "Truncated" in runner.events[0].error
    assert runner.events[-1].cache_hit
    assert len(list(tmp_path.glob("*.json"))) == 1
    # A result produced with a larger retry budget cannot bypass a lower cap via cache.
    with pytest.raises(TaskFailure, match="Truncated"):
        runner.run(llm, "draft", "i", {}, Draft, num_predict=1800, max_predict=1800)
    assert llm.limits == [1800, 3600, 1800]


def test_truncation_expansion_respects_context_room_without_repeating_cap():
    llm = TruncatingLLM(threshold=10000)
    llm.num_ctx = 4000
    runner = TaskRunner(cache=None)
    with pytest.raises(TaskFailure, match="Truncated"):
        runner.run(llm, "draft", "i", {}, Draft, num_predict=1800, max_predict=5600, retries=4)
    assert len(llm.limits) == 2
    assert 1800 < llm.limits[1] < llm.num_ctx * 0.9


@pytest.mark.parametrize("max_calls,retries,limits", [(1, 1, [1800]), (18, 0, [1800])])
def test_truncation_cannot_exceed_call_or_retry_budget(max_calls, retries, limits, tmp_path):
    llm = TruncatingLLM()
    runner = TaskRunner(max_calls=max_calls, cache=tmp_path)
    with pytest.raises(TaskFailure):
        runner.run(
            llm, "draft", "i", {}, Draft, num_predict=1800, max_predict=3600, retries=retries
        )
    assert llm.limits == limits
    assert not list(tmp_path.glob("*.json"))


def test_invented_quote_does_not_pass_sufficiency():
    check = EvidenceCheck(
        supports=[
            Support(
                requirement="기한",
                sources=[Quote(chunk_id="a#1", quote="신청 기한은 2025년 6월 1일까지입니다.")],
            )
        ]
    )
    result = check_evidence(
        Runner(check), None, "기한?", QuestionPlan(requirements=["기한"]), [hit()]
    )
    assert not result.complete(["기한"])
    assert result.missing == ["기한"]


def test_real_quote_does_not_cover_a_different_requirement():
    check = EvidenceCheck(
        supports=[
            Support(requirement="기한", sources=[Quote(chunk_id="a#1", quote=hit().source["body"])])
        ]
    )
    result = check_evidence(
        Runner(check), None, "기한과 금액?", QuestionPlan(requirements=["기한", "금액"]), [hit()]
    )
    assert not result.complete(["기한", "금액"])
    assert "금액" in result.missing


def test_short_complete_source_lines_are_valid_but_short_substrings_are_not():
    body = "포상의 종류는 다음 각 호와 같다.\n1. 표창\n2. 포상금\n3. 특별 승급 또는 승진"
    assert reasoning.quote_exists("1. 표창", body)
    assert reasoning.quote_exists("2. 포상금", body)
    assert not reasoning.quote_exists("표창", body)
    assert not reasoning.quote_exists("1. 상품", body)
    assert not reasoning.quote_exists("", body)
    assert not reasoning.quote_exists("①", "제1조\n①\n업무상 부상은 휴직 사유이다.")


def test_short_list_quotes_pass_evidence_and_sentence_review():
    h = hit(body="포상의 종류는 다음 각 호와 같다.\n1. 표창\n2. 포상금")
    quotes = [Quote(chunk_id=h.chunk_id, quote=line) for line in ["1. 표창", "2. 포상금"]]
    plan = QuestionPlan(requirements=["포상의 종류"])
    check = check_evidence(
        Runner(EvidenceCheck(supports=[Support(requirement="포상의 종류", sources=quotes)])),
        None,
        "포상의 종류?",
        plan,
        [h],
    )
    assert check.complete(plan.requirements)
    review = review_answer(
        Runner(
            AnswerReview(
                claims=[
                    ClaimReview(sentence_id=i, supported=True, sources=[quote])
                    for i, quote in enumerate(quotes, 1)
                ]
            )
        ),
        None,
        "포상의 종류?",
        plan,
        "1. 표창 [cite:1]\n2. 포상금 [cite:1]",
        [h],
        {1: h.chunk_id},
    )
    assert review.accepted()


def test_explaining_an_applicability_boundary_can_be_supported():
    h = hit(body="병가는 업무 외 부상 또는 질병에 적용한다.\n업무상 부상은 휴직 사유에 해당한다.")
    requirement = "병가를 받을 수 있어?"
    result = check_evidence(
        Runner(
            EvidenceCheck(
                supports=[
                    Support(
                        requirement=requirement,
                        sources=[Quote(chunk_id=h.chunk_id, quote=h.source["body"])],
                    )
                ]
            )
        ),
        None,
        "일하다가 다치면 병가를 받을 수 있어?",
        QuestionPlan(requirements=[requirement]),
        [h],
    )
    assert result.complete([requirement])
    assert not result.visual_chunk_ids()


def test_visual_gaps_require_missing_requirements_and_shown_chunk_ids():
    h = hit()
    check = EvidenceCheck(
        missing=["금액", "기간"],
        gaps=[
            EvidenceGap(requirement="금액", reason="table_layout", chunk_ids=[h.chunk_id, "fake"]),
            EvidenceGap(requirement="기간", reason="diagram", chunk_ids=["fake"]),
            EvidenceGap(requirement="없는 요구", reason="unreadable_text", chunk_ids=[h.chunk_id]),
        ],
    )
    result = check_evidence(
        Runner(check),
        None,
        "금액과 기간?",
        QuestionPlan(requirements=["금액", "기간"]),
        [h],
    )
    assert result.visual_chunk_ids() == [h.chunk_id]
    assert [(gap.requirement, gap.reason) for gap in result.gaps] == [
        ("금액", "table_layout"),
        ("기간", "missing_context"),
    ]


def test_visual_gap_cannot_reference_a_chunk_excluded_from_model_input():
    h = hit(body="x" * 11001)
    check = EvidenceCheck(
        missing=["기한"],
        gaps=[
            EvidenceGap(requirement="기한", reason="unreadable_text", chunk_ids=[h.chunk_id]),
        ],
    )
    result = check_evidence(
        Runner(check),
        None,
        "기한?",
        QuestionPlan(requirements=["기한"]),
        [h],
    )
    assert not result.visual_chunk_ids()


def test_wrong_citation_cannot_use_another_chunks_quote():
    sentence = "신청 기한은 2024년 6월 1일입니다. [cite:2]"
    review = AnswerReview(
        claims=[
            ClaimReview(
                sentence_id=1,
                supported=True,
                sources=[Quote(chunk_id="a#1", quote=hit().source["body"])],
            )
        ],
        repair_instructions="수정이 필요하지 않습니다.",
    )
    result = review_answer(
        Runner(review),
        None,
        "기한?",
        QuestionPlan(),
        sentence,
        [hit(), hit("b#1", doc="b")],
        {1: "a#1", 2: "b#1"},
    )
    assert not result.accepted()
    assert "출처와 다릅니다" in result.claims[0].issue
    assert "문장 1" in result.repair_instructions
    assert "수정이 필요하지 않습니다" not in result.repair_instructions


def test_paraphrased_review_quote_is_rejected_with_actionable_feedback():
    h = hit(body="AC 어댑터를 콘센트(소켓)에 연결합니다.")
    review = AnswerReview(
        claims=[
            ClaimReview(
                sentence_id=1,
                supported=True,
                sources=[Quote(chunk_id="a#1", quote="AC 어댑터를 콘센트에 연결합니다.")],
            )
        ],
        repair_instructions="수정이 필요하지 않습니다.",
    )
    result = review_answer(
        Runner(review),
        None,
        "연결 방법?",
        QuestionPlan(),
        "AC 어댑터를 콘센트에 연결합니다. [cite:1]",
        [h],
        {1: "a#1"},
    )
    assert not result.accepted()
    assert "원문에 그대로 존재하지 않습니다" in result.claims[0].issue
    assert "수정이 필요하지 않습니다" not in result.repair_instructions


def test_unreviewed_sentence_blocks_answer():
    sentence = "신청 기한은 2024년 6월 1일입니다. [cite:1]"
    review = AnswerReview(
        claims=[
            ClaimReview(
                sentence_id=1,
                supported=True,
                sources=[Quote(chunk_id="a#1", quote=hit().source["body"])],
            )
        ]
    )
    result = review_answer(
        Runner(review),
        None,
        "기한?",
        QuestionPlan(),
        sentence + "\n수수료는 무료입니다.",
        [hit()],
        {1: "a#1"},
    )
    assert not result.accepted()


def test_index_augmentation_drops_invented_entities_and_quotes():
    item = ChunkContext(
        chunk_id="a#1",
        context="신청 기한",
        summary="기한 안내",
        keywords=["신청", "환불"],
        entities=["없는기관"],
        source_quote=hit().source["body"],
    )
    bad = item.model_copy(update={"chunk_id": "invented#1"})
    runner = Runner(ContextBatch(chunks=[item, bad]), Quality(accepted_ids=["a#1", "invented#1"]))
    source = {**hit().source, "chunk_id": "a#1"}
    result = augment_batch(runner, None, [source])
    assert len(result) == 1
    assert result[0].keywords == ["신청"]
    assert result[0].entities == []
    assert source["body"] == hit().source["body"]


def test_rerank_rejects_unknown_ids_and_restores_missing_candidates():
    result = reasoning.rank_evidence(
        Runner(reasoning.Ranking(chunk_ids=["2", "unknown", "2"])),
        None,
        "q",
        QuestionPlan(),
        [hit(), hit("a#2")],
    )
    assert [h.chunk_id for h in result] == ["a#2", "a#1"]


def test_rerank_excerpt_uses_planned_search_terms_for_colloquial_questions():
    body = "휴직 절차\n신청 안내\n" + "다른 일반 안내 사항입니다.\n" * 60
    body += "업무상 부상에는 휴직 제도가 적용됩니다.\n" + "추가 안내.\n" * 8
    payloads = []

    class CaptureRunner:
        def run(self, llm, task, instruction, payload, schema, **kwargs):
            payloads.append(payload)
            return reasoning.Ranking(chunk_ids=["1"])

    reasoning.rank_evidence(
        CaptureRunner(),
        None,
        "일하다가 다치면 쉴 수 있어?",
        QuestionPlan(queries=["일하다가 다치면 쉴 수 있어?", "업무상 부상 휴직"]),
        [hit(body=body), hit("b#1")],
    )
    assert "업무상 부상에는 휴직 제도" in payloads[0]["candidates"][0]["excerpt"]


def test_source_metadata_cannot_become_answer_evidence():
    h = hit()
    h.source.update(summary="금액은 1억입니다.", contextual_text="금액은 1억입니다.")
    context = pipeline.make_context([h])
    assert context.blocks[0].text == h.source["body"]
    assert "1억" not in context.render()


def prepare_pipeline(monkeypatch, *, sufficient=True):
    monkeypatch.setattr(pipeline, "plan_question", lambda *a: QuestionPlan(requirements=["기한"]))
    monkeypatch.setattr(
        pipeline,
        "route_question",
        lambda *a, **k: Route(doc_ids=["a"], allow_multiple_docs=False, language="ko"),
    )
    searches = []

    def search(*args, **kwargs):
        searches.append(kwargs["doc_ids"][:])
        return Result(query="q", hits=[hit()])

    monkeypatch.setattr(pipeline, "search", search)
    monkeypatch.setattr(pipeline, "section_candidates", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "fetch_chunks", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "neighbors", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "recover_pages", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "rank_evidence", lambda runner, llm, question, plan, hits: hits)
    evidence = (
        EvidenceCheck(
            supports=[
                Support(
                    requirement="기한", sources=[Quote(chunk_id="a#1", quote=hit().source["body"])]
                )
            ]
        )
        if sufficient
        else EvidenceCheck(missing=["기한"], queries=["다른 문서를 검색하라"])
    )
    monkeypatch.setattr(pipeline, "check_evidence", lambda *a, **k: evidence)
    return searches


def test_retry_search_stays_inside_selected_document(monkeypatch):
    searches = prepare_pipeline(monkeypatch, sufficient=False)
    trace = pipeline.run_pipeline(
        None, None, None, "기한?", known_docs=["a", "b"], explicit_docs=["a"], cache=None
    )
    assert searches and all(docs == ["a"] for docs in searches)
    assert trace.answer.refusal == Refusal.INSUFFICIENT_EVIDENCE
    assert not trace.answer.text


def test_cross_document_scope_mismatch_is_explained_before_model_or_search(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("선택하지 않은 문서가 필요한 질문은 검색·모델 호출 전에 안내해야 합니다")

    monkeypatch.setattr(pipeline, "plan_question", unexpected_call)
    monkeypatch.setattr(pipeline, "search", unexpected_call)
    certificate = "서울시-평생학습포털-수료증-개인정보보호교육"
    trace = pipeline.run_pipeline(
        None,
        None,
        None,
        "취업규칙 교육훈련에 서울시 평생학습포털 수료증의 내용이 해당해",
        known_docs=["취업규칙-260417", certificate],
        explicit_docs=["취업규칙-260417"],
        cache=None,
    )
    assert trace.documents == ["취업규칙-260417"]
    assert certificate in trace.clarification
    assert trace.calls == 0
    assert trace.answer.refusal is None


def test_cross_document_evidence_survives_ranking_and_retry_without_comparison_keyword(monkeypatch):
    prepare_pipeline(monkeypatch, sufficient=False)
    from haeindex.routing import route_question

    monkeypatch.setattr(pipeline, "route_question", route_question)
    # 실제 한국어 문서 별칭과 해당 여부 질문으로 단일 문서 필터 회귀를 재현한다.
    docs = ["취업규칙-260417", "서울시-평생학습포털-수료증-개인정보보호교육"]
    question = "취업규칙 교육훈련에 서울시 평생학습포털 수료증의 내용이 해당해"
    monkeypatch.setattr(
        pipeline, "plan_question", lambda *a: QuestionPlan(requirements=[question])
    )
    policy = hit("policy", body="정보보안 및 개인정보보호 교육", doc=docs[0])
    certificate = hit("certificate", body="강의명: 개인정보보호 교육", doc=docs[1])
    seen_searches, seen_evidence = [], []

    def search(*args, **kwargs):
        seen_searches.append(set(kwargs["doc_ids"]))
        return Result(query="q", hits=[policy, certificate])

    def check(*args):
        seen_evidence.append({h.source["doc_id"] for h in args[-1]})
        return EvidenceCheck(missing=[question], queries=["개인정보보호 교육훈련"])

    monkeypatch.setattr(pipeline, "search", search)
    monkeypatch.setattr(pipeline, "check_evidence", check)
    trace = pipeline.run_pipeline(
        None,
        None,
        None,
        question,
        known_docs=docs,
        cache=None,
    )
    assert len(seen_evidence) == 2
    assert all(scope == set(docs) for scope in [*seen_searches, *seen_evidence])
    assert set(trace.documents) == set(docs)


def test_bracketed_title_is_reviewed_as_one_supported_cross_document_claim():
    title = "[권일용과 표창원의 질문들] 개인정보보호 교육"
    policy = hit("policy", body="3. 정보보안 및 개인정보보호 교육", doc="policy")
    certificate = hit("certificate", body=f"강 의 명 : {title}", doc="certificate")
    text = f"수료증의 강의명은 {title}이며 교육훈련 항목에 해당합니다.[cite:1][cite:2]"
    review = AnswerReview(
        claims=[
            ClaimReview(
                sentence_id=1,
                supported=True,
                sources=[
                    Quote(chunk_id=policy.chunk_id, quote=policy.source["body"]),
                    Quote(chunk_id=certificate.chunk_id, quote=certificate.source["body"]),
                ],
            )
        ]
    )
    checked = review_answer(
        Runner(review),
        None,
        "해당하는 교육이야?",
        QuestionPlan(),
        text,
        [policy, certificate],
        {1: "policy", 2: "certificate"},
    )
    assert checked.accepted()


def test_initial_search_uses_at_most_two_planned_query_variants(monkeypatch):
    prepare_pipeline(monkeypatch, sufficient=False)
    monkeypatch.setattr(
        pipeline,
        "plan_question",
        lambda *a: QuestionPlan(
            requirements=["기한"],
            queries=["원 질문", "정규화 용어", "관련 조건", "추가 질의"],
        ),
    )
    calls = []

    def search(*args, **kwargs):
        calls.append(kwargs)
        return Result(query="q", hits=[hit()])

    monkeypatch.setattr(pipeline, "search", search)
    pipeline.run_pipeline(None, None, None, "원 질문", known_docs=["a"], cache=None)
    assert calls[0]["extra_queries"] == ["정규화 용어", "관련 조건"]
    assert calls[0]["doc_ids"] == ["a"]


def test_readable_but_insufficient_evidence_does_not_trigger_vision(monkeypatch):
    prepare_pipeline(monkeypatch, sufficient=False)
    recoveries = []

    def recover(*args, **kwargs):
        recoveries.append(kwargs)
        return []

    monkeypatch.setattr(pipeline, "recover_pages", recover)
    trace = pipeline.run_pipeline(
        None, None, None, "기한?", known_docs=["a"], vision=object(), cache=None
    )
    assert any(call.get("max_pages") == 5 for call in recoveries)
    assert not any(call.get("use_vision") for call in recoveries)
    assert not any(stage.name == "reading_pdf" for stage in trace.stages)
    assert trace.answer.refusal == Refusal.INSUFFICIENT_EVIDENCE
    assert all(stage.seconds >= 0 for stage in trace.stages)


def test_only_requested_visual_evidence_is_sent_to_pdf_recovery(monkeypatch):
    prepare_pipeline(monkeypatch, sufficient=False)
    h = hit()
    check = EvidenceCheck(
        missing=["기한"],
        gaps=[
            EvidenceGap(requirement="기한", reason="table_layout", chunk_ids=[h.chunk_id]),
        ],
    )
    monkeypatch.setattr(pipeline, "check_evidence", lambda *a, **k: check)
    recoveries = []

    def recover(hits, **kwargs):
        recoveries.append(([item.chunk_id for item in hits], kwargs))
        return []

    monkeypatch.setattr(pipeline, "recover_pages", recover)
    trace = pipeline.run_pipeline(
        None,
        None,
        None,
        "기한?",
        known_docs=["a"],
        vision=object(),
        cache=None,
    )
    visual = [(ids, options) for ids, options in recoveries if options.get("use_vision")]
    assert len(visual) == 1
    assert visual[0][0] == [h.chunk_id]
    assert visual[0][1]["max_pages"] == 2
    assert visual[0][1]["question"] == "기한?"
    assert any(stage.name == "reading_pdf" for stage in trace.stages)


def test_not_found_is_reported_after_recovery_without_vision_or_draft(monkeypatch):
    searches = prepare_pipeline(monkeypatch, sufficient=False)
    check = EvidenceCheck(
        missing=["기한"],
        queries=["신청 기한"],
        gaps=[
            EvidenceGap(requirement="기한", reason="not_found"),
        ],
    )
    monkeypatch.setattr(pipeline, "check_evidence", lambda *a, **k: check)
    trace = pipeline.run_pipeline(
        None,
        None,
        None,
        "기한?",
        known_docs=["a"],
        vision=object(),
        cache=None,
    )
    assert len(searches) == 2
    assert trace.answer.refusal == Refusal.NOT_FOUND
    assert trace.stages[-1].name == "not_found"
    assert not any(stage.name in {"reading_pdf", "answering"} for stage in trace.stages)


def test_low_cosine_can_answer_when_raw_evidence_passes_review(monkeypatch):
    prepare_pipeline(monkeypatch)
    sentence = "신청 기한은 2024년 6월 1일입니다. [1]"
    monkeypatch.setattr(
        pipeline,
        "review_answer",
        lambda *a: AnswerReview(
            claims=[
                ClaimReview(
                    sentence_id=1,
                    supported=True,
                    sources=[Quote(chunk_id="a#1", quote=hit().source["body"])],
                )
            ]
        ),
    )
    trace = pipeline.run_pipeline(
        None,
        LLM(Draft(text=sentence).model_dump_json()),
        None,
        "기한?",
        known_docs=["a"],
        cache=None,
    )
    assert trace.answer.refusal is None
    assert trace.answer.cited == [1]
    assert "[cite:1]" in trace.answer.text


def test_rejected_answer_is_repaired_once_then_withheld(monkeypatch):
    prepare_pipeline(monkeypatch)
    monkeypatch.setattr(
        pipeline, "review_answer", lambda *a: AnswerReview(missing_requirements=["날짜가 틀립니다"])
    )

    class RejectedLLM(LLM):
        def chat(self, messages, **kwargs):
            self.calls += 1
            bad = "기한은 내일입니다. [cite:1]"
            response = (
                Draft(text=bad)
                if self.calls == 1
                else AnswerPatch(edits=[SentenceEdit(sentence_id=1, text=bad)])
            )
            return ChatResult(content=response.model_dump_json())

    llm = RejectedLLM()
    trace = pipeline.run_pipeline(None, llm, None, "기한?", known_docs=["a"], cache=None)
    assert llm.calls == 2
    assert trace.answer.refusal == Refusal.UNGROUNDED_CLAIMS
    assert not trace.answer.text


def test_repair_receives_rejected_sentence_and_validation_error(monkeypatch):
    prepare_pipeline(monkeypatch)
    bad = "신청 기한은 2025년 6월 1일입니다. [cite:1]"
    good = "신청 기한은 2024년 6월 1일입니다. [cite:1]"

    class SequenceLLM(LLM):
        def __init__(self, *responses):
            super().__init__()
            self.responses = iter(responses)
            self.payloads = []

        def chat(self, messages, **kwargs):
            self.payloads.append(json.loads(messages[-1]["content"]))
            return ChatResult(content=next(self.responses).model_dump_json())

    answer_llm = SequenceLLM(
        Draft(text=bad), AnswerPatch(edits=[SentenceEdit(sentence_id=1, text=good)])
    )
    analysis_llm = SequenceLLM(
        *[
            AnswerReview(
                claims=[
                    ClaimReview(
                        sentence_id=1,
                        supported=True,
                        sources=[Quote(chunk_id="a#1", quote=quote)],
                    )
                ],
                repair_instructions="수정이 필요하지 않습니다.",
            )
            for quote in ["신청 기한은 2025년 6월 1일까지입니다.", hit().source["body"]]
        ]
    )
    trace = pipeline.run_pipeline(
        None, answer_llm, analysis_llm, "기한?", known_docs=["a"], cache=None
    )
    feedback = answer_llm.payloads[1]["review"]
    assert feedback["issues"][0]["sentence"] == bad
    assert feedback["issues"][0]["sentence_id"] == 1
    assert "원문에 그대로 존재하지 않습니다" in feedback["issues"][0]["issue"]
    assert "수정이 필요하지 않습니다" not in feedback["instructions"]
    assert analysis_llm.payloads[0]["sentences"][0]["cited_chunk_ids"] == ["a#1"]
    assert trace.answer.refusal is None
    assert trace.answer.text == good
    assert trace.answer.citation_retries == 1


@pytest.mark.parametrize("exhaustive", [False, True])
def test_overview_repair_recovers_truncation_and_requires_source_review(monkeypatch, exhaustive):
    prepare_pipeline(monkeypatch)
    plan = QuestionPlan(intent="overview", exhaustive=exhaustive, requirements=["기한"])
    monkeypatch.setattr(pipeline, "plan_question", lambda *a: plan)
    bad = "신청 기한은 2025년 6월 1일입니다. [cite:1]"
    good = "신청 기한은 2024년 6월 1일입니다. [cite:1]"

    class RepairLLM(LLM):
        def __init__(self):
            super().__init__()
            self.limits = []
            self.schemas = []

        def chat(self, messages, **kwargs):
            self.calls += 1
            self.limits.append(kwargs["num_predict"])
            self.schemas.append(kwargs["response_schema"])
            if self.calls == 2:
                raise Truncated("max_output_tokens")
            response = (
                Draft(text=bad)
                if self.calls == 1
                else AnswerPatch(edits=[SentenceEdit(sentence_id=1, text=good)])
            )
            return ChatResult(content=response.model_dump_json())

    class ReviewLLM(LLM):
        def chat(self, messages, **kwargs):
            self.calls += 1
            payload = json.loads(messages[-1]["content"])
            correct = payload["sentences"][0]["text"] == good
            return ChatResult(
                content=AnswerReview(
                    claims=[
                        ClaimReview(
                            sentence_id=1,
                            supported=correct,
                            sources=[Quote(chunk_id="a#1", quote=hit().source["body"])],
                            issue="" if correct else "연도가 원문과 다릅니다.",
                        )
                    ]
                ).model_dump_json()
            )

    answer_llm, analysis_llm = RepairLLM(), ReviewLLM()
    trace = pipeline.run_pipeline(
        None, answer_llm, analysis_llm, "기한 규칙?", known_docs=["a"], cache=None
    )
    assert answer_llm.limits == [1800, 1800, 3600]
    assert analysis_llm.calls == 2
    assert trace.answer.refusal is None
    assert trace.answer.text.startswith(good)
    assert bad not in trace.answer.text
    assert trace.answer.citation_retries == 1
    assert not trace.reviews[0].accepted()
    assert trace.reviews[-1].accepted()
    assert ("주요 규칙 요약" in trace.answer.text) is not exhaustive
    assert ("maxLength" in answer_llm.schemas[0]["properties"]["text"]) is not exhaustive
    assert all("edits" in schema["properties"] for schema in answer_llm.schemas[1:])
    assert trace.repairs[0].edits[0].text == good


def test_reviewer_may_omit_citation_marker_but_must_match_actual_sentence():
    sentence = "신청 기한은 2024년 6월 1일입니다."
    review = AnswerReview(
        claims=[
            ClaimReview(
                sentence_id=1,
                supported=True,
                sources=[Quote(chunk_id="a#1", quote=hit().source["body"])],
            )
        ]
    )
    result = review_answer(
        Runner(review), None, "기한?", QuestionPlan(), sentence + " [cite:1]", [hit()], {1: "a#1"}
    )
    assert result.accepted()


def test_requirement_planning_cannot_add_unasked_conditions():
    plan = QuestionPlan(requirements=["교육 기간", "교육 시간", "환불 조건"])
    result = reasoning.plan_question(Runner(plan), None, "교육 기간과 교육 시간을 알려줘", [])
    assert result.requirements == ["교육 기간", "교육 시간"]


def test_index_updates_preserve_original_body_and_embedding(monkeypatch):
    from haeindex import augmentation

    source = {**hit().source, "chunk_id": "a#1", "seq": 0}
    monkeypatch.setattr(augmentation, "scan", lambda *a, **k: [{"_source": source}])
    monkeypatch.setattr(augmentation, "ensure_accuracy_fields", lambda *a: None)
    monkeypatch.setattr(augmentation, "section_groups", lambda *a: [])
    item = ChunkContext(
        chunk_id="a#1", context="신청 안내", summary="신청 기한", source_quote=source["body"]
    )
    monkeypatch.setattr(augmentation, "augment_batch", lambda *a: [item])
    captured = []

    def bulk(client, actions, **kwargs):
        captured.extend(actions)
        return len(actions), []

    monkeypatch.setattr(augmentation, "bulk", bulk)
    os_client = SimpleNamespace(
        indices=SimpleNamespace(refresh=lambda **k: None), delete_by_query=lambda **k: None
    )
    llm = LLM()
    llm.embed_batched = lambda texts: [[0.1] * 1024 for _ in texts]
    result = augmentation.enhance_document(os_client, llm, llm, "a", cache=None)
    assert result["accepted"] == 1
    update = captured[0]
    assert update["_op_type"] == "update"
    assert "body" not in update["doc"] and "embedding" not in update["doc"]
    assert "context_embedding" in update["doc"]
    assert source["body"] == hit().source["body"]


def test_citation_before_full_stop_still_matches_review_sentences():
    first = "기한은 2024년 6월 1일입니다 [cite:1]."
    second = "기한 뒤에는 신청할 수 없습니다 [cite:1]."
    h = hit(body="기한은 2024년 6월 1일입니다. 기한 뒤에는 신청할 수 없습니다.")
    review = AnswerReview(
        claims=[
            ClaimReview(
                sentence_id=i + 1,
                supported=True,
                sources=[Quote(chunk_id="a#1", quote=h.source["body"])],
            )
            for i, s in enumerate([first, second])
        ]
    )
    result = review_answer(
        Runner(review), None, "기한?", QuestionPlan(), first + " " + second, [h], {1: "a#1"}
    )
    assert result.accepted()

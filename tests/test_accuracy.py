import json
from types import SimpleNamespace

import pytest

from haeindex import pipeline, reasoning
from haeindex.answer import Refusal
from haeindex.augmentation import ChunkContext, ContextBatch, Quality, augment_batch
from haeindex.bedrock import ChatResult
from haeindex.llm_tasks import BudgetExceeded, TaskFailure, TaskRunner
from haeindex.reasoning import (
    AnswerReview,
    ClaimReview,
    Draft,
    EvidenceCheck,
    QuestionPlan,
    Quote,
    Support,
    check_evidence,
    review_answer,
)
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
        lambda *a, **k: SimpleNamespace(doc_ids=["a"], allow_multiple_docs=False),
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


def test_insufficient_evidence_reextracts_then_reads_relevant_page_image(monkeypatch):
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
    assert any(call.get("use_vision") and call.get("question") == "기한?" for call in recoveries)
    assert any(stage.name == "reading_pdf" for stage in trace.stages)
    assert all(stage.seconds >= 0 for stage in trace.stages)


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
    llm = LLM(Draft(text="기한은 내일입니다. [cite:1]").model_dump_json())
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

    answer_llm = SequenceLLM(Draft(text=bad), Draft(text=good))
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

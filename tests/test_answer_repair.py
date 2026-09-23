import json

import pytest

from haeindex.bedrock import ChatResult
from haeindex.llm_tasks import TaskFailure, TaskRunner
from haeindex.reasoning import (
    AnswerPatch,
    AnswerReview,
    ClaimReview,
    QuestionPlan,
    Quote,
    SentenceEdit,
    answer_sentences,
    apply_answer_patch,
    repair_answer,
)

UNSUPPORTED = "- 취업규칙에는 독감에 관한 별도 규정이 없습니다.[cite:1]"
APPROVED = (
    "- 사전 신고가 불가능한 경우에는 결근 당일 전화, 메신저 등으로 통보하고 "
    "사후에 소정의 절차를 밟아야 합니다.[cite:1]"
)
SOURCE = (
    "다만, 불가피한 경우에는 결근 당일에 전화, 메신저 등으로 통보하고 "
    "사후에 소정의 절차를 밟아야 한다."
)


@pytest.mark.parametrize(
    "title",
    ["[권일용과 표창원의 질문들] 개인정보보호 교육", "[필수] 정보보안 교육"],
)
def test_bracketed_course_title_stays_with_its_claim_and_citations(title):
    first = f"수료증의 강의명은 {title}이며 취업규칙의 교육훈련 항목에 해당합니다.[cite:1][cite:2]"
    second = "비용 지원은 회사의 사전 승인을 받은 경우 가능합니다.[cite:3]"
    assert answer_sentences(first + "\n" + second) == {1: first, 2: second}


def test_real_citations_still_separate_adjacent_review_claims():
    assert answer_sentences("첫 주장입니다.[cite:1]. 두 번째 주장입니다.[cite:2]") == {
        1: "첫 주장입니다.[cite:1]",
        2: "두 번째 주장입니다.[cite:2]",
    }


def initial_review():
    return AnswerReview(
        claims=[
            ClaimReview(
                sentence_id=1, supported=False, issue="문서 전체에 규정이 없다고 단정했습니다."
            ),
            ClaimReview(
                sentence_id=2, supported=True, sources=[Quote(chunk_id="a#1", quote=SOURCE)]
            ),
        ]
    )


def test_removing_unsupported_absence_claim_preserves_approved_condition_and_citation():
    patch = AnswerPatch(edits=[SentenceEdit(sentence_id=1, text="")])
    result = apply_answer_patch(UNSUPPORTED + "\n" + APPROVED, initial_review(), patch)
    assert result == APPROVED
    assert "사전 신고가 불가피한" not in result


@pytest.mark.parametrize(
    "edits",
    [
        [SentenceEdit(sentence_id=2, text=APPROVED.replace("불가능한", "불가피한"))],
        [SentenceEdit(sentence_id=2, text="")],
        [SentenceEdit(sentence_id=999, text="삭제합니다.[cite:1]")],
        [
            SentenceEdit(sentence_id=1, text=""),
            SentenceEdit(sentence_id=1, text="중복 수정[cite:1]"),
        ],
    ],
)
def test_patch_cannot_edit_delete_or_duplicate_forbidden_targets(edits):
    with pytest.raises(ValueError):
        apply_answer_patch(
            UNSUPPORTED + "\n" + APPROVED, initial_review(), AnswerPatch(edits=edits)
        )


def test_only_missing_requirements_allow_additions_and_leave_approved_text_intact():
    review = initial_review()
    patch = AnswerPatch(
        edits=[SentenceEdit(sentence_id=1, text="")],
        additions=["- 지각 시간은 무급 처리가 원칙입니다.[cite:2]"],
    )
    with pytest.raises(ValueError, match="새 문장"):
        apply_answer_patch(UNSUPPORTED + "\n" + APPROVED, review, patch)
    review.missing_requirements = ["지각 시간의 급여"]
    result = apply_answer_patch(UNSUPPORTED + "\n" + APPROVED, review, patch)
    assert answer_sentences(result) == {1: APPROVED, 2: patch.additions[0]}


def test_unreviewed_or_invalidly_quoted_sentences_are_editable():
    review = AnswerReview(claims=[ClaimReview(sentence_id=1, supported=True, sources=[])])
    patch = AnswerPatch(
        edits=[
            SentenceEdit(sentence_id=1, text=""),
            SentenceEdit(sentence_id=2, text=APPROVED),
        ]
    )
    assert apply_answer_patch(UNSUPPORTED + "\n" + APPROVED, review, patch) == APPROVED


class PatchLLM:
    chat_model = "patch-test-model"
    num_ctx = 16384

    def __init__(self, *patches):
        self.patches = iter(patches)
        self.payloads = []

    def chat(self, messages, **kwargs):
        self.payloads.append(json.loads(messages[-1]["content"]))
        return ChatResult(content=next(self.patches).model_dump_json())


def test_locked_edit_is_retried_and_only_valid_patch_is_cached(tmp_path):
    forbidden = AnswerPatch(edits=[SentenceEdit(sentence_id=2, text="반대로 수정합니다.[cite:1]")])
    correct = AnswerPatch(edits=[SentenceEdit(sentence_id=1, text="")])
    llm = PatchLLM(forbidden, correct)
    runner = TaskRunner(cache=tmp_path)
    text = UNSUPPORTED + "\n" + APPROVED
    for _ in range(2):
        merged, _ = repair_answer(
            runner, llm, "독감에 걸리면?", QuestionPlan(), text, initial_review(), {}
        )
        assert merged == APPROVED
    assert runner.calls == 2
    assert runner.events[0].error
    assert runner.events[-1].cache_hit
    assert llm.payloads[0]["sentences"][1]["locked"]
    cached = list(tmp_path.glob("*.json"))
    assert len(cached) == 1
    assert AnswerPatch.model_validate_json(cached[0].read_text()) == correct


def test_merged_overview_length_is_checked_before_caching_patch(tmp_path):
    oversized = AnswerPatch(edits=[SentenceEdit(sentence_id=1, text="긴 설명" * 400)])
    llm = PatchLLM(oversized, oversized)
    with pytest.raises(TaskFailure, match="ValidationError"):
        repair_answer(
            TaskRunner(cache=tmp_path),
            llm,
            "규칙 요약",
            QuestionPlan(intent="overview"),
            UNSUPPORTED + "\n" + APPROVED,
            initial_review(),
            {},
        )
    assert len(llm.payloads) == 2
    assert not list(tmp_path.glob("*.json"))

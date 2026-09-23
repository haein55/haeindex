"""질문 계획, 원문 근거 충분성·충돌·답변의 주장 검증."""

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from haeindex.bedrock import Bedrock
from haeindex.llm_tasks import TaskRunner
from haeindex.routing import detect_language
from haeindex.search import Hit


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Constraint(Record):
    field: str
    value: str
    operator: Literal["eq", "lt", "lte", "gt", "gte", "contains"] = "eq"
    quote: str


class QuestionPlan(Record):
    intent: Literal["fact", "procedure", "comparison", "conditions", "overview"] = "fact"
    exhaustive: bool = False
    requirements: list[str] = Field(default_factory=list, max_length=4)
    constraints: list[Constraint] = Field(default_factory=list, max_length=8)
    queries: list[str] = Field(default_factory=list, max_length=4)
    ambiguous: bool = False
    clarification: str = ""


def compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def quote_exists(quote: str, source: str) -> bool:
    needle = compact(quote)
    if not needle or needle not in compact(source):
        return False
    if len(needle) >= min(6, len(compact(source))):
        return True
    # 짧은 목록 항목도 독립된 원문 한 줄이면 유효하다. 긴 문장의 짧은 일부는 인정하지 않는다.
    return any(c.isalpha() for c in needle) and any(
        needle == compact(line) for line in source.splitlines()
    )


def plan_question(
    runner: TaskRunner, llm: Bedrock, question: str, documents: list[str]
) -> QuestionPlan:
    result = runner.run(
        llm,
        "question-plan",
        "질문을 검색 계획으로 변환하세요. 답을 만들지 마세요. requirements는 질문에 명시된 "
        "답변 요구만 1~4개로 나누세요. 각 requirement는 질문 속 표현을 그대로 복사하세요. "
        "예: 교육 기간과 교육 시간을 알려줘 → [교육 기간, 교육 시간]. "
        "질문에 없는 요구나 문서 활용 지시는 추가하지 마세요. "
        "'휴가 규칙을 알려줘'처럼 넓은 제도의 안내·요약은 intent=overview입니다. "
        "특정 항목의 수치·조건·종류를 묻는 질문은 해당 intent로 분류하세요. "
        "질문이 모든 조항·세부 조건·예외를 빠짐없이 설명하라고 명시한 경우에만 "
        "exhaustive=true로 설정하세요. 검색 대상인 '전체 문서'는 상세 설명 요구가 아닙니다. "
        "constraints에는 질문에 실제 있는 대상·연도·수치·항행구역 등만 넣고 quote는 질문의 "
        "해당 표현을 그대로 복사하세요. queries는 원래 의미를 보존한 검색어이며 최대3개입니다. "
        "질문의 일반적인 단어 '서로'만으로 comparison으로 분류하지 마세요. "
        "문서명이 없는 질문은 검색으로 찾을 수 있으므로 그 자체로 ambiguous가 아닙니다. "
        "'이 파일'처럼 지시 대상이 없거나 조번호만 있고 여러 문서가 가능하면 ambiguous를 "
        "설정하고 질문과 같은 언어로 짧은 확인 질문을 작성하세요.",
        {"question": question, "documents": documents[:80]},
        QuestionPlan,
        num_predict=1000,
    )
    result.constraints = [
        c for c in result.constraints if compact(c.quote) and compact(c.quote) in compact(question)
    ]
    result.requirements = [
        x.strip() for x in result.requirements if compact(x) and compact(x) in compact(question)
    ] or [question]
    result.queries = list(
        dict.fromkeys([question, *[q.strip() for q in result.queries if q.strip()]])
    )[:4]
    return result


def is_overview(plan: QuestionPlan) -> bool:
    return plan.intent == "overview" and not plan.exhaustive


def answer_scope(plan: QuestionPlan) -> str:
    if is_overview(plan):
        return (
            "답변 범위는 주요 규칙의 요약입니다. 확인한 대표 제도·종류를 고르게 다루되 "
            "최대 8개 항목, 인용 포함 1400자 이내로 작성하세요. 각 항목은 핵심 대상·내용과 "
            "그 주장의 의미를 바꾸는 필수 조건만 담으세요. 세부 산식·신청 절차·관련 제도를 "
            "전부 나열하지 마세요. 선택한 주장의 수치·유무급·적용 대상·예외를 바꾸거나 "
            "조건부 권리를 무조건적인 권리로 요약해서는 안 됩니다. 복잡한 제도는 조건을 "
            "삭제한 단정 대신 원문으로 확인한 범위의 짧은 설명을 사용하세요. "
            "검토 시 사용자가 명시한 요구와 요약에 실제 담긴 주장을 확인하세요. 관련 조항의 "
            "모든 세부사항을 쓰지 않았다는 이유만으로 missing이나 수정 요구를 만들지 마세요. "
            "다만 명시된 질문 요구의 누락, 사실 오류와 주장의 의미를 바꾸는 조건 누락은 "
            "반드시 거절하세요. 수정도 같은 요약 범위 안에서 잘못된 주장을 고치거나 "
            "불필요한 세부 주장을 제거하며, 관련 규정 전체로 확장하지 마세요."
        )
    return (
        "질문에 명시된 항목·조건·예외는 모두 답하고 검토하세요. 특정 목록의 종류를 물으면 "
        "그 목록의 모든 항목을 포함하세요. 상세·전부 설명을 요구했다면 요약으로 대체하지 "
        "마세요. 질문하지 않은 관련 제도까지 답변 요구를 늘리지는 마세요."
    )


class Quote(Record):
    chunk_id: str
    quote: str


class Support(Record):
    requirement: str
    sources: list[Quote] = Field(default_factory=list, max_length=5)


class Conflict(Record):
    description: str
    sources: list[Quote] = Field(default_factory=list, max_length=4)
    needs_clarification: bool = True


class EvidenceGap(Record):
    requirement: str
    reason: Literal["not_found", "missing_context", "table_layout", "diagram", "unreadable_text"]
    chunk_ids: list[str] = Field(default_factory=list, max_length=4)


VISUAL_GAP_REASONS = frozenset({"table_layout", "diagram", "unreadable_text"})


class EvidenceCheck(Record):
    supports: list[Support] = Field(default_factory=list, max_length=4)
    missing: list[str] = Field(default_factory=list, max_length=4)
    conflicts: list[Conflict] = Field(default_factory=list, max_length=3)
    queries: list[str] = Field(default_factory=list, max_length=2)
    clarification: str = ""
    gaps: list[EvidenceGap] = Field(default_factory=list, max_length=4)

    def complete(self, requirements: Sequence[str]) -> bool:
        supported = {s.requirement for s in self.supports if s.sources}
        return (
            set(requirements) <= supported
            and not self.missing
            and not self.clarification
            and not any(c.needs_clarification for c in self.conflicts)
        )

    def visual_chunk_ids(self) -> list[str]:
        return list(
            dict.fromkeys(
                cid
                for gap in self.gaps
                if gap.requirement in self.missing and gap.reason in VISUAL_GAP_REASONS
                for cid in gap.chunk_ids
            )
        )

    def not_found(self) -> bool:
        reasons = {gap.requirement: gap.reason for gap in self.gaps}
        return (
            bool(self.missing)
            and not (self.supports or self.clarification or self.conflicts)
            and all(reasons.get(requirement) == "not_found" for requirement in self.missing)
        )


def evidence_payload(hits: Sequence[Hit], max_chars: int = 11000) -> list[dict]:
    remaining = max_chars
    out = []
    for hit in hits:
        source = hit.source
        body = str(source.get("body", ""))
        if len(body) > remaining:
            continue
        remaining -= len(body)
        out.append(
            {
                "chunk_id": hit.chunk_id,
                "doc_id": source.get("doc_id"),
                "page": source.get("page"),
                "end_page": source.get("end_page"),
                "heading": source.get("path") or source.get("title"),
                "body": body,
                "origin": source.get("evidence_origin", "indexed_pdf"),
            }
        )
    return out


def valid_quotes(quotes: Sequence[Quote], hits: Sequence[Hit]) -> list[Quote]:
    bodies = {h.chunk_id: str(h.source.get("body", "")) for h in hits}
    return [q for q in quotes if q.chunk_id in bodies and quote_exists(q.quote, bodies[q.chunk_id])]


def check_evidence(
    runner: TaskRunner,
    llm: Bedrock,
    question: str,
    plan: QuestionPlan,
    hits: Sequence[Hit],
) -> EvidenceCheck:
    evidence = evidence_payload(hits)
    result = runner.run(
        llm,
        "evidence-check",
        "각 requirements에 직접 답할 원문 근거가 있는지 조사하세요. supports.requirement는 "
        "입력 요구 문장을 그대로 사용하세요. supports.sources에는 답의 수치·대상·조건을 "
        "실제로 포함한 충분한 길이의 원문 quote와 chunk_id만 넣으세요. 제목 일치, 주제가 "
        "비슷함, 외부 지식은 근거가 아닙니다. 표는 행·열·단위·연도·대상·미만/이상을 함께 "
        "구분할 수 있어야 합니다. '반환'을 '귀속', '제출'을 '교부' 규정으로 대신하지 마세요. "
        "질문의 전제가 원문과 다르면 적용 범위를 구분해 설명하는 것도 유효한 답입니다. "
        "예/아니오 질문은 긍정뿐 아니라 부정·조건부 답변도 supports로 인정하세요. "
        "질문이 지칭한 제도의 적용 대상이 다르면 그 범위를 명시한 원문과 실제 해당하는 "
        "제도의 원문을 함께 인용해 답할 수 있습니다. 용어가 다르다는 이유만으로 missing으로 "
        "처리하지 마세요. 단, 다른 대상을 위한 혜택을 질문 대상에게 적용하거나 명시되지 않은 "
        "금지·권리·수치를 추론해서는 안 됩니다. 목록을 물으면 해당 목록 전체를 확인하고, "
        "질문하지 않은 다른 제도나 조건까지 필수 요구로 늘리지 마세요. "
        "없는 근거는 missing에 해당 requirement를 넣으세요. 더 찾을 수 있다면 문서에 "
        "실제로 나타난 용어를 써 queries 최대2개를 제안하세요. 같은 주장의 충돌은 양쪽 "
        "원문을 conflicts에 넣고, 서로 다른 조건이면 충돌로 오인하지 마세요. 명시된 시행일 "
        "근거 없이 최신 규정을 추측하지 마세요. 해결 불가 충돌에만 clarification을 쓰세요. "
        "missing의 각 요구에 대해 gaps에 원인을 기록하세요. not_found는 검색 근거에 해당 "
        "내용이 없음, missing_context는 이어지는 조항·조건이 더 필요함입니다. "
        "이미지 확인으로 해결할 구체적인 문제가 있을 때만 table_layout(표의 행·열 관계), "
        "diagram(도식·화살표), unreadable_text(손상·누락된 문자)를 사용하고, 해당 문제가 "
        "보이는 입력 청크 ID를 chunk_ids에 넣으세요. 단순히 답이 없거나 적용 대상이 다른 "
        "경우는 이미지 판독 사유가 아닙니다. 충분히 읽히는 본문을 더 읽기 위해 비전을 "
        "요청하지 마세요. 답할 근거가 충분하면 missing과 gaps는 비우세요. " + answer_scope(plan),
        {"question": question, "plan": plan.model_dump(), "evidence": evidence},
        EvidenceCheck,
        num_predict=2300,
    )
    for support in result.supports:
        support.sources = valid_quotes(support.sources, hits)
    result.supports = [
        s for s in result.supports if s.requirement in plan.requirements and s.sources
    ]
    covered = {s.requirement for s in result.supports}
    result.missing = list(
        dict.fromkeys(
            [
                *[r for r in result.missing if r in plan.requirements],
                *[r for r in plan.requirements if r not in covered],
            ]
        )
    )
    # 모델이 실제로 본 청크와 아직 부족한 요구에만 이미지 복구를 허용한다.
    shown_ids = {item["chunk_id"] for item in evidence}
    gaps = {}
    for gap in result.gaps:
        if gap.requirement not in result.missing:
            continue
        gap.chunk_ids = list(dict.fromkeys(cid for cid in gap.chunk_ids if cid in shown_ids))
        if gap.reason in VISUAL_GAP_REASONS and not gap.chunk_ids:
            gap.reason = "missing_context"
        gaps.setdefault(gap.requirement, gap)
    result.gaps = [
        gaps.get(requirement, EvidenceGap(requirement=requirement, reason="missing_context"))
        for requirement in result.missing
    ]
    for conflict in result.conflicts:
        conflict.sources = valid_quotes(conflict.sources, hits)
    result.conflicts = [c for c in result.conflicts if len({q.chunk_id for q in c.sources}) >= 2]
    if not any(c.needs_clarification for c in result.conflicts):
        result.clarification = ""
    elif not result.clarification:
        result.clarification = "서로 다른 근거가 있습니다. 적용할 문서와 조건을 지정해 주세요."
    return result


class Ranking(Record):
    chunk_ids: list[str] = Field(max_length=30)


def relevant_excerpt(body: str, question: str, limit: int = 900) -> str:
    if len(body) <= limit:
        return body
    terms = [t.lower() for t in re.findall(r"[가-힣A-Za-z0-9]{2,}", question)]
    lines = body.splitlines()
    scores = [(sum(t in line.lower() for t in terms), i) for i, line in enumerate(lines)]
    chosen = {0, 1}
    for _, i in sorted(scores, reverse=True)[:4]:
        chosen.update(range(max(0, i - 1), min(len(lines), i + 3)))
    return "\n".join(lines[i] for i in sorted(chosen) if i < len(lines))[:limit]


def rank_evidence(
    runner: TaskRunner,
    llm: Bedrock,
    question: str,
    plan: QuestionPlan,
    hits: list[Hit],
) -> list[Hit]:
    if len(hits) < 2:
        return hits
    excerpt_query = " ".join([question, *plan.queries[1:3]])
    candidates = [
        {
            "chunk_id": str(i + 1),
            "doc_id": h.source.get("doc_id"),
            "heading": h.source.get("path") or h.source.get("title"),
            "excerpt": relevant_excerpt(str(h.source.get("body", "")), excerpt_query, 550),
        }
        for i, h in enumerate(hits[:20])
    ]
    order = runner.run(
        llm,
        "evidence-rank",
        "질문의 조건과 요구에 직접 답하는 후보부터 chunk_id를 재정렬하세요. 제목보다 "
        "본문을 우선하고 목차·단순 언급은 내리세요. 조건·예외의 이어지는 청크도 우선하세요. "
        "목록에 없는 ID를 만들지 말고 상위 후보 최대20개를 반환하세요.",
        {"question": question, "plan": plan.model_dump(), "candidates": candidates},
        Ranking,
        num_predict=1800,
    )
    by_id = {str(i + 1): h for i, h in enumerate(hits[:20])}
    ids = list(dict.fromkeys([*[i for i in order.chunk_ids if i in by_id], *by_id]))
    return [by_id[i] for i in ids]


class Draft(Record):
    text: str


class OverviewDraft(Draft):
    text: str = Field(max_length=1400)


class ClaimReview(Record):
    sentence_id: int = Field(ge=1)
    supported: bool
    sources: list[Quote] = Field(default_factory=list, max_length=5)
    issue: str = ""


class AnswerReview(Record):
    claims: list[ClaimReview] = Field(default_factory=list, max_length=20)
    missing_requirements: list[str] = Field(default_factory=list, max_length=6)
    conflicts: list[str] = Field(default_factory=list, max_length=4)
    repair_instructions: str = ""

    def accepted(self) -> bool:
        return (
            bool(self.claims)
            and all(c.supported and c.sources for c in self.claims)
            and not (self.missing_requirements or self.conflicts)
        )


def answer_sentences(text: str) -> dict[int, str]:
    # 강의명·문서명에 있는 일반 대괄호는 문장 경계가 아니다. 실제 출처 표기만 나눈다.
    separated = re.sub(r"(\[cite:[\d, ]+\])(?:[.!?。])?\s+", r"\1\n", text)
    sentences = [
        s.strip()
        for s in separated.splitlines()
        if re.search(r"[가-힣A-Za-z]", s)
    ]
    return {i + 1: sentence for i, sentence in enumerate(sentences)}


class SentenceEdit(Record):
    sentence_id: int = Field(ge=1)
    text: str


class AnswerPatch(Record):
    edits: list[SentenceEdit] = Field(default_factory=list, max_length=20)
    additions: list[str] = Field(default_factory=list, max_length=6)


def locked_sentence_ids(text: str, review: AnswerReview) -> set[int]:
    locked = set()
    for n in answer_sentences(text):
        claims = [c for c in review.claims if c.sentence_id == n]
        if len(claims) == 1 and claims[0].supported and claims[0].sources:
            locked.add(n)
    return locked


def apply_answer_patch(text: str, review: AnswerReview, patch: AnswerPatch) -> str:
    sentences = answer_sentences(text)
    editable = sentences.keys() - locked_sentence_ids(text, review)
    edits = {}
    for edit in patch.edits:
        if edit.sentence_id not in editable:
            raise ValueError("검토를 통과했거나 존재하지 않는 문장은 수정할 수 없습니다")
        if edit.sentence_id in edits:
            raise ValueError("같은 문장의 수정이 중복되었습니다")
        replacement = edit.text.strip()
        if replacement and not answer_sentences(replacement):
            raise ValueError("교체할 문장에 답변 내용이 없습니다")
        edits[edit.sentence_id] = replacement
    if patch.additions and not (review.missing_requirements or review.conflicts):
        raise ValueError("누락 요구나 충돌 보완이 없으면 새 문장을 추가할 수 없습니다")
    additions = [s.strip() for s in patch.additions]
    if any(not answer_sentences(s) for s in additions):
        raise ValueError("추가할 문장에 답변 내용이 없습니다")
    return "\n".join(
        s for s in [*[edits.get(n, original) for n, original in sentences.items()], *additions] if s
    )


def repair_answer(
    runner: TaskRunner,
    llm: Bedrock,
    question: str,
    plan: QuestionPlan,
    text: str,
    review: AnswerReview,
    payload: dict,
) -> tuple[str, AnswerPatch]:
    # 대상 문장 검증도 캐시 읽기·모델 재시도의 안쪽에서 수행한다.
    class CheckedPatch(AnswerPatch):
        @model_validator(mode="after")
        def check_edits(self):
            merged = apply_answer_patch(text, review, self)
            if is_overview(plan):
                OverviewDraft(text=merged)
            return self

    locked = locked_sentence_ids(text, review)
    patch = runner.run(
        llm,
        "answer-repair",
        answer_instruction(question, plan)
        + "\n이번 작업은 전체 답변 작성이 아니라 문장별 수정입니다. text 필드의 전체 답변 대신 "
        "edits와 additions만 반환하세요. locked=true 문장은 인용·표현·조건을 포함해 수정하거나 "
        "삭제할 수 없습니다. 검토에서 지적된 locked=false 문장만 sentence_id로 지정하여 "
        "교체하세요. 해당 주장이 불필요하거나 근거가 없으면 text를 빈 문자열로 하여 삭제할 수 "
        "있습니다. 바꾸지 않을 문장은 edits에서 생략하세요. 기존 문장을 모두 다시 쓰지 마세요. "
        "검토의 missing 또는 conflicts 보완이 필요한 경우에만 additions에 인용을 갖춘 새 "
        "문장을 추가하세요. 통과한 문장을 부정하거나 뒤집는 내용을 추가하지 마세요.",
        {
            **payload,
            "sentences": [
                {"sentence_id": n, "text": s, "locked": n in locked}
                for n, s in answer_sentences(text).items()
            ],
            "review": {
                "instructions": review.repair_instructions,
                "missing": review.missing_requirements,
                "conflicts": review.conflicts,
                "issues": [
                    {
                        "sentence_id": c.sentence_id,
                        "sentence": answer_sentences(text).get(c.sentence_id, ""),
                        "issue": c.issue,
                    }
                    for c in review.claims
                    if not c.supported
                ],
            },
        },
        CheckedPatch,
        num_predict=1800,
        max_predict=3600,
    )
    return apply_answer_patch(text, review, patch), patch


def review_answer(
    runner: TaskRunner,
    llm: Bedrock,
    question: str,
    plan: QuestionPlan,
    text: str,
    hits: list[Hit],
    citation_ids: dict[int, str],
) -> AnswerReview:
    sentence_map = answer_sentences(text)
    cited_chunks = {}
    for n, sentence in sentence_map.items():
        numbers = [
            int(number)
            for group in re.findall(r"\[cite:([\d, ]+)\]", sentence)
            for number in re.findall(r"\d+", group)
        ]
        cited_chunks[n] = list(dict.fromkeys(citation_ids[i] for i in numbers if i in citation_ids))
    result = runner.run(
        llm,
        "answer-review",
        "답변의 모든 사실 주장을 문장별로 검증하세요. 인용한 청크의 원문만 근거로 삼으세요. "
        "수치·단위·연도·직책·선종·적용 대상·의무/가능·조건·예외가 달라지면 supported=false. "
        "원문에 없는 추가 주장도 실패입니다. sentences의 각 sentence_id를 그대로 사용하고 "
        "그 번호의 문장 전체를 검토하세요. 문장을 요약하거나 바꾸지 마세요. "
        "sources는 각 문장의 cited_chunk_ids에 있는 청크에서만 근거를 그대로 복사하세요. "
        "같은 내용을 설명하더라도 다른 청크로 대체하지 마세요. quote는 원문의 연속된 "
        "구간이며 괄호·낱말을 생략하거나 문장을 다듬지 마세요. "
        "짧은 목록 항목은 번호를 포함한 원문 한 줄 전체를 quote로 복사하세요. "
        "원문에 명시된 적용 범위·조건의 차이를 설명하는 부정·조건부 답변도 검토하세요. "
        "질문의 전제를 그대로 긍정해야 정답인 것은 아닙니다. 관련 제도를 구분한 설명이 "
        "인용 원문과 일치하면 인정하고, 명시되지 않은 금지나 권리는 인정하지 마세요. "
        "모든 requirements에 답했는지와 중요한 조건·예외 누락을 검사하세요. "
        "조건 누락이나 충돌 때문에 기존 문장을 바꿔야 한다면 해당 문장의 supported를 "
        "false로 표시하고 issue에 수정할 부분을 명시하세요. 모든 문장이 맞고 단지 새로운 "
        "내용을 추가해야 한다면 missing_requirements에 적으세요. "
        "불확실한 표에서 등급을 추측한 답은 거부하세요. repair_instructions에 구체적인 "
        "수정 지시를 쓰되 외부 지식으로 답을 추가하지 마세요. " + answer_scope(plan),
        {
            "question": question,
            "requirements": plan.requirements,
            "constraints": [c.model_dump() for c in plan.constraints],
            "sentences": [
                {"sentence_id": n, "text": sentence, "cited_chunk_ids": cited_chunks[n]}
                for n, sentence in sentence_map.items()
            ],
            "citation_ids": citation_ids,
            "evidence": evidence_payload(hits),
        },
        AnswerReview,
        num_predict=2800,
        max_predict=5600,
    )

    # 검토 대상은 모델이 다시 쓴 문장이 아니라 입력한 문장의 고정 번호다.
    reviewed = set()
    validation_issues = []
    for claim in result.claims:
        sentence = sentence_map.get(claim.sentence_id, "")
        issues = []
        if not sentence:
            issues.append("검토 대상에 없는 문장 번호입니다.")
        elif claim.sentence_id in reviewed:
            issues.append("동일한 문장을 중복 검토했습니다.")
        claim.sources = valid_quotes(claim.sources, hits)
        if not claim.sources:
            issues.append("검토에 사용한 quote가 원문에 그대로 존재하지 않습니다.")
        allowed = set(cited_chunks.get(claim.sentence_id, []))
        if not allowed:
            issues.append("문장에 유효한 출처 번호가 없습니다.")
        elif claim.sources and not any(s.chunk_id in allowed for s in claim.sources):
            issues.append("검토 근거의 청크가 문장에 붙인 출처와 다릅니다.")
        claim.sources = [s for s in claim.sources if s.chunk_id in allowed]
        if issues:
            claim.supported = False
            claim.issue = " ".join(filter(None, [claim.issue, *issues]))
            validation_issues.append(f"문장 {claim.sentence_id}: {claim.issue}")
        elif not claim.supported and not claim.issue:
            claim.issue = "문장의 주장·조건을 인용 원문으로 확인하지 못했습니다."
        reviewed.add(claim.sentence_id)
    if set(sentence_map) - reviewed:
        result.missing_requirements.append("검토되지 않은 답변 문장이 있습니다")
    if validation_issues:
        # 모델의 '수정 불필요' 판정보다 실제 출처·원문 대조 결과를 우선한다.
        result.repair_instructions = (
            "출처·원문 대조에서 다음 오류가 확인되었습니다. 해당 문장을 원문과 출처에 맞게 "
            "수정하고 질문에 불필요한 문장은 삭제하세요. 같은 답변을 반복하지 마세요. "
            + " ".join(validation_issues)
        )
    return result


def answer_instruction(question: str, plan: QuestionPlan | None = None) -> str:
    language = "한국어" if detect_language(question) == "ko" else "English"
    return (
        f"{language}로만 답하세요. 제공한 원문에 직접 있는 사실만 쓰세요. "
        "질문의 모든 requirements에 답하되 불필요한 추가 설명은 하지 마세요. "
        "질문한 대상과 직접 관련된 사실만 쓰고 다른 대상·용도의 설명을 섞지 마세요. "
        "질문의 전제나 제도명이 원문의 적용 범위와 다르면 그 차이를 설명하고, 실제 해당하는 "
        "규정이 근거에 있으면 함께 안내하세요. 근거가 없는 금지나 권리를 단정하지 마세요. "
        "검색된 조항에 없다는 이유로 문서 전체에 별도 규정이 없다고 단정하지 마세요. "
        "방법을 물으면 준비·실행·완료 순서로 간결하게 안내하세요. "
        "숫자·연도·단위·대상·조건·예외를 보존하세요. 원문에 없는 계산을 하지 마세요. "
        "각 문장/목록 항목 끝에 [cite:n]을 붙이세요. 표의 행·열을 혼동하지 마세요. "
        "목록 답변은 불필요한 제목·도입문을 생략하세요. 도입문을 쓴다면 그 문장에도 "
        "인용을 붙이세요. 짧은 목록 항목도 빠뜨리지 마세요. "
        "수정 요청에서는 지적된 문장과 출처를 실제로 고치고 필요한 조건을 보존하세요. "
        "내용을 지어내는 대신 근거 부족을 명시하세요. text 필드에 완성된 답변을 쓰세요. "
        + answer_scope(plan or QuestionPlan())
    )

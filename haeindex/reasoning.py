"""질문 계획, 원문 근거 충분성·충돌·답변의 주장 검증."""

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    requirements: list[str] = Field(default_factory=list, max_length=4)
    constraints: list[Constraint] = Field(default_factory=list, max_length=8)
    queries: list[str] = Field(default_factory=list, max_length=4)
    ambiguous: bool = False
    clarification: str = ""


def compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def quote_exists(quote: str, source: str) -> bool:
    needle = compact(quote)
    return (
        len(needle) >= min(6, len(compact(source))) and bool(needle) and needle in compact(source)
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


class EvidenceCheck(Record):
    supports: list[Support] = Field(default_factory=list, max_length=4)
    missing: list[str] = Field(default_factory=list, max_length=4)
    conflicts: list[Conflict] = Field(default_factory=list, max_length=3)
    queries: list[str] = Field(default_factory=list, max_length=2)
    clarification: str = ""

    def complete(self, requirements: Sequence[str]) -> bool:
        supported = {s.requirement for s in self.supports if s.sources}
        return (
            set(requirements) <= supported
            and not self.missing
            and not self.clarification
            and not any(c.needs_clarification for c in self.conflicts)
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
    result = runner.run(
        llm,
        "evidence-check",
        "각 requirements에 직접 답할 원문 근거가 있는지 조사하세요. supports.requirement는 "
        "입력 요구 문장을 그대로 사용하세요. supports.sources에는 답의 수치·대상·조건을 "
        "실제로 포함한 충분한 길이의 원문 quote와 chunk_id만 넣으세요. 제목 일치, 주제가 "
        "비슷함, 외부 지식은 근거가 아닙니다. 표는 행·열·단위·연도·대상·미만/이상을 함께 "
        "구분할 수 있어야 합니다. '반환'을 '귀속', '제출'을 '교부' 규정으로 대신하지 마세요. "
        "없는 근거는 missing에 해당 requirement를 넣으세요. 더 찾을 수 있다면 문서에 "
        "실제로 나타난 용어를 써 queries 최대2개를 제안하세요. 같은 주장의 충돌은 양쪽 "
        "원문을 conflicts에 넣고, 서로 다른 조건이면 충돌로 오인하지 마세요. 명시된 시행일 "
        "근거 없이 최신 규정을 추측하지 마세요. 해결 불가 충돌에만 clarification을 쓰세요.",
        {"question": question, "plan": plan.model_dump(), "evidence": evidence_payload(hits)},
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
    candidates = [
        {
            "chunk_id": str(i + 1),
            "doc_id": h.source.get("doc_id"),
            "heading": h.source.get("path") or h.source.get("title"),
            "excerpt": relevant_excerpt(str(h.source.get("body", "")), question, 550),
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
    sentences = [
        s.strip()
        for s in re.split(r"(?<=\])(?:[.!?。])?\s+|\n+", text)
        if re.search(r"[가-힣A-Za-z]", s)
    ]
    return {i + 1: sentence for i, sentence in enumerate(sentences)}


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
        "모든 requirements에 답했는지와 중요한 조건·예외 누락을 검사하세요. "
        "불확실한 표에서 등급을 추측한 답은 거부하세요. repair_instructions에 구체적인 "
        "수정 지시를 쓰되 외부 지식으로 답을 추가하지 마세요.",
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


def answer_instruction(question: str) -> str:
    language = "한국어" if detect_language(question) == "ko" else "English"
    return (
        f"{language}로만 답하세요. 제공한 원문에 직접 있는 사실만 쓰세요. "
        "질문의 모든 requirements에 답하되 불필요한 추가 설명은 하지 마세요. "
        "질문한 대상과 직접 관련된 사실만 쓰고 다른 대상·용도의 설명을 섞지 마세요. "
        "방법을 물으면 준비·실행·완료 순서로 간결하게 안내하세요. "
        "숫자·연도·단위·대상·조건·예외를 보존하세요. 원문에 없는 계산을 하지 마세요. "
        "각 문장/목록 항목 끝에 [cite:n]을 붙이세요. 표의 행·열을 혼동하지 마세요. "
        "수정 요청에서는 지적된 문장과 출처를 실제로 고치고 필요한 조건을 보존하세요. "
        "내용을 지어내는 대신 근거 부족을 명시하세요. text 필드에 완성된 답변을 쓰세요."
    )

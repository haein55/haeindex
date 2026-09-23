"""ask/why/웹이 공유하는 제한된 LLM 검토·복구 파이프라인."""

import re
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from haeindex.answer import Answer, Block, Context, Refusal, keep_only_known, language_matches
from haeindex.augmentation import section_candidates
from haeindex.bedrock import Bedrock
from haeindex.document_cards import search_cards, search_lexical_docs
from haeindex.index import INDEX
from haeindex.llm_tasks import TaskFailure, TaskRunner
from haeindex.profile import Profile
from haeindex.profiling import Profiler
from haeindex.reasoning import (
    AnswerPatch,
    AnswerReview,
    Draft,
    EvidenceCheck,
    OverviewDraft,
    QuestionPlan,
    answer_instruction,
    check_evidence,
    is_overview,
    plan_question,
    rank_evidence,
    repair_answer,
    review_answer,
)
from haeindex.routing import detect_language, route_question
from haeindex.search import Hit, Result, search
from haeindex.source_recovery import fetch_chunks, neighbors, recover_pages


class Stage(BaseModel):
    name: str
    detail: str
    chunk_ids: list[str] = Field(default_factory=list)
    at: float = 0
    seconds: float = 0


class PipelineTrace(BaseModel):
    question: str
    plan: QuestionPlan = Field(default_factory=QuestionPlan)
    result: Result
    answer: Answer = Field(default_factory=Answer)
    clarification: str = ""
    documents: list[str] = Field(default_factory=list)
    stages: list[Stage] = Field(default_factory=list)
    assessments: list[EvidenceCheck] = Field(default_factory=list)
    reviews: list[AnswerReview] = Field(default_factory=list)
    repairs: list[AnswerPatch] = Field(default_factory=list)
    calls: int = 0
    cache_hits: int = 0
    events: list[dict] = Field(default_factory=list)
    spans: list[dict] = Field(default_factory=list)
    error: str = ""


def merge_hits(*groups: list[Hit]) -> list[Hit]:
    out = {}
    for group in groups:
        for hit in group:
            out.setdefault(hit.chunk_id, hit)
    return list(out.values())


def normalize_citations(text: str) -> str:
    # 숫자 인용의 단순 표기 차이만 정규화한다.
    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", r"[cite:\1]", text)


def make_context(hits: list[Hit], max_chars: int = 11000) -> Context:
    blocks = []
    used = 0
    for h in hits:
        body = str(h.source.get("body", ""))
        if not body.strip() or used + len(body) > max_chars:
            continue
        used += len(body)
        blocks.append(
            Block(
                n=len(blocks) + 1,
                doc_id=str(h.source.get("doc_id", "")),
                page=int(h.source.get("page", 0)),
                label=str(h.source.get("path") or h.source.get("title") or ""),
                text=body,
                chunk_id=h.chunk_id,
                origin=str(h.source.get("evidence_origin", "indexed_pdf")),
            )
        )
    return Context(blocks=blocks, dropped=len(hits) - len(blocks), est_tokens=int(used / 1.6))


def run_pipeline(
    os_client,
    answer_llm: Bedrock,
    analysis_llm: Bedrock,
    question: str,
    *,
    known_docs: list[str],
    explicit_docs: list[str] | None = None,
    index: str = INDEX,
    top_k: int = 8,
    max_calls: int = 18,
    max_retries: int = 1,
    vision: Bedrock | None = None,
    cache: Path | None = Path("work/llm-cache"),
    progress: Callable[[str, str], None] | None = None,
) -> PipelineTrace:
    runner = TaskRunner(max_calls=max_calls, cache=cache)
    trace = PipelineTrace(question=question, result=Result(query=question))
    profiler = Profiler()
    profiler_token = profiler.activate()
    pipeline_started = stage_started = time.monotonic()

    def stage(name: str, detail: str, hits: list[Hit] | None = None) -> None:
        nonlocal stage_started
        now = time.monotonic()
        if trace.stages:
            trace.stages[-1].seconds = round(now - stage_started, 3)
        trace.stages.append(
            Stage(
                name=name,
                detail=detail,
                chunk_ids=[h.chunk_id for h in hits or []],
                at=round(now - pipeline_started, 6),
            )
        )
        profiler.stage = name
        stage_started = now
        if progress:
            progress(name, detail)

    try:
        stage("planning", "질문의 요구사항과 조건을 정리합니다")
        route = route_question(question, known_docs, explicit_doc_ids=explicit_docs or [])
        if route.clarification:
            trace.documents = route.doc_ids
            trace.clarification = route.clarification
            stage("clarifying", route.clarification)
            return trace
        plan = plan_question(runner, analysis_llm, question, known_docs)
        trace.plan = plan
        multi = len(explicit_docs or []) > 1 or route.allow_multiple_docs
        wanted = route.doc_ids
        if (
            not wanted
            and plan.ambiguous
            and re.search(r"이\s*(파일|문서)|this\s+(file|document)", question, re.I)
        ):
            trace.clarification = (
                plan.clarification or "어느 문서에서 찾을까요? 문서를 선택해 주세요."
            )
            return trace
        stage("routing", "관련 문서와 절을 찾습니다")
        if not wanted:
            cards = search_cards(os_client, question, embedder=analysis_llm, top_k=3)
            lexical = search_lexical_docs(os_client, question)
            wanted = list(dict.fromkeys([*lexical, *(c.doc_id for c in cards)]))
        trace.documents = wanted
        if (
            plan.ambiguous
            and not explicit_docs
            and len(wanted) > 1
            and re.fullmatch(r"\s*제?\s*\d+\s*조\s*[?？]?\s*", question)
        ):
            trace.clarification = "어느 문서의 조항을 찾을까요? " + ", ".join(wanted[:3])
            return trace
        stage("retrieving", "원문·문맥 벡터와 키워드로 후보를 검색합니다")
        base = search(
            os_client,
            question,
            embedder=analysis_llm,
            doc_ids=wanted,
            top_k=20,
            candidate_k=50,
            index=index,
            contextual=True,
            extra_queries=plan.queries[1:3],
        )
        trace.result = base
        if base.degraded:
            trace.answer = Answer(refusal=Refusal.SEARCH_DEGRADED)
            return trace
        stage("retrieved", f"검색 후보 {len(base.hits)}개", base.hits)
        sections = fetch_chunks(
            os_client,
            section_candidates(os_client, question, wanted, embedder=analysis_llm),
            wanted,
            index,
        )
        pool = merge_hits(base.hits[:14], sections[:6], base.hits[14:])
        if not pool:
            trace.answer = Answer(refusal=Refusal.NO_HITS)
            return trace
        stage("ranking", "조건·예외가 있는 원문을 우선 배치합니다")
        ranked = rank_evidence(runner, analysis_llm, question, plan, pool[:20])
        stage("ranked", "재정렬 완료", ranked)
        if not multi:
            selected_doc = ranked[0].source["doc_id"]
            ranked = [h for h in ranked if h.source.get("doc_id") == selected_doc]
            # 재검색으로 명시한 문서 범위를 넘어가지 않는다.
            wanted = [selected_doc]
        trace.documents = wanted
        selected = ranked[:top_k]
        adjacent = neighbors(os_client, selected[:3], index)
        selected = merge_hits(selected, adjacent)[:12]
        # 이중 문자 보정 이력이 있으면 숫자 손상을 막기 위해 원본 좌표에서 다시 읽는다.
        damaged = []
        for h in selected:
            try:
                if Profile.load(h.source["doc_id"]).undouble:
                    damaged.append(h)
            except (FileNotFoundError, ValueError):
                pass
        if damaged:
            stage("reading_text", "중복 문자 보정 문서의 원본 PDF를 다시 읽습니다")
            original = recover_pages(damaged[:3])
            pages = {(h.source["doc_id"], h.source["page"]) for h in original}
            remaining = [
                h
                for h in selected
                if (h.source["doc_id"], h.source["page"]) not in pages
                or h.source.get("end_page", h.source["page"]) > h.source["page"]
            ]
            selected = merge_hits(original, remaining)[:12]
        stage("evidence", "원문 근거와 이어지는 조건을 확인합니다", selected)
        check = check_evidence(runner, analysis_llm, question, plan, selected)
        trace.assessments.append(check)
        for attempt in range(max_retries):
            if check.complete(plan.requirements) or check.clarification:
                break
            stage("recovering", f"부족한 근거를 보완합니다 ({attempt + 1}/{max_retries})")
            queries = list(dict.fromkeys([*check.queries, *plan.queries]))[:2]
            additional = []
            for query in queries:
                res = search(
                    os_client,
                    query,
                    embedder=analysis_llm,
                    doc_ids=wanted,
                    top_k=15,
                    index=index,
                    contextual=True,
                )
                if res.degraded:
                    raise TaskFailure("보완 검색의 벡터 경로에 문제가 있습니다")
                additional = merge_hits(additional, res.hits)
            originals = recover_pages(merge_hits(selected, additional), max_pages=5)
            pool = merge_hits(originals, selected, additional)
            ranked = rank_evidence(runner, analysis_llm, question, plan, pool[:20])
            selected = ranked[: max(top_k, 10)]
            check = check_evidence(runner, analysis_llm, question, plan, selected)
            trace.assessments.append(check)
        by_visual_id = {h.chunk_id: h for h in selected}
        visual_candidates = [
            by_visual_id[cid]
            for cid in check.visual_chunk_ids()
            if cid in by_visual_id
            and by_visual_id[cid].source.get("evidence_origin") != "vision_transcription"
        ]
        if (
            not check.complete(plan.requirements)
            and not check.clarification
            and vision is not None
            and visual_candidates
        ):
            stage("reading_pdf", "표·도식·문자 손상이 확인된 근거를 이미지로 다시 확인합니다")
            recovered = recover_pages(
                visual_candidates[:4],
                runner=runner,
                vision=vision,
                max_pages=2,
                use_vision=True,
                question=question,
            )
            if recovered:
                selected = merge_hits(recovered, selected)[:12]
                check = check_evidence(runner, analysis_llm, question, plan, selected)
                trace.assessments.append(check)
        trace.result = base.model_copy(update={"hits": selected})
        if check.clarification:
            trace.clarification = check.clarification
            return trace
        if not check.complete(plan.requirements):
            trace.answer = Answer(
                refusal=Refusal.NOT_FOUND if check.not_found() else Refusal.INSUFFICIENT_EVIDENCE,
                context=make_context(selected),
            )
            stage(
                "not_found" if check.not_found() else "insufficient",
                "보완 검색에서도 질문에 해당하는 내용을 찾지 못했습니다"
                if check.not_found()
                else "질문에 답할 원문 근거를 충분히 확인하지 못했습니다",
            )
            return trace
        supported_ids = list(dict.fromkeys(q.chunk_id for s in check.supports for q in s.sources))
        by_id = {h.chunk_id: h for h in selected}
        selected = merge_hits([by_id[i] for i in supported_ids if i in by_id], selected)[:10]
        context = make_context(selected)
        selected = [by_id[b.chunk_id] for b in context.blocks]
        citation_ids = {b.n: b.chunk_id for b in context.blocks}
        payload = {
            "question": question,
            "plan": plan.model_dump(),
            "evidence": [
                {
                    "n": b.n,
                    "chunk_id": b.chunk_id,
                    "doc_id": b.doc_id,
                    "page": b.page,
                    "body": b.text,
                }
                for b in context.blocks
            ],
            "requirements_support": [
                {"requirement": s.requirement, "chunks": [q.chunk_id for q in s.sources]}
                for s in check.supports
            ],
        }
        stage("answering", "확인한 원문으로 답변을 작성합니다")
        text = ""
        for repair in range(2):
            if repair and trace.reviews:
                stage("repairing", "검토에서 지적된 문장만 한 번 수정합니다")
                text, patch = repair_answer(
                    runner, answer_llm, question, plan, text, trace.reviews[-1], payload
                )
                trace.repairs.append(patch)
            else:
                if repair:
                    # 문장 검토에 도달하지 못했으므로 아직 보존할 승인 문장이 없다.
                    payload["previous_answer"] = text
                    payload["review"] = {"issue": "언어 또는 인용 형식이 맞지 않습니다"}
                    stage("repairing", "답변 언어 또는 인용 형식을 한 번 수정합니다")
                draft = runner.run(
                    answer_llm,
                    "answer-repair" if repair else "answer-draft",
                    answer_instruction(question, plan),
                    payload,
                    OverviewDraft if is_overview(plan) else Draft,
                    num_predict=1800,
                    max_predict=3600,
                )
                text = draft.text
            text, cited = keep_only_known(normalize_citations(text), set(citation_ids))
            if not cited or not language_matches(text, detect_language(question)):
                continue
            stage("verifying", "문장별 주장·수치·조건과 인용 원문을 대조합니다")
            review = review_answer(
                runner, analysis_llm, question, plan, text, selected, citation_ids
            )
            trace.reviews.append(review)
            if review.accepted():
                if is_overview(plan):
                    text += (
                        "\n\n주요 규칙 요약입니다. "
                        "필요한 항목을 지정하면 세부 조건을 확인할 수 있습니다."
                        if detect_language(question) == "ko"
                        else "\n\nThis summarizes the main rules. "
                        "Ask about a specific item for details."
                    )
                trace.answer = Answer(
                    text=text,
                    cited=sorted(cited),
                    context=context,
                    language=detect_language(question),
                    citation_retries=repair,
                )
                trace.result = base.model_copy(update={"hits": selected})
                stage("complete", "근거 대조를 마쳤습니다", selected)
                return trace
        trace.answer = Answer(refusal=Refusal.UNGROUNDED_CLAIMS, context=context)
        stage("unverified", "원문과 일치하는 답변을 확인하지 못했습니다")
    except TaskFailure as exc:
        trace.error = str(exc)
        trace.answer = Answer(refusal=Refusal.MODEL_FAILURE)
        stage("error", str(exc))
    finally:
        if trace.stages:
            trace.stages[-1].seconds = round(time.monotonic() - stage_started, 3)
        trace.calls = runner.calls
        trace.cache_hits = sum(e.cache_hit for e in runner.events)
        trace.events = [e.model_dump() for e in runner.events]
        trace.spans = profiler.spans
        profiler.deactivate(profiler_token)
    return trace

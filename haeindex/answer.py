import enum
import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from haeindex.bedrock import Bedrock
from haeindex.routing import detect_language
from haeindex.search import Result

MIN_COS_TOP1 = 0.78
MAX_CONTEXT_CHARS = 6000
CITE = re.compile(r"\[cite:\s*([\d\s,]+)\]")
SENTENCE_BREAK = re.compile(r"(?<=\])\s+|(?<=[.!?。])\s+|\n+")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CHARS_PER_TOKEN = 1.6
MAX_LANGUAGE_RETRIES = 2

SYSTEM = """당신은 문서 질의응답 시스템입니다. 아래 규칙을 반드시 지키세요.

① 제공된 문서 조각만 사용하세요. 사전 지식을 쓰지 마세요.
② 조각에 답이 없으면 "문서에서 확인할 수 없습니다" 라고만 답하세요.
   비슷한 다른 조각으로 대신 답하지 마세요.
③ 모든 문장 끝에 근거 번호를 [cite:n] 형식으로 붙이세요.
④ 숫자와 규격은 문서 값을 그대로 옮기세요. 계산·반올림 금지.
⑤ 3문장 이내로 답하되 조건·예외는 반드시 포함하세요.
⑥ **질문과 같은 언어로 답하세요.** 한국어 질문에는 한국어로, 영어 질문에는 영어로."""


class Refusal(enum.StrEnum):
    NO_HITS = "no_hits"
    LOW_CONFIDENCE = "low_confidence"
    NO_CITATIONS = "no_citations"
    UNGROUNDED_CLAIMS = "ungrounded_claims"
    LANGUAGE_MISMATCH = "language_mismatch"
    SEARCH_DEGRADED = "search_degraded"
    MODEL_FAILURE = "model_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_FOUND = "not_found"


MESSAGES = {
    Refusal.MODEL_FAILURE: "모델 처리 중 문제가 생겼습니다. 잠시 후 다시 시도해 주세요.",
    Refusal.INSUFFICIENT_EVIDENCE: "질문에 답할 원문 근거를 충분히 확인하지 못했습니다.",
    Refusal.NOT_FOUND: "검색한 문서에서 질문에 해당하는 내용을 찾지 못했습니다.",
    Refusal.NO_HITS: "문서에서 확인할 수 없습니다. (검색 결과 없음)",
    Refusal.LOW_CONFIDENCE: "문서에서 확인할 수 없습니다. (관련도가 임계값 미만)",
    Refusal.NO_CITATIONS: "⚠ 인용 없음 — 근거 없이 생성된 답변으로 판단해 폐기했습니다.",
    Refusal.UNGROUNDED_CLAIMS: "답변과 출처의 일치를 확인하지 못해 답변을 보류했습니다.",
    Refusal.LANGUAGE_MISMATCH: "질문과 같은 언어로 답변을 생성하지 못했습니다.",
    Refusal.SEARCH_DEGRADED: "검색이 온전하지 않아 답변하지 않습니다. 문서에 없다는 뜻이 아닙니다.",
}


def should_refuse(result: Result, min_cos: float = MIN_COS_TOP1) -> Refusal | None:
    if not result.hits:
        return Refusal.NO_HITS
    if any(d.startswith("knn") for d in result.degraded):
        return Refusal.SEARCH_DEGRADED
    return None if result.top_score("knn") >= min_cos else Refusal.LOW_CONFIDENCE


class Block(BaseModel):
    model_config = ConfigDict(frozen=True)

    n: int
    doc_id: str
    page: int
    label: str
    text: str
    chunk_id: str = ""
    origin: str = "indexed_pdf"


class Context(BaseModel):
    model_config = ConfigDict(frozen=True)

    blocks: list[Block] = Field(default_factory=list)
    dropped: int = 0
    est_tokens: int = 0

    def render(self) -> str:
        return "\n\n".join(
            f"[{b.n}] ({b.doc_id} · p.{b.page} · {b.label})\n{b.text}" for b in self.blocks
        )


def est_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def assemble(result: Result, max_chars: int = MAX_CONTEXT_CHARS) -> Context:
    blocks: list[Block] = []
    used = 0
    dropped = 0
    for i, hit in enumerate(result.hits, 1):
        src = hit.source
        body = str(src.get("body", ""))
        if used + len(body) > max_chars:
            dropped = len(result.hits) - i + 1
            break
        blocks.append(
            Block(
                n=i,
                doc_id=str(src.get("doc_id", "")),
                page=int(src.get("page", 0)),
                label=str(src.get("path") or src.get("title") or ""),
                text=body,
            )
        )
        used += len(body)
    return Context(blocks=blocks, dropped=dropped, est_tokens=est_tokens("x" * used))


def keep_only_known(answer: str, valid: set[int]) -> tuple[str, set[int]]:
    used: set[int] = set()

    def repl(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
        keep = [n for n in nums if n in valid]
        used.update(keep)
        return f"[cite:{','.join(map(str, keep))}]" if keep else ""

    return (CITE.sub(repl, answer), used)


def language_matches(text: str, language: str) -> bool:
    clean = CITE.sub("", text)
    if HAN.search(clean):
        return False
    ko = sum("가" <= c <= "힣" for c in clean)
    en = sum(c.isascii() and c.isalpha() for c in clean)
    if language == "ko":
        return ko >= 3 and ko >= en * 0.2
    if language == "en":
        return en >= 3 and en >= ko
    return True


def claims_have_citations(text: str) -> bool:
    """빈 줄·출처 표기 외의 각 문장이 적어도 하나의 유효 인용을 가져야 한다."""
    attached = re.sub(r"([.!?。])\s*(\[cite:[^]]+\])", r"\1\2", text)
    claims = [part.strip() for part in SENTENCE_BREAK.split(attached) if part.strip()]
    return bool(claims) and all(CITE.search(claim) for claim in claims)


def scope_single_citation(text: str, cited: set[int]) -> str:
    """근거가 하나면 문장마다 반복하지 않고 답변 전체의 출처로 한 번 표시한다."""
    if len(cited) != 1:
        return text
    number = next(iter(cited))
    clean = CITE.sub("", text).strip()
    clean = re.sub(rf"^\[{number}\]\s*", "", clean)
    return f"{clean} [cite:{number}]"


class Answer(BaseModel):
    text: str = ""
    refusal: Refusal | None = None
    cited: list[int] = Field(default_factory=list)
    context: Context = Context()
    language: str = ""
    language_retries: int = 0
    citation_retries: int = 0
    citations_complete: bool = True


def answer(
    llm: Bedrock,
    result: Result,
    question: str,
    *,
    min_cos: float = MIN_COS_TOP1,
    max_chars: int = MAX_CONTEXT_CHARS,
    strict: bool = True,
) -> Answer:
    refusal = should_refuse(result, min_cos)
    if refusal is not None:
        return Answer(refusal=refusal)

    ctx = assemble(result, max_chars)
    prompt = f"문서 조각:\n\n{ctx.render()}\n\n질문: {question}"
    need = est_tokens(SYSTEM + prompt)
    if llm.num_ctx < need * 1.2:
        raise ValueError(
            f"num_ctx {llm.num_ctx} 가 프롬프트 추정 {need} 토큰 × 1.2 보다 작다. "
            "컨텍스트 초과는 '검색이 나쁘다'처럼 보일 수 있으므로 요청 전에 차단한다"
        )

    raw = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    ).content
    language = detect_language(question)
    retries = 0
    requested = "한국어" if language == "ko" else "English"
    while not language_matches(raw, language) and retries < MAX_LANGUAGE_RETRIES:
        if language == "ko":
            language_rule = (
                "한글과 필요한 영문 고유명사만 사용하세요. "
                "중국어 문장과 한자(CJK ideographs)는 절대 출력하지 마세요."
            )
        else:
            language_rule = "Use English only. Do not output Korean, Chinese, or CJK ideographs."
        raw = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"{SYSTEM}\n\n출력 언어 추가 규칙: {requested}로만 답하세요. "
                        f"{language_rule}"
                    ),
                },
                {"role": "user", "content": prompt},
            ]
        ).content
        retries += 1
    if not language_matches(raw, language):
        return Answer(
            refusal=Refusal.LANGUAGE_MISMATCH,
            context=ctx,
            language=language,
            language_retries=retries,
        )
    text, cited = keep_only_known(raw, {b.n for b in ctx.blocks})
    if strict and not cited:
        return Answer(
            refusal=Refusal.NO_CITATIONS,
            context=ctx,
            language=language,
            language_retries=retries,
        )
    citation_retries = 0
    citations_complete = claims_have_citations(text)
    if strict and cited and not citations_complete and len(cited) == 1:
        text = scope_single_citation(text, cited)
        citations_complete = True
    if strict and cited and not citations_complete:
        correction = llm.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "내용을 추가하거나 빼지 말고 인용 위치만 고치세요. "
                        "각 문장과 목록 항목의 끝에 그 내용을 뒷받침하는 [cite:n]을 붙이세요. "
                        "예: 수당을 지급합니다. [cite:1]"
                    ),
                },
            ]
        ).content
        citation_retries = 1
        corrected, corrected_cited = keep_only_known(correction, {b.n for b in ctx.blocks})
        if language_matches(correction, language) and corrected_cited:
            text, cited = corrected, corrected_cited
            citations_complete = claims_have_citations(text)
    return Answer(
        text=text.strip(),
        cited=sorted(cited),
        context=ctx,
        language=language,
        language_retries=retries,
        citation_retries=citation_retries,
        citations_complete=citations_complete,
    )


def cos_top1(results: Sequence[Result]) -> list[float]:
    return [r.top_score("knn") for r in results]

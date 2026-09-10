import enum
import math
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from haeindex.ollama import Ollama
from haeindex.search import Result

MIN_COS_TOP1 = 0.78
MAX_CONTEXT_CHARS = 6000
CITE = re.compile(r"\[cite:\s*([\d\s,]+)\]")
CHARS_PER_TOKEN = 1.6

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
    SEARCH_DEGRADED = "search_degraded"


MESSAGES = {
    Refusal.NO_HITS: "문서에서 확인할 수 없습니다. (검색 결과 없음)",
    Refusal.LOW_CONFIDENCE: "문서에서 확인할 수 없습니다. (관련도가 임계값 미만)",
    Refusal.NO_CITATIONS: "⚠ 인용 없음 — 근거 없이 생성된 답변으로 판단해 폐기했습니다.",
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


class Answer(BaseModel):
    text: str = ""
    refusal: Refusal | None = None
    cited: list[int] = Field(default_factory=list)
    context: Context = Context()


def answer(
    llm: Ollama,
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
            "Ollama 는 초과분을 경고 없이 버리고, 그건 '검색이 나쁘다'처럼 보인다"
        )

    raw = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    ).content
    text, cited = keep_only_known(raw, {b.n for b in ctx.blocks})
    if strict and not cited:
        return Answer(refusal=Refusal.NO_CITATIONS, context=ctx)
    return Answer(text=text.strip(), cited=sorted(cited), context=ctx)


def cos_top1(results: Sequence[Result]) -> list[float]:
    return [r.top_score("knn") for r in results]

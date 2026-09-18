import json
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from haeindex.answer import est_tokens
from haeindex.bedrock import Bedrock, Truncated
from haeindex.search import Hit

DEPTH = 20
WINDOW = 20
STEP = 10
PASSAGE_CHARS = 200
NUM_PREDICT = 200
CTX_SAFETY = 1.2

SYSTEM = """당신은 문서 검색 결과를 다시 정렬하는 채점자입니다.

질문에 **직접 답하는 내용이 적혀 있는** 조각을 위로 올리세요.
- 주제가 비슷한 것보다 답이 들어 있는 것이 먼저입니다.
- 제목만 맞고 본문에 답이 없으면 내리세요.
- 새 조각을 만들지 말고 주어진 번호만 재배열하세요.

번호만 JSON 배열로 출력하세요. 설명 금지. 예: [3, 1, 7, 2]"""

_ARRAY = re.compile(r"\[[\d\s,]*\]")


class Ranked(BaseModel):
    model_config = ConfigDict(frozen=True)

    hits: list[Hit] = Field(default_factory=list)
    calls: int = 0
    parse_fails: int = 0
    moved: int = 0


def passage(hit: Hit, n: int, chars: int = PASSAGE_CHARS) -> str:
    s = hit.source
    label = str(s.get("path") or s.get("title") or "")
    body = " ".join(str(s.get("body", "")).split())[:chars]
    return f"[{n}] ({s.get('doc_id', '')} · p.{s.get('page', 0)} · {label})\n{body}"


def parse_order(raw: str, n: int) -> list[int] | None:
    """부분 순열도 받는다 — 전부 나열하라고 요구하면 파싱 실패율이 올라간다."""
    m = _ARRAY.search(raw)
    if not m:
        return None
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    seen: list[int] = []
    for x in items:
        if isinstance(x, int) and 1 <= x <= n and x not in seen:
            seen.append(x)
    if not seen:
        return None
    return [i - 1 for i in seen] + [i for i in range(n) if i + 1 not in seen]


def rank_window(llm: Bedrock, query: str, hits: Sequence[Hit]) -> tuple[list[Hit], bool]:
    """한 창을 재정렬한다. 실패하면 원래 순서를 그대로 돌려준다."""
    listing = "\n\n".join(passage(h, i + 1) for i, h in enumerate(hits))
    prompt = f"문서 조각:\n\n{listing}\n\n질문: {query}\n\n답이 있는 순서대로 번호를 출력하세요."
    need = est_tokens(SYSTEM + prompt)
    if llm.num_ctx < need * CTX_SAFETY:
        raise ValueError(
            f"num_ctx {llm.num_ctx} 가 프롬프트 추정 {need} 토큰 × {CTX_SAFETY} 보다 작다. "
            "컨텍스트 초과는 '재정렬이 나쁘다'처럼 보일 수 있어 요청 전에 차단한다"
        )
    try:
        raw = llm.chat(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            num_predict=NUM_PREDICT,
        ).content
    except Truncated:
        return (list(hits), False)
    order = parse_order(raw, len(hits))
    if order is None:
        return (list(hits), False)
    return ([hits[i] for i in order], True)


def windows(n: int, window: int, step: int) -> list[tuple[int, int]]:
    """뒤에서 앞으로 민다 — 올라온 조각이 다음 창에 다시 들어가야 맨 위까지 갈 수 있다."""
    if n <= window:
        return [(0, n)]
    out: list[tuple[int, int]] = []
    end = n
    while end > window:
        out.append((end - window, end))
        end -= step
    out.append((0, window))
    return out


def rerank(
    llm: Bedrock,
    query: str,
    hits: Sequence[Hit],
    *,
    depth: int = DEPTH,
    window: int = WINDOW,
    step: int = STEP,
) -> Ranked:
    """후보 앞 depth 개만 LLM 으로 재정렬한다. 나머지는 융합 순서를 유지한다."""
    head = list(hits[:depth])
    tail = list(hits[depth:])
    if len(head) < 2:
        return Ranked(hits=list(hits))

    calls = fails = 0
    for lo, hi in windows(len(head), window, step):
        ordered, ok = rank_window(llm, query, head[lo:hi])
        head[lo:hi] = ordered
        calls += 1
        fails += 0 if ok else 1

    before = [h.chunk_id for h in hits[:depth]]
    after = [h.chunk_id for h in head]
    return Ranked(
        hits=head + tail,
        calls=calls,
        parse_fails=fails,
        moved=sum(1 for a, b in zip(before, after, strict=True) if a != b),
    )

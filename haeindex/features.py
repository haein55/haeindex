import json
import re
from collections.abc import Sequence

from haeindex.ollama import Ollama, Truncated

MAX_QUERIES = 5
BODY_LIMIT = 1500
NUM_PREDICT = 400

PROMPT = """당신은 문서 검색 시스템을 만드는 사람입니다.
아래 문서 조각을 읽고, **실제 사용자가 이 내용을 찾을 때 입력할 만한 질문**을
3~5개 만드세요.

규칙:
- 문서는 공식·전문 용어를 쓰지만 사용자는 **일상 표현**을 씁니다.
  (문서 "출산전후휴가" → 사용자 "애 낳고 쉬는 기간")
- 조각에 답이 있는 질문만 만드세요. 없는 사실을 넣지 마세요.
- 각 질문은 한 문장. 문서와 같은 언어로.
- JSON 배열만 출력하세요.

문서 조각:
---
{body}
---"""

_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


def gen_queries(llm: Ollama, body: str) -> list[str]:
    try:
        raw = llm.chat(
            [{"role": "user", "content": PROMPT.format(body=body[:BODY_LIMIT])}],
            num_predict=NUM_PREDICT,
        ).content
    except Truncated:
        return []
    m = _ARRAY.search(raw)
    if not m:
        return []
    try:
        items = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    out: list[str] = []
    for x in items:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out[:MAX_QUERIES]


def gen_many(
    llm: Ollama, bodies: Sequence[str], on_step: object = None
) -> tuple[list[list[str]], int]:
    out: list[list[str]] = []
    failed = 0
    for i, body in enumerate(bodies):
        got = gen_queries(llm, body)
        if not got:
            failed += 1
        out.append(got)
        if callable(on_step):
            on_step(i + 1, len(bodies))
    return (out, failed)

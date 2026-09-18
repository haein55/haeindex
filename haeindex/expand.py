import json
import re
from collections.abc import Sequence

from haeindex.bedrock import Bedrock

MAX_REWRITES = 2

PROMPT = """당신은 문서 검색 시스템의 질의 재작성기입니다.

사용자는 일상 표현으로 묻고, 문서는 공식·전문 용어로 쓰여 있습니다.
아래 질의를 **문서에 실제로 쓰였을 표현**으로 다시 쓰세요.

예:
  "애 낳고 쉬는 기간" → "출산전후휴가 기간"
  "야근하면 돈 더 주나요" → "연장근로 가산수당"
  "소리가 안 들려요" → "오디오 출력 이상"

규칙:
- 명사 위주의 짧은 구로 쓰세요. 문장으로 쓰지 마세요.
- 새로운 사실을 넣지 마세요. 표현만 바꾸세요.
- 최대 2개. JSON 배열만 출력하세요.

질의: {query}"""

_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


def rewrite(llm: Bedrock, query: str) -> list[str]:
    raw = llm.chat(
        [{"role": "user", "content": PROMPT.format(query=query)}], num_predict=120
    ).content
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
        if s and s.lower() != query.lower() and s not in out:
            out.append(s)
    return out[:MAX_REWRITES]


def rewrite_many(llm: Bedrock, queries: Sequence[str]) -> dict[str, list[str]]:
    return {q: rewrite(llm, q) for q in queries}

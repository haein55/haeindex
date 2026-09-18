import json
import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from haeindex.bedrock import Bedrock, Truncated

BODY_LIMIT = 1800
NUM_PREDICT = 400
_JSON = re.compile(r"\{.*\}", re.DOTALL)

PROMPT = """문서 검색용 메타데이터를 만드세요. 원문에 없는 사실을 추측하지 마세요.
반드시 다음 JSON 객체 하나만 출력하세요.
{{
  "summary": "원문의 핵심을 한 문장으로 요약",
  "keywords": ["원문 검색어"],
  "entities": ["조항·제품·기관·고유명사"],
  "content_type": "definition|procedure|requirement|table|fact|other",
  "language": "ko|en"
}}

원문:
---
{body}
---"""


class Enrichment(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    content_type: str = "other"
    language: str = ""


def parse_enrichment(raw: str) -> Enrichment | None:
    match = _JSON.search(raw)
    if not match:
        return None
    try:
        item = Enrichment.model_validate(json.loads(match.group()))
    except (json.JSONDecodeError, ValidationError):
        return None
    allowed = {"definition", "procedure", "requirement", "table", "fact", "other"}
    if item.content_type not in allowed or item.language not in {"ko", "en"}:
        return None
    return item


def enrich(llm: Bedrock, body: str) -> Enrichment | None:
    try:
        raw = llm.chat(
            [{"role": "user", "content": PROMPT.format(body=body[:BODY_LIMIT])}],
            num_predict=NUM_PREDICT,
        ).content
    except Truncated:
        return None
    return parse_enrichment(raw)


def enrich_many(
    llm: Bedrock, bodies: Sequence[str], on_step: object = None
) -> tuple[list[Enrichment | None], int]:
    out: list[Enrichment | None] = []
    failed = 0
    for i, body in enumerate(bodies):
        item = enrich(llm, body)
        failed += item is None
        out.append(item)
        if callable(on_step):
            on_step(i + 1, len(bodies))
    return out, failed

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from haeindex.bedrock import EMBED_DIM, Bedrock, Truncated
from haeindex.chunks import Chunk
from haeindex.index import INDEX, settings

DOC_INDEX = "haeindex_bedrock_docs_v1"
DOC_CANDIDATES = 3
LEXICAL_DOC_CANDIDATES = 2
SAMPLE_CHUNKS = 20
SAMPLE_BODY_CHARS = 260
INPUT_LIMIT = 7000
NUM_PREDICT = 600
RRF_K = 60
_JSON = re.compile(r"\{.*\}", re.DOTALL)

PROMPT = """문서 검색 라우팅에 사용할 문서 카드를 만드세요.
제공된 파일명·목차·원문 표본에 있는 정보만 사용하고 추측하지 마세요.
반드시 다음 JSON 객체 하나만 출력하세요.
{{
  "summary": "문서 전체의 목적과 내용을 2~3문장으로 요약",
  "topics": ["사용자가 찾을 핵심 주제"],
  "entities": ["제품·기관·법령·문서명·주요 고유명사"],
  "aliases": ["사용자가 이 문서를 부를 만한 짧은 이름"],
  "document_type": "manual|policy|law|guide|paper|certificate|budget|other",
  "language": "ko|en|mixed"
}}

문서 ID: {doc_id}
파일명: {source}

목차와 원문 표본:
---
{content}
---"""


class DocumentCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str
    summary: str
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    document_type: str = "other"
    language: str = "mixed"

    def search_text(self) -> str:
        return "\n".join(
            [
                self.source,
                self.doc_id,
                *self.aliases,
                self.summary,
                *self.topics,
                *self.entities,
                self.document_type,
            ]
        )


class CardHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    fused: float
    source: dict[str, Any]
    legs: dict[str, int] = Field(default_factory=dict)


def _text_field() -> dict[str, str]:
    return {"type": "text", "analyzer": "ko_index", "search_analyzer": "ko_search"}


def card_mappings(dim: int = EMBED_DIM) -> dict[str, Any]:
    return {
        "properties": {
            "doc_id": {"type": "keyword"},
            "source": _text_field(),
            "summary": _text_field(),
            "topics": _text_field(),
            "entities": _text_field(),
            "aliases": _text_field(),
            "document_type": {"type": "keyword"},
            "language": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": dim,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 128, "m": 16},
                },
            },
        }
    }


def ensure_doc_index(
    os_client: OpenSearch, name: str = DOC_INDEX, dim: int = EMBED_DIM
) -> None:
    if not os_client.indices.exists(index=name):
        os_client.indices.create(
            index=name,
            body={"settings": settings(), "mappings": card_mappings(dim)},
        )
        return
    live = os_client.indices.get_mapping(index=name)[name]["mappings"].get("properties", {})
    missing = {k: v for k, v in card_mappings(dim)["properties"].items() if k not in live}
    if missing:
        os_client.indices.put_mapping(index=name, body={"properties": missing})


def sampled_content(chunks: Sequence[Chunk]) -> str:
    if not chunks:
        return ""
    count = min(SAMPLE_CHUNKS, len(chunks))
    if count == 1:
        indexes = [0]
    else:
        indexes = sorted(
            {round(i * (len(chunks) - 1) / (count - 1)) for i in range(count)}
        )
    parts: list[str] = []
    for i in indexes:
        chunk = chunks[i]
        label = chunk.path or chunk.title
        body = " ".join(chunk.body.split())[:SAMPLE_BODY_CHARS]
        parts.append(f"- p.{chunk.page} {label}\n  {body}")
    return "\n".join(parts)[:INPUT_LIMIT]


def normalize_document_type(generated: object, source: str) -> str:
    low = source.lower()
    hints = [
        (("manual", "매뉴얼"), "manual"),
        (("입찰", "안내서"), "guide"),
        (("survey", "논문", "paper"), "paper"),
        (("취업규칙", "사규"), "policy"),
        (("법", "시행령", "별표"), "law"),
        (("수료증", "certificate"), "certificate"),
        (("예산", "budget"), "budget"),
    ]
    for needles, kind in hints:
        if any(needle in low for needle in needles):
            return kind
    allowed = {"manual", "policy", "law", "guide", "paper", "certificate", "budget", "other"}
    value = str(generated or "other").lower()
    return value if value in allowed else "other"


def infer_source_language(content: str) -> str:
    ko = sum("가" <= char <= "힣" for char in content)
    en = sum(char.isascii() and char.isalpha() for char in content)
    if ko == 0:
        return "en" if en else "mixed"
    if en == 0 or ko >= en * 0.2:
        return "ko"
    if en >= ko * 5:
        return "en"
    return "mixed"


def parse_card(raw: str, *, doc_id: str, source: str) -> DocumentCard | None:
    match = _JSON.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    data["doc_id"] = doc_id
    data["source"] = source
    data["document_type"] = normalize_document_type(data.get("document_type"), source)
    if str(data.get("language", "mixed")) not in {"ko", "en", "mixed"}:
        data["language"] = "mixed"
    try:
        card = DocumentCard.model_validate(data)
    except ValidationError:
        return None
    deterministic = [source, Path(source).stem, doc_id]
    aliases = list(dict.fromkeys(x.strip() for x in [*deterministic, *card.aliases] if x.strip()))
    return card.model_copy(update={"aliases": aliases[:20]})


def generate_card(
    llm: Bedrock, doc_id: str, source: str, chunks: Sequence[Chunk]
) -> DocumentCard | None:
    content = sampled_content(chunks)
    prompt = PROMPT.format(doc_id=doc_id, source=source, content=content)
    for seed in (0, 1):
        try:
            raw = llm.chat(
                [{"role": "user", "content": prompt}],
                num_predict=NUM_PREDICT,
                seed=seed,
            ).content
        except Truncated:
            continue
        card = parse_card(raw, doc_id=doc_id, source=source)
        if card is not None:
            return card.model_copy(update={"language": infer_source_language(content)})
    return None


def index_card(
    os_client: OpenSearch,
    card: DocumentCard,
    embedding: Sequence[float],
    *,
    name: str = DOC_INDEX,
) -> None:
    source = card.model_dump()
    source["embedding"] = list(embedding)
    os_client.index(index=name, id=card.doc_id, body=source, refresh=True)


def card_bm25_body(query: str, size: int) -> dict[str, Any]:
    return {
        "size": size,
        "_source": {"excludes": ["embedding"]},
        "query": {
            "multi_match": {
                "query": query,
                "type": "best_fields",
                "tie_breaker": 0.3,
                "fields": [
                    "source^5",
                    "aliases^5",
                    "entities^3",
                    "topics^2",
                    "summary^1",
                ],
            }
        },
    }


def card_knn_body(vector: Sequence[float], size: int) -> dict[str, Any]:
    return {
        "size": size,
        "_source": {"excludes": ["embedding"]},
        "query": {"knn": {"embedding": {"vector": list(vector), "k": size}}},
    }


def fuse_card_hits(results: dict[str, Sequence[dict[str, Any]]]) -> list[CardHit]:
    sources: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = {}
    legs: dict[str, dict[str, int]] = {}
    for leg, rows in results.items():
        for rank, row in enumerate(rows, 1):
            doc_id = str(row["_source"]["doc_id"])
            sources.setdefault(doc_id, row["_source"])
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            legs.setdefault(doc_id, {})[leg] = rank
    return [
        CardHit(doc_id=d, fused=scores[d], source=sources[d], legs=legs[d])
        for d in sorted(scores, key=lambda x: (-scores[x], x))
    ]


def search_cards(
    os_client: OpenSearch,
    query: str,
    *,
    embedder: Bedrock,
    top_k: int = DOC_CANDIDATES,
    candidate_k: int = 20,
    name: str = DOC_INDEX,
) -> list[CardHit]:
    if not os_client.indices.exists(index=name):
        return []
    vector = embedder.embed([query])[0]
    bm25 = os_client.search(index=name, body=card_bm25_body(query, candidate_k))
    knn = os_client.search(index=name, body=card_knn_body(vector, candidate_k))
    results = {"bm25": bm25["hits"]["hits"], "knn": knn["hits"]["hits"]}
    return fuse_card_hits(results)[:top_k]


def search_lexical_docs(
    os_client: OpenSearch,
    query: str,
    *,
    top_k: int = LEXICAL_DOC_CANDIDATES,
    index: str = INDEX,
) -> list[str]:
    """문서 카드가 놓치는 희귀 원문 용어를 전역 BM25에서 문서 단위로 보존한다."""
    if not os_client.indices.exists(index=index):
        return []
    from haeindex.search import bm25_body

    body = bm25_body(query, [], top_k)
    body["_source"] = ["doc_id"]
    body["collapse"] = {"field": "doc_id"}
    rows = os_client.search(index=index, body=body)["hits"]["hits"]
    return list(dict.fromkeys(str(row["_source"]["doc_id"]) for row in rows))


def doc_card_count(os_client: OpenSearch, name: str = DOC_INDEX) -> int:
    if not os_client.indices.exists(index=name):
        return 0
    return int(os_client.count(index=name)["count"])

"""원문을 보존하는 문맥·조건·예외 생성과 절별 카드 색인."""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from opensearchpy.helpers import bulk, scan
from pydantic import Field

from haeindex.bedrock import EMBED_DIM, Bedrock
from haeindex.document_cards import DOC_INDEX, DocumentCard, ensure_doc_index, index_card
from haeindex.index import INDEX, SOURCE_EXCLUDE, settings
from haeindex.llm_tasks import TaskFailure, TaskRunner
from haeindex.profiling import span
from haeindex.reasoning import Record, compact, quote_exists

SECTION_INDEX = "haeindex_bedrock_sections_v1"
AUGMENT_VERSION = "context-rules-v1"


class Rule(Record):
    subject: str
    statement: str
    quote: str
    conditions: list[str] = Field(default_factory=list, max_length=5)
    exceptions: list[str] = Field(default_factory=list, max_length=4)


class ChunkContext(Record):
    chunk_id: str
    context: str = Field(max_length=400)
    summary: str = Field(max_length=500)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    entities: list[str] = Field(default_factory=list, max_length=12)
    content_type: str = "other"
    rules: list[Rule] = Field(default_factory=list, max_length=5)
    source_quote: str


class ContextBatch(Record):
    chunks: list[ChunkContext] = Field(max_length=3)


class Quality(Record):
    accepted_ids: list[str]
    reasons: list[str] = Field(default_factory=list)


class SectionSummary(Record):
    summary: str = Field(max_length=900)
    topics: list[str] = Field(default_factory=list, max_length=15)
    entities: list[str] = Field(default_factory=list, max_length=15)
    aliases: list[str] = Field(default_factory=list, max_length=10)
    document_type: str = "other"


def augment_batch(runner: TaskRunner, llm: Bedrock, sources: Sequence[dict]) -> list[ChunkContext]:
    inputs = [
        {k: s.get(k, "") for k in ["chunk_id", "doc_id", "path", "title", "body"]} for s in sources
    ]
    generated = runner.run(
        llm,
        "index-context",
        "각 청크의 검색용 문맥을 만드세요. context는 문서명·절과 청크 내용의 관계를 짧게 "
        "설명하세요. summary는 핵심입니다. 원문에 없는 숫자·명칭·적용조건을 만들지 마세요. "
        "keywords와 entities는 원문/제목의 표현을 그대로 쓰세요. content_type은 "
        "definition/procedure/requirement/table/fact/other 중 하나입니다. "
        "규정·절차는 rules에 대상(subject), 규칙(statement), 이를 포함한 원문 전체 문장 "
        "quote, conditions와 exceptions를 추출하세요. 조건·예외도 quote의 부분문자열이어야 "
        "합니다. 없는 예외는 빈 목록입니다. 숫자가 손상된 것 같아도 고치지 마세요. "
        "source_quote에는 summary를 뒷받침하는 원문 문장을 그대로 넣으세요. "
        "주어진 모든 청크 ID에 대해 항목 하나씩 반환하세요.",
        {"chunks": inputs},
        ContextBatch,
        num_predict=2800,
    )
    known = {s["chunk_id"]: s for s in sources}
    candidates = []
    for item in generated.chunks:
        source = known.get(item.chunk_id)
        if source is None or not quote_exists(item.source_quote, source["body"]):
            continue
        text = compact(f"{source.get('path', '')} {source.get('title', '')} {source['body']}")
        item.keywords = [x for x in item.keywords if compact(x) and compact(x) in text]
        item.entities = [x for x in item.entities if compact(x) and compact(x) in text]
        item.rules = [
            r
            for r in item.rules
            if quote_exists(r.quote, source["body"])
            and all(compact(x) in compact(r.quote) for x in [*r.conditions, *r.exceptions])
        ]
        if item.content_type not in {
            "definition",
            "procedure",
            "requirement",
            "table",
            "fact",
            "other",
        }:
            item.content_type = "other"
        candidates.append(item)
    if not candidates:
        return []
    checked = runner.run(
        llm,
        "index-quality",
        "생성된 검색 메타데이터를 원문과 대조하세요. "
        "요약·문맥·규칙에 원문에 없는 사실이나 잘못된 조건이 있으면 해당 ID를 "
        "제외하세요. 충실한 항목만 accepted_ids에 넣고 거부 이유를 reasons에 쓰세요.",
        {"original": inputs, "generated": [x.model_dump() for x in candidates]},
        Quality,
        num_predict=700,
    )
    return [x for x in candidates if x.chunk_id in checked.accepted_ids]


def section_groups(sources: Sequence[dict], batch_size: int = 10) -> list[list[dict]]:
    ordered = sorted(sources, key=lambda s: s.get("seq", 0))
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def ensure_accuracy_fields(os_client, index: str = INDEX) -> None:
    props = {
        "contextual_text": {"type": "text", "analyzer": "ko_index", "search_analyzer": "ko_search"},
        "context_summary": {"type": "text", "analyzer": "ko_index", "search_analyzer": "ko_search"},
        "context_embedding": {
            "type": "knn_vector",
            "dimension": EMBED_DIM,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene",
                "parameters": {"ef_construction": 128, "m": 16},
            },
        },
        "grounded_rules": {"type": "object", "enabled": False},
        "augmentation": {"type": "object", "enabled": False},
    }
    os_client.indices.put_mapping(index=index, body={"properties": props})
    if not os_client.indices.exists(index=SECTION_INDEX):
        os_client.indices.create(
            index=SECTION_INDEX,
            body={
                "settings": settings(),
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "card_id": {"type": "keyword"},
                        "chunk_ids": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "end_page": {"type": "integer"},
                        "text": {
                            "type": "text",
                            "analyzer": "ko_index",
                            "search_analyzer": "ko_search",
                        },
                        "embedding": props["context_embedding"],
                    },
                },
            },
        )


def enhance_document(
    os_client,
    llm: Bedrock,
    embedder: Bedrock,
    doc_id: str,
    *,
    index: str = INDEX,
    progress: Callable[[str], None] | None = None,
    cache: Path | None = Path("work/llm-cache"),
) -> dict:
    report = progress or (lambda _: None)
    sources = [
        h["_source"]
        for h in scan(
            os_client,
            index=index,
            query={
                "query": {"term": {"doc_id": doc_id}},
                "_source": {"excludes": SOURCE_EXCLUDE},
            },
        )
    ]
    sources.sort(key=lambda s: s.get("seq", 0))
    if not sources:
        raise ValueError(f"색인된 청크가 없습니다: {doc_id}")
    ensure_accuracy_fields(os_client, index)
    runner = TaskRunner(max_calls=max(20, len(sources) * 3), cache=cache)
    accepted = 0
    failures = []
    for start in range(0, len(sources), 3):
        batch = sources[start : start + 3]
        try:
            items = augment_batch(runner, llm, batch)
        except TaskFailure as exc:
            items = []
            failures.append(str(exc))
        by_id = {item.chunk_id: item for item in items}
        updates = []
        texts = []
        for source in batch:
            item = by_id.get(source["chunk_id"])
            if item is None:
                continue
            heading = source.get("path") or source.get("title") or ""
            text = f"{doc_id}\n{heading}\n{item.context}\n{source['body']}"
            texts.append(text)
            updates.append(
                {
                    "contextual_text": text,
                    "context_summary": item.context,
                    "summary": item.summary,
                    "keywords": item.keywords,
                    "entities": item.entities,
                    "content_type": item.content_type,
                    "grounded_rules": [r.model_dump() for r in item.rules],
                    "augmentation": {
                        "version": AUGMENT_VERSION,
                        "model": llm.chat_model,
                        "digest": llm.model_digest(),
                        "body_sha256": hashlib.sha256(source["body"].encode()).hexdigest(),
                    },
                }
            )
        if updates:
            vectors = embedder.embed_batched(texts)
            ids = [s["chunk_id"] for s in batch if s["chunk_id"] in by_id]
            actions = [
                {
                    "_op_type": "update",
                    "_index": index,
                    "_id": cid,
                    "doc": {**fields, "context_embedding": vector},
                }
                for cid, fields, vector in zip(ids, updates, vectors, strict=True)
            ]
            ok, errors = bulk(os_client, actions, raise_on_error=False)
            if errors:
                failures.append(f"청크 업데이트 실패 {len(errors)}건")
            accepted += ok
        report(
            f"{doc_id}: 청크 {min(start + 3, len(sources))}/{len(sources)} · 검증 통과 {accepted}"
        )
    cards = []
    for n, group in enumerate(section_groups(sources)):
        data = [
            {
                "chunk_id": s["chunk_id"],
                "heading": s.get("path") or s.get("title"),
                "body": s["body"][:850],
            }
            for s in group
        ]
        try:
            card = runner.run(
                llm,
                "section-card",
                "연속된 원문 묶음의 검색용 절 카드를 만드세요. "
                "주제·기관명·조건은 원문에 있는 것만 사용하세요. "
                "새로운 수치나 규정을 만들지 마세요.",
                {"doc_id": doc_id, "chunks": data},
                SectionSummary,
                num_predict=1200,
            )
        except TaskFailure as exc:
            failures.append(str(exc))
            continue
        text = "\n".join([doc_id, card.summary, *card.topics, *card.entities, *card.aliases])
        vector = embedder.embed([text])[0]
        card_id = f"{doc_id}#group{n:04d}"
        record = {
            "doc_id": doc_id,
            "card_id": card_id,
            "text": text,
            "chunk_ids": [s["chunk_id"] for s in group],
            "page": min(s["page"] for s in group),
            "end_page": max(s.get("end_page", s["page"]) for s in group),
            "embedding": vector,
        }
        os_client.index(index=SECTION_INDEX, id=card_id, body=record)
        cards.append(card)
        report(f"{doc_id}: 절 카드 {n + 1}/{len(section_groups(sources))}")
    if cards:
        summary = merge_section_cards(runner, llm, doc_id, cards)
        old = {}
        if os_client.indices.exists(index=DOC_INDEX) and os_client.exists(
            index=DOC_INDEX, id=doc_id
        ):
            old = os_client.get(index=DOC_INDEX, id=doc_id)["_source"]
        from haeindex.source_recovery import pdf_for

        path = pdf_for(doc_id)
        source_name = old.get("source") or (path.name if path else doc_id)
        summary.aliases = list(dict.fromkeys([*old.get("aliases", []), *summary.aliases]))[:15]
        from haeindex.document_cards import infer_source_language, normalize_document_type

        card = DocumentCard(
            doc_id=doc_id,
            source=source_name,
            **summary.model_dump(),
            language=infer_source_language(" ".join(s["body"][:100] for s in sources)),
        )
        card = card.model_copy(
            update={"document_type": normalize_document_type(card.document_type, doc_id)}
        )
        ensure_doc_index(os_client)
        index_card(os_client, card, embedder.embed([card.search_text()])[0])
    groups = section_groups(sources)
    if len(cards) == len(groups):
        keep = [f"{doc_id}#group{n:04d}" for n in range(len(groups))]
        os_client.delete_by_query(
            index=SECTION_INDEX,
            body={
                "query": {
                    "bool": {
                        "filter": [{"term": {"doc_id": doc_id}}],
                        "must_not": [{"terms": {"card_id": keep}}],
                    }
                }
            },
            refresh=True,
        )
    os_client.indices.refresh(index=index)
    os_client.indices.refresh(index=SECTION_INDEX)
    return {
        "doc_id": doc_id,
        "total": len(sources),
        "accepted": accepted,
        "section_cards": len(cards),
        "failures": failures,
        "calls": runner.calls,
        "events": [e.model_dump() for e in runner.events],
    }


def merge_section_cards(runner, llm, doc_id, cards):
    """모든 절을 계층적으로 통합해 긴 문서도 입력 예산 안에 담는다."""
    level = list(cards)
    while True:
        merged = []
        for start in range(0, len(level), 10):
            payload = [
                {
                    "summary": c.summary[:600],
                    "topics": c.topics[:8],
                    "entities": c.entities[:8],
                    "aliases": c.aliases[:4],
                    "document_type": c.document_type,
                }
                for c in level[start : start + 10]
            ]
            merged.append(
                runner.run(
                    llm,
                    "document-hierarchy",
                    "절 카드들을 검색용 문서 카드로 통합하세요. 모든 절의 주제를 포함하고 "
                    "새로운 사실은 추가하지 마세요. summary는 900자 이내입니다.",
                    {"doc_id": doc_id, "sections": payload},
                    SectionSummary,
                    num_predict=1400,
                )
            )
        if len(merged) == 1:
            return merged[0]
        level = merged


def section_candidates(
    os_client, query: str, doc_ids: Sequence[str], top_k: int = 3, embedder: Bedrock | None = None
) -> list[str]:
    with span("opensearch", "절 카드 인덱스 확인", index=SECTION_INDEX):
        if not os_client.indices.exists(index=SECTION_INDEX):
            return []
    filters = [{"terms": {"doc_id": list(doc_ids)}}] if doc_ids else []
    bodies = [
        {
            "size": top_k,
            "_source": ["chunk_ids"],
            "query": {"bool": {"must": [{"match": {"text": query}}], "filter": filters}},
        }
    ]
    if embedder is not None:
        knn = {"vector": embedder.embed([query])[0], "k": top_k}
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        bodies.append(
            {"size": top_k, "_source": ["chunk_ids"], "query": {"knn": {"embedding": knn}}}
        )
    scores, rows = {}, {}
    for number, body in enumerate(bodies):
        leg = "bm25" if number == 0 else "knn"
        with span("opensearch", "절 카드 검색", index=SECTION_INDEX, leg=leg):
            result = os_client.search(index=SECTION_INDEX, body=body)
        for rank, hit in enumerate(result["hits"]["hits"], 1):
            cid = hit["_id"]
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
            rows[cid] = hit["_source"]["chunk_ids"]
    order = sorted(scores, key=lambda cid: -scores[cid])[:top_k]
    # 절 하나의 첫 청크만 후보 예산을 독점하지 않게 번갈아 선택한다.
    groups = [rows[cid] for cid in order]
    return list(
        dict.fromkeys(
            group[i]
            for i in range(max(map(len, groups), default=0))
            for group in groups
            if i < len(group)
        )
    )

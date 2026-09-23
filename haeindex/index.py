from collections.abc import Iterator, Sequence
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from haeindex.bedrock import EMBED_DIM
from haeindex.chunks import Chunk
from haeindex.enrich import Enrichment

INDEX = "haeindex_bedrock_v1"
OS_HOST = "http://localhost:9200"
SOURCE_EXCLUDE = ["embedding", "context_embedding"]

KO_STOPTAGS = [
    "E",
    "IC",
    "J",
    "MM",
    "SP",
    "SSC",
    "SSO",
    "SC",
    "SE",
    "XPN",
    "XSA",
    "XSN",
    "XSV",
    "UNA",
    "NA",
    "VSV",
]


def settings() -> dict[str, Any]:
    return {
        "index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": True},
        "analysis": {
            "tokenizer": {
                "ko_tk_index": {"type": "nori_tokenizer", "decompound_mode": "mixed"},
                "ko_tk_search": {"type": "nori_tokenizer", "decompound_mode": "discard"},
            },
            "filter": {
                "ko_pos": {"type": "nori_part_of_speech", "stoptags": KO_STOPTAGS},
                "ko_reading": {"type": "nori_readingform"},
            },
            "analyzer": {
                "ko_index": {
                    "tokenizer": "ko_tk_index",
                    "filter": ["ko_pos", "ko_reading", "lowercase"],
                },
                "ko_search": {
                    "tokenizer": "ko_tk_search",
                    "filter": ["ko_pos", "ko_reading", "lowercase"],
                },
            },
        },
    }


def _text_field() -> dict[str, Any]:
    return {"type": "text", "analyzer": "ko_index", "search_analyzer": "ko_search"}


def mappings(dim: int = EMBED_DIM) -> dict[str, Any]:
    return {
        "properties": {
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "seq": {"type": "integer"},
            "section_ids": {"type": "keyword"},
            "title": _text_field(),
            "path": _text_field(),
            "depth": {"type": "integer"},
            "page": {"type": "integer"},
            "end_page": {"type": "integer"},
            "body": {"type": "text", "index": False},
            "text": _text_field(),
            "queries": _text_field(),
            "summary": _text_field(),
            "keywords": _text_field(),
            "entities": _text_field(),
            "content_type": {"type": "keyword"},
            "language": {"type": "keyword"},
            "char_len": {"type": "integer"},
            "token_len": {"type": "integer"},
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


def client(host: str = OS_HOST) -> OpenSearch:
    return OpenSearch([host])


def ensure_index(os_client: OpenSearch, name: str = INDEX, dim: int = EMBED_DIM) -> None:
    plugins = {p["component"] for p in os_client.cat.plugins(format="json")}
    if not any("nori" in p for p in plugins):
        raise RuntimeError(f"analysis-nori 플러그인이 없다. 설치된 것: {sorted(plugins)}")
    if not os_client.indices.exists(index=name):
        os_client.indices.create(
            index=name, body={"settings": settings(), "mappings": mappings(dim)}
        )
        return
    live = os_client.indices.get_mapping(index=name)[name]["mappings"].get("properties", {})
    missing = {k: v for k, v in mappings(dim)["properties"].items() if k not in live}
    if missing:
        os_client.indices.put_mapping(index=name, body={"properties": missing})


def replace_doc(os_client: OpenSearch, doc_id: str, name: str = INDEX) -> int:
    if not os_client.indices.exists(index=name):
        return 0
    r = os_client.delete_by_query(
        index=name, body={"query": {"term": {"doc_id": doc_id}}}, refresh=True
    )
    return int(r.get("deleted", 0))


def index_chunks(
    os_client: OpenSearch,
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    name: str = INDEX,
    queries: Sequence[Sequence[str]] | None = None,
    enrichments: Sequence[Enrichment | None] | None = None,
) -> tuple[int, list]:
    def actions() -> Iterator[dict[str, Any]]:
        for i, (c, vec) in enumerate(zip(chunks, embeddings, strict=True)):
            source = c.model_dump()
            if any(vec):
                source["embedding"] = list(vec)
            if queries and queries[i]:
                source["queries"] = "\n".join(queries[i])
            if enrichments and enrichments[i]:
                source.update(enrichments[i].model_dump())
            yield {"_index": name, "_id": c.chunk_id, "_source": source}

    ok, errors = bulk(os_client, actions(), chunk_size=50, raise_on_error=False)
    os_client.indices.refresh(index=name)
    return (int(ok), list(errors))


def find_covering(
    os_client: OpenSearch,
    doc_id: str,
    *,
    sections: Sequence[str] = (),
    pages: Sequence[int] = (),
    name: str = INDEX,
    size: int = 50,
) -> list[dict[str, Any]]:
    """정답 라벨을 덮는 청크를 찾는다. 채점(covered_targets)과 같은 술어를 쓴다."""
    if sections:
        should: list[dict[str, Any]] = [{"terms": {"section_ids": list(sections)}}]
    else:
        should = [
            {
                "bool": {
                    "filter": [
                        {"range": {"page": {"lte": p}}},
                        {"range": {"end_page": {"gte": p}}},
                    ]
                }
            }
            for p in pages
        ]
    if not should:
        return []
    body = {
        "size": size,
        "_source": {"excludes": SOURCE_EXCLUDE},
        "query": {
            "bool": {
                "filter": [{"term": {"doc_id": doc_id}}],
                "should": should,
                "minimum_should_match": 1,
            }
        },
        "sort": [{"seq": "asc"}],
    }
    return [h["_source"] for h in os_client.search(index=name, body=body)["hits"]["hits"]]


def doc_counts(os_client: OpenSearch, name: str = INDEX) -> dict[str, int]:
    return {doc_id: row["chunks"] for doc_id, row in doc_stats(os_client, name).items()}


def doc_stats(os_client: OpenSearch, name: str = INDEX) -> dict[str, dict[str, int]]:
    if not os_client.indices.exists(index=name):
        return {}
    agg = os_client.search(
        index=name,
        body={
            "size": 0,
            "aggs": {
                "d": {
                    "terms": {"field": "doc_id", "size": 200},
                    "aggs": {"last_page": {"max": {"field": "end_page"}}},
                }
            },
        },
    )
    return {
        bucket["key"]: {
            "chunks": int(bucket["doc_count"]),
            "last_page": int(bucket.get("last_page", {}).get("value") or 0),
        }
        for bucket in agg["aggregations"]["d"]["buckets"]
    }

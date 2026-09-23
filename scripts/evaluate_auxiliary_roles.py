"""Evaluate frozen role trials using isolated lexical indices and exact cosine.

No LLM judge. Page relevance requires matching document identity. These retrieval
metrics are candidate coverage, not final-answer accuracy.
"""

import argparse
import hashlib
import math
import random
import statistics
import time
from pathlib import Path

from opensearchpy.helpers import bulk

from haeindex.bedrock import Bedrock
from haeindex.evaluate import Question, covered_targets, score_query
from haeindex.index import client, mappings, settings
from haeindex.search import bm25_body, dedupe, doc_filter, rrf_fuse
from scripts.benchmark_auxiliary_roles import ROOT, dump, embedding_request, read

PREFIX = "haeindex_bench_aux_20260921_"


def load_vectors(root, name, label):
    return [
        v
        for p in sorted((root / "embeddings" / name).glob(f"{label}-*.json"))
        for v in read(p)["vectors"]
    ]


def plain_source(s):
    return {
        k: v
        for k, v in s.items()
        if k
        not in {
            "embedding",
            "context_embedding",
            "summary",
            "keywords",
            "entities",
            "augmentation",
            "contextual_text",
            "context_summary",
            "grounded_rules",
            "content_type",
        }
    }


def scratch(os_client, root, name, sources):
    import json

    suffix = (
        ""
        if root.resolve() == ROOT.resolve()
        else hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:8] + "_"
    )
    index = PREFIX + suffix + name
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if os_client.indices.exists(index=index):
        meta = os_client.indices.get_mapping(index=index)[index]["mappings"].get("_meta", {})
        if meta.get("source_sha256") != fingerprint:
            raise ValueError(f"Scratch fingerprint mismatch: {index}")
        if os_client.count(index=index)["count"] != len(sources):
            raise ValueError(f"Incomplete scratch index: {index}")
        return index
    mapping = mappings()
    mapping["properties"].pop("embedding", None)
    mapping["properties"]["contextual_text"] = {
        "type": "text",
        "analyzer": "ko_index",
        "search_analyzer": "ko_search",
    }
    mapping["_meta"] = {"source_sha256": fingerprint, "purpose": "isolated benchmark"}
    os_client.indices.create(index=index, body={"settings": settings(), "mappings": mapping})
    bulk(os_client, [{"_index": index, "_id": s["chunk_id"], "_source": s} for s in sources])
    os_client.indices.refresh(index=index)
    dump(
        root / f"scratch-{name}.json",
        {"index": index, "fingerprint": fingerprint, "count": len(sources)},
    )
    return index


def vector_ranks(sources, vectors, queries, query_vectors, scoped):
    result = {}
    for q, qv in zip(queries, query_vectors, strict=True):
        scores = [
            (math.sumprod(v, qv), s)
            for s, v in zip(sources, vectors, strict=True)
            if not scoped or s["doc_id"] == q.doc_id
        ]
        scores.sort(key=lambda x: (-x[0], x[1]["chunk_id"]))
        result[q.id] = [{"_score": score, "_source": source} for score, source in scores[:50]]
    return result


def lexical_ranks(os_client, index, queries, scoped, contextual=False):
    return {
        q.id: os_client.search(
            index=index,
            body=bm25_body(
                q.query,
                doc_filter([q.doc_id] if scoped else []),
                50,
                contextual=contextual,
            ),
        )["hits"]["hits"]
        for q in queries
    }


def grade(q, legs):
    hits, _ = dedupe(rrf_fuse(legs))
    sources = [h.source for h in hits]
    result = {"id": q.id, "doc_id": q.doc_id, "top_ids": [h.chunk_id for h in hits[:20]]}
    for k in [1, 5, 10, 20]:
        scored = score_query(q, sources, k)
        result.update(
            {
                f"hit@{k}": scored.rr > 0,
                f"recall@{k}": scored.recall,
                f"ndcg@{k}": scored.ndcg,
                f"mrr@{k}": scored.rr,
            }
        )
    return result


def aggregate(rows):
    keys = [k for k in rows[0] if "@" in k]
    return {
        "n": len(rows),
        **{k: statistics.fmean(r[k] for r in rows) for k in keys},
        "misses@5": [r["id"] for r in rows if "hit@5" in r and not r["hit@5"]],
    }


def evaluate_embeddings(root):
    sources = [plain_source(s) for s in read(root / "corpus.json")]
    queries = [Question(**q) for q in read(root / "questions.json")]
    os_client = client()
    summaries, rows = {}, {}
    try:
        index = scratch(os_client, root, "plain", sources)
        for scoped in [False, True]:
            scope = "document" if scoped else "global"
            bm25 = lexical_ranks(os_client, index, queries, scoped)
            key = f"{scope}/bm25"
            rows[key] = [grade(q, {"bm25": bm25[q.id]}) for q in queries]
            summaries[key] = aggregate(rows[key])
            for name in ["titan", "cohere-v3", "cohere-v4"]:
                vectors = load_vectors(root, name, "documents")
                qvectors = load_vectors(root, name, "queries")
                if len(vectors) != len(sources) or len(qvectors) != len(queries):
                    raise ValueError(f"Incomplete embeddings: {name}")
                ranked = vector_ranks(sources, vectors, queries, qvectors, scoped)
                for mode in ["vector", "hybrid"]:
                    key = f"{scope}/{name}/{mode}"
                    rows[key] = [
                        grade(
                            q,
                            {
                                "knn": ranked[q.id],
                                **({"bm25": bm25[q.id]} if mode == "hybrid" else {}),
                            },
                        )
                        for q in queries
                    ]
                    summaries[key] = aggregate(rows[key])
                    print(key, summaries[key], flush=True)
    finally:
        os_client.close()
    dump(root / "embedding-scores.json", rows)
    dump(root / "embedding-summary.json", summaries)


def contextual_sources(root, name, sources):
    items = {
        item["chunk_id"]: item
        for p in (root / "enrich" / name).glob("*.json")
        for item in read(p).get("accepted", [])
    }
    result, contexts = [], []
    for source in sources:
        s = dict(source)
        if item := items.get(s["chunk_id"]):
            heading = s.get("path") or s.get("title") or ""
            text = f"{s['doc_id']}\n{heading}\n{item['context']}\n{s['body']}"
            s.update(
                contextual_text=text,
                summary=item["summary"],
                keywords=item["keywords"],
                entities=item["entities"],
            )
            contexts.append(s)
        result.append(s)
    return result, contexts


def cached_titan(root, key, texts):
    import json

    path = root / "evaluation-vectors" / f"{key}.json"
    fingerprint = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    if path.exists():
        value = read(path)
        if value["fingerprint"] != fingerprint:
            raise ValueError(f"Vector cache mismatch {key}")
        return value["vectors"]
    with Bedrock() as llm:
        vectors = llm.embed_batched(texts)
    dump(path, {"fingerprint": fingerprint, "vectors": vectors})
    return vectors


def evaluate_enrichment(root, *, include_cards=True):
    sources = [plain_source(s) for s in read(root / "corpus.json")]
    all_queries = [Question(**q) for q in read(root / "questions.json")]
    documents = read(root / "manifest.json")["enrich_docs"]
    expected_batches = sum((sum(s["doc_id"] == d for s in sources) + 2) // 3 for d in documents)
    for name in ["haiku", "terra", "sol"]:
        if len(list((root / "enrich" / name).glob("*.json"))) != expected_batches:
            raise ValueError(f"Incomplete enrichment: {name}")
        if include_cards and len(list((root / "cards" / name).glob("*.json"))) != len(documents):
            raise ValueError(f"Incomplete cards: {name}")
    queries = [q for q in all_queries if q.doc_id in documents]
    all_qv = dict(
        zip([q.id for q in all_queries], load_vectors(root, "titan", "queries"), strict=True)
    )
    qvectors = [all_qv[q.id] for q in queries]
    vectors = load_vectors(root, "titan", "documents")
    summaries, rows = {}, {}
    os_client = client()
    try:
        for scoped in [False, True]:
            scope = "document" if scoped else "global"
            base_ranks = vector_ranks(sources, vectors, queries, qvectors, scoped)
            for name in ["plain", "haiku", "terra", "sol"]:
                enhanced, contexts = (
                    (sources, []) if name == "plain" else contextual_sources(root, name, sources)
                )
                index = scratch(os_client, root, name, enhanced)
                bm25 = lexical_ranks(os_client, index, queries, scoped, contextual=name != "plain")
                contextual = {}
                if contexts:
                    cvs = cached_titan(
                        root, f"context-{name}", [s["contextual_text"] for s in contexts]
                    )
                    contextual = vector_ranks(contexts, cvs, queries, qvectors, scoped)
                for mode in ["bm25", "hybrid"]:
                    key = f"{scope}/{name}/{mode}"
                    rows[key] = []
                    for q in queries:
                        legs = {"bm25": bm25[q.id]}
                        if mode == "hybrid":
                            legs["knn"] = base_ranks[q.id]
                            if contexts:
                                legs["context-knn"] = contextual[q.id]
                        rows[key].append(grade(q, legs))
                    summaries[key] = aggregate(rows[key])
                    print(key, summaries[key], flush=True)
    finally:
        os_client.close()
    dump(root / "enrichment-scores.json", rows)
    dump(root / "enrichment-summary.json", summaries)
    if include_cards:
        evaluate_cards(root, sources, queries, qvectors)


def evaluate_chunks(root):
    evaluate_enrichment(root, include_cards=False)


def evaluate_cardsonly(root):
    sources = [plain_source(s) for s in read(root / "corpus.json")]
    all_queries = [Question(**q) for q in read(root / "questions.json")]
    documents = read(root / "manifest.json")["enrich_docs"]
    for name in ["haiku", "terra", "sol"]:
        if len(list((root / "cards" / name).glob("*.json"))) != len(documents):
            raise ValueError(f"Incomplete cards: {name}")
    pairs = [
        (q, v)
        for q, v in zip(all_queries, load_vectors(root, "titan", "queries"), strict=True)
        if q.doc_id in documents
    ]
    evaluate_cards(root, sources, [q for q, _ in pairs], [v for _, v in pairs])


def evaluate_cards(root, sources, queries, qvectors):
    """Vector-only card routing diagnostic; no claim of full pipeline accuracy."""
    from haeindex.augmentation import section_groups

    docs = sorted({s["doc_id"] for s in sources})
    summaries = {}
    for name in ["plain", "haiku", "terra", "sol"]:
        records = {p.stem: read(p) for p in (root / "cards" / name).glob("*.json")}
        doc_texts = []
        sections = []
        for doc in docs:
            group = [s for s in sources if s["doc_id"] == doc]
            r = records.get(doc, {})
            card = r.get("document")
            text = (
                "\n".join(
                    [doc, card["summary"], *card["topics"], *card["entities"], *card["aliases"]]
                )
                if card
                else "\n".join(
                    [doc, *[s["body"][:260] for s in group[:: max(1, len(group) // 20)]]]
                )
            )
            doc_texts.append(text)
            if doc not in {q.doc_id for q in queries}:
                continue
            if r.get("sections"):
                for sec in r["sections"]:
                    sections.append(
                        {
                            **sec,
                            "text": "\n".join(
                                [
                                    doc,
                                    sec["summary"],
                                    *sec["topics"],
                                    *sec["entities"],
                                    *sec["aliases"],
                                ]
                            ),
                        }
                    )
            else:
                for g in section_groups(group):
                    sections.append(
                        {
                            "doc_id": doc,
                            "chunk_ids": [s["chunk_id"] for s in g],
                            "text": "\n".join([doc, *[s["body"][:260] for s in g]]),
                        }
                    )
        dv = cached_titan(root, f"doccards-{name}", doc_texts)
        sv = cached_titan(root, f"sections-{name}", [s["text"] for s in sections])
        by_id = {s["chunk_id"]: s for s in sources}
        results = []
        for q, qv in zip(queries, qvectors, strict=True):
            dr = sorted(range(len(docs)), key=lambda i: -math.sumprod(dv[i], qv))
            sr = sorted(
                [i for i, s in enumerate(sections) if s["doc_id"] == q.doc_id],
                key=lambda i: -math.sumprod(sv[i], qv),
            )
            row = {
                "id": q.id,
                "doc@1": docs[dr[0]] == q.doc_id,
                "doc@3": q.doc_id in [docs[i] for i in dr[:3]],
            }
            for k in [1, 3]:
                row[f"section@{k}"] = any(
                    covered_targets(by_id[cid], q)
                    for i in sr[:k]
                    for cid in sections[i]["chunk_ids"]
                )
            results.append(row)
        summaries[name] = aggregate(results)
        dump(root / f"card-scores-{name}.json", results)
    dump(root / "card-summary.json", summaries)
    print("cards", summaries, flush=True)


def evaluate_latency(root):
    path = root / "embedding-single-latency.json"
    if path.exists():
        raise ValueError("Latency sample already exists")
    queries = read(root / "questions.json")[:10]
    jobs = [(name, q) for name in ["titan", "cohere-v3", "cohere-v4"] for q in queries]
    random.Random(20260921).shuffle(jobs)
    rows = []
    with Bedrock() as llm:
        for name, q in jobs:
            started = time.monotonic()
            embedding_request(llm, name, [q["query"]], "search_query")
            rows.append({"model": name, "id": q["id"], "seconds": time.monotonic() - started})
    summary = {
        name: {
            "n": 10,
            "mean_seconds": statistics.fmean(r["seconds"] for r in rows if r["model"] == name),
        }
        for name in ["titan", "cohere-v3", "cohere-v4"]
    }
    dump(path, {"rows": rows, "summary": summary, "seed": 20260921})
    print(summary, flush=True)


def evaluate_verify(root):
    from opensearchpy.helpers import scan

    from haeindex.augmentation import SECTION_INDEX
    from haeindex.document_cards import DOC_INDEX
    from haeindex.index import INDEX

    os_client = client()
    try:
        live = sorted(
            (h["_source"] for h in scan(os_client, index=INDEX)),
            key=lambda s: (s["doc_id"], s["seq"]),
        )
        result = {
            "live_corpus_equal_to_snapshot": live == read(root / "corpus.json"),
            "counts": {
                i: os_client.count(index=i)["count"] if os_client.indices.exists(index=i) else None
                for i in [INDEX, SECTION_INDEX, DOC_INDEX]
            },
        }
        dump(root / "live-verification.json", result)
        if not result["live_corpus_equal_to_snapshot"]:
            raise ValueError("Live corpus differs from snapshot")
        print(result, flush=True)
    finally:
        os_client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["embeddings", "enrichment", "chunks", "cardsonly", "latency", "verify"]
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    files = [
        Path(__file__),
        Path("haeindex/evaluate.py"),
        Path("haeindex/search.py"),
        Path("haeindex/index.py"),
    ]
    dump(
        args.root / "evaluation-manifest.json",
        {
            "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "matching": "document identity plus annotated page range; not answer accuracy",
        },
    )
    globals()[f"evaluate_{args.command}"](args.root)


if __name__ == "__main__":
    main()

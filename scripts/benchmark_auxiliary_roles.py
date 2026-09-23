"""Reproducible vision, enrichment and embedding trials; never write live indices.

Run prepare, probe, vision, enrich, embeddings in order (vision/enrich may overlap).
Artifacts contain source documents and must remain in ignored work/.
"""

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pdfplumber
import yaml
from opensearchpy.helpers import scan

from haeindex.augmentation import SectionSummary, augment_batch, merge_section_cards, section_groups
from haeindex.bedrock import DEFAULT_LIGHT_MODEL, DEFAULT_MEDIUM_MODEL, Bedrock, normalize
from haeindex.evaluate import Goldset
from haeindex.index import INDEX, client
from haeindex.llm_tasks import TaskRunner
from haeindex.models import model_client
from haeindex.profiling import Profiler
from haeindex.search import Hit
from haeindex.source_recovery import pdf_for, recover_pages

ROOT = Path("work/auxiliary-comparison-20260921")
MODELS = {
    "haiku": DEFAULT_LIGHT_MODEL,
    "sonnet": DEFAULT_MEDIUM_MODEL,
    "terra": "us.openai.gpt-5.6-terra",
    "sol": "us.openai.gpt-5.6-sol",
}
EMBED_MODELS = {
    "titan": "amazon.titan-embed-text-v2:0",
    "cohere-v3": "cohere.embed-multilingual-v3",
    "cohere-v4": "cohere.embed-v4:0",
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path):
    return json.loads(path.read_text())


def prepare(root):
    if (root / "manifest.json").exists():
        raise ValueError("Prepared already; use existing snapshot or a new root")
    root.mkdir(parents=True, exist_ok=True)
    os_client = client()
    try:
        corpus = sorted(
            (h["_source"] for h in scan(os_client, index=INDEX)),
            key=lambda s: (s["doc_id"], s["seq"]),
        )
        dump(root / "corpus.json", corpus)
        state = os_client.indices.stats(index=INDEX, metric="docs", level="shards")
        dump(root / "live-state-before.json", state)
    finally:
        os_client.close()
    gold = Goldset.load(Path("goldset/all.yaml"))
    absent = yaml.safe_load(Path("goldset/answer-rubric.yaml").read_text())["expected_absent"]
    # Audited corrections: continuation pages and two incorrectly absent labels.
    corrections = {
        "bid-03": [51, 52],
        "bid-10": [52, 53],
        "rule-17": [5],
        "ship-01": [2, 3],
        "cam-10": [7, 8],
        "absent-03": [12],
        "absent-05": [1],
    }
    questions = []
    for q in gold.questions:
        if q.id in absent:
            continue
        data = q.model_dump()
        if q.id in corrections:
            data.update(pages=corrections[q.id], sections=[], bucket="A")
        questions.append(data)
    dump(root / "questions.json", questions)
    by_id = {q.id: q for q in gold.questions}
    cases = []
    for case, qid, page in [
        ("bid-percent", "bid-01", 44),
        ("battery", "cam-05", 19),
        ("format", "cam-01", 60),
        ("budget-table", "mohw-01", 3),
        ("budget-history", "mohw-05", 4),
        ("certificate", "cert-01", 1),
        ("ship-table", "ship-01", 2),
        ("paper-diagram", "paper-03", 15),
    ]:
        q = by_id[qid]
        item = {"id": case, "qid": qid, "doc_id": q.doc_id, "page": page, "question": q.query}
        if case == "budget-table":
            item["question"] = "건물대여료의 2022년 결산, 2023년 본예산, 2024년 확정 예산과 증감은?"
        with pdfplumber.open(pdf_for(q.doc_id)) as pdf:
            p = pdf.pages[page - 1].dedupe_chars()
            dest = root / "pages" / f"{case}.png"
            dest.parent.mkdir(exist_ok=True)
            p.to_image(resolution=160).original.save(dest)
            item["image_sha256"] = hashlib.sha256(dest.read_bytes()).hexdigest()
            item["pdf_text"] = p.extract_text() or ""
        cases.append(item)
    dump(root / "vision-cases.json", cases)
    counts = Counter(s["doc_id"] for s in corpus)
    enrich_docs = sorted(d for d, n in counts.items() if n <= 101)
    files = {}
    for p in [
        *Path("haeindex").glob("*.py"),
        Path(__file__),
        Path("goldset/all.yaml"),
        Path("goldset/answer-rubric.yaml"),
    ]:
        relative = p.relative_to(Path.cwd()) if p.is_absolute() else p
        target = root / "source-snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
        files[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump(
        root / "manifest.json",
        {
            "created": datetime.now(UTC).isoformat(),
            "files": files,
            "models": MODELS,
            "embedding_models": EMBED_MODELS,
            "index": INDEX,
            "counts": dict(counts),
            "corpus_sha256": hashlib.sha256((root / "corpus.json").read_bytes()).hexdigest(),
            "questions": len(questions),
            "corrections": corrections,
            "enrich_docs": enrich_docs,
            "vision": "160 DPI; real recover_pages; unchanged prompts/caps; no cache",
            "embedding": "same normalized text; 1024 dims; v3 END, v4 NONE; query/doc input types",
            "enrichment": "complete four small documents; unchanged augment_batch; no cache",
        },
    )
    print(
        json.dumps(
            {"counts": counts, "questions": len(questions), "enrich_docs": enrich_docs},
            ensure_ascii=False,
        ),
        flush=True,
    )


def embedding_request(llm, name, texts, input_type):
    if name == "titan":
        return llm.embed_batched(texts), {}
    body = {
        "texts": [normalize(t) for t in texts],
        "input_type": input_type,
        "embedding_types": ["float"],
        "truncate": "END" if name == "cohere-v3" else "NONE",
    }
    if name == "cohere-v4":
        body["output_dimension"] = 1024
    response = llm._runtime.invoke_model(
        modelId=EMBED_MODELS[name],
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    data = json.loads(response["body"].read())
    vectors = data["embeddings"]
    if isinstance(vectors, dict):
        vectors = vectors["float"]
    normalized = []
    for v in vectors:
        if len(v) != 1024:
            raise ValueError(f"Unexpected dimension: {len(v)}")
        norm = math.sqrt(sum(x * x for x in v)) or 1
        normalized.append([x / norm for x in v])
    if len(normalized) != len(texts):
        raise ValueError("Embedding count mismatch")
    return normalized, {k: v for k, v in data.items() if k not in {"embeddings", "texts"}}


def probe(root):
    results = []
    with Bedrock() as llm:
        for name in EMBED_MODELS:
            started = time.monotonic()
            try:
                vectors, usage = embedding_request(
                    llm, name, ["배터리 충전 시간은 약 105분이다."], "search_document"
                )
                result = {"model": name, "ok": True, "dim": len(vectors[0]), "usage": usage}
            except Exception as exc:
                result = {"model": name, "ok": False, "error": str(exc)}
            result["seconds"] = time.monotonic() - started
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    dump(root / "embedding-probes.json", results)


class RecordingRunner(TaskRunner):
    def __init__(self):
        super().__init__(max_calls=20, cache=None)
        self.outputs = []

    def run(self, *args, **kwargs):
        value = super().run(*args, **kwargs)
        self.outputs.append({"task": args[1], "value": value.model_dump()})
        return value


def measured_job(root, kind, name, key, work):
    output = root / kind / name / f"{key}.json"
    if output.exists():
        return read(output)
    runner = RecordingRunner()
    profiler = Profiler()
    token = profiler.activate()
    started = time.monotonic()
    result = {"model": name, "id": key}
    try:
        with model_client(kind if kind == "vision" else "enrich", model=MODELS[name]) as llm:
            result.update(work(runner, llm))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        Profiler.deactivate(token)
    result.update(
        seconds=time.monotonic() - started,
        calls=runner.calls,
        events=[e.model_dump() for e in runner.events],
        outputs=runner.outputs,
        spans=profiler.spans,
    )
    dump(output, result)
    print(
        f"{kind} {name} {key}: {result['seconds']:.1f}s {result.get('error', 'done')}", flush=True
    )
    return result


def vision(root, workers):
    def run(name, case):
        def work(runner, llm):
            hit = Hit(chunk_id=case["id"], fused=0, source=case)
            hits = recover_pages(
                [hit],
                runner=runner,
                vision=llm,
                max_pages=1,
                use_vision=True,
                question=case["question"],
            )
            return {"case": case, "hits": [h.model_dump() for h in hits]}

        return measured_job(root, "vision", name, case["id"], work)

    jobs = [(name, case) for name in MODELS for case in read(root / "vision-cases.json")]
    random.Random(20260921).shuffle(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run, *job) for job in jobs]):
            future.result()


def enrich(root, workers):
    sources = read(root / "corpus.json")
    documents = read(root / "manifest.json")["enrich_docs"]
    groups = {d: [s for s in sources if s["doc_id"] == d] for d in documents}
    names = ["haiku", "terra", "sol"]
    jobs = [
        (name, d, i, group[i : i + 3])
        for name in names
        for d, group in groups.items()
        for i in range(0, len(group), 3)
    ]
    random.Random(20260921).shuffle(jobs)

    def run_batch(name, doc, offset, group):
        def work(runner, llm):
            items = augment_batch(runner, llm, group)
            return {
                "doc_id": doc,
                "source_ids": [s["chunk_id"] for s in group],
                "accepted": [item.model_dump() for item in items],
            }

        return measured_job(root, "enrich", name, f"{doc}-{offset:04d}", work)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run_batch, *j) for j in jobs]):
            f.result()

    def run_cards(name, doc, group):
        def work(runner, llm):
            runner.max_calls = 60
            cards, records = [], []
            for n, sections in enumerate(section_groups(group)):
                data = [
                    {
                        "chunk_id": s["chunk_id"],
                        "heading": s.get("path") or s.get("title"),
                        "body": s["body"][:850],
                    }
                    for s in sections
                ]
                card = runner.run(
                    llm,
                    "section-card",
                    "연속된 원문 묶음의 검색용 절 카드를 만드세요. "
                    "주제·기관명·조건은 원문에 있는 것만 사용하세요. "
                    "새로운 수치나 규정을 만들지 마세요.",
                    {"doc_id": doc, "chunks": data},
                    SectionSummary,
                    num_predict=1200,
                )
                cards.append(card)
                records.append(
                    {
                        "card_id": f"{doc}#group{n:04d}",
                        "doc_id": doc,
                        "chunk_ids": [s["chunk_id"] for s in sections],
                        **card.model_dump(),
                    }
                )
            merged = merge_section_cards(runner, llm, doc, cards)
            return {"doc_id": doc, "sections": records, "document": merged.model_dump()}

        return measured_job(root, "cards", name, doc, work)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [(n, d, g) for n in names for d, g in groups.items()]
        for f in as_completed([pool.submit(run_cards, *j) for j in jobs]):
            f.result()


def embeddings(root):
    corpus = read(root / "corpus.json")
    questions = read(root / "questions.json")
    available = {r["model"] for r in read(root / "embedding-probes.json") if r["ok"]}
    with Bedrock() as llm:
        for name in EMBED_MODELS:
            if name not in available:
                continue
            for label, texts, kind in [
                ("documents", [s["text"] for s in corpus], "search_document"),
                ("queries", [q["query"] for q in questions], "search_query"),
            ]:
                for start in range(0, len(texts), 32):
                    path = root / "embeddings" / name / f"{label}-{start:04d}.json"
                    if path.exists():
                        continue
                    batch = texts[start : start + 32]
                    prof = Profiler()
                    token = prof.activate()
                    begun = time.monotonic()
                    try:
                        vectors, usage = embedding_request(llm, name, batch, kind)
                    finally:
                        Profiler.deactivate(token)
                    dump(
                        path,
                        {
                            "offset": start,
                            "vectors": vectors,
                            "usage": usage,
                            "seconds": time.monotonic() - begun,
                            "spans": prof.spans,
                            "input_chars": sum(len(normalize(t)) for t in batch),
                        },
                    )
                    print(
                        f"embed {name} {label} {min(start + 32, len(texts))}/{len(texts)}",
                        flush=True,
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "probe", "vision", "enrich", "embeddings"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command in {"vision", "enrich"}:
        globals()[args.command](args.root, args.workers)
    else:
        globals()[args.command](args.root)


if __name__ == "__main__":
    main()

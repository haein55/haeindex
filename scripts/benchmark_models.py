"""실제 ask의 검색 후보를 고정하고 모델별 재정렬·답변을 기록한다.

실행: .venv/bin/python -m scripts.benchmark_models prepare|run|review
모델 목록은 HAEINDEX_BEDROCK_BENCHMARK_MODELS에 쉼표로 지정한다.
"""

import argparse
import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import pdfplumber

from haeindex.answer import answer, assemble
from haeindex.bedrock import DEFAULT_EMBED_MODEL, Bedrock
from haeindex.document_cards import search_cards, search_lexical_docs
from haeindex.evaluate import Goldset, Question, score_query
from haeindex.evidence import select_evidence
from haeindex.index import client, doc_counts
from haeindex.listwise import rerank
from haeindex.models import model_client
from haeindex.paths import slugify
from haeindex.routing import Route, route_question
from haeindex.search import Result, search

ROOT = Path("work/model-benchmark-20260914")
GOLD = Path("goldset/all.yaml")
MODELS = [
    model.strip()
    for model in os.environ.get("HAEINDEX_BEDROCK_BENCHMARK_MODELS", "").split(",")
    if model.strip()
]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path) -> None:
    snapshot = root / "candidates.json"
    if snapshot.exists():
        raise SystemExit("후보 스냅샷이 이미 있습니다. 다른 --root를 사용하세요.")
    os_client = client()
    counts = doc_counts(os_client)
    gold = Goldset.load(GOLD)
    gold.verify(sorted(counts))
    records = []
    with model_client("analysis") as ol:
        for i, q in enumerate(gold.questions, 1):
            started = time.monotonic()
            route = route_question(q.query, sorted(counts))
            wanted = route.doc_ids
            if not wanted:
                cards = search_cards(os_client, q.query, embedder=ol, top_k=3)
                lexical = search_lexical_docs(os_client, q.query)
                wanted = list(dict.fromkeys([*lexical, *(h.doc_id for h in cards)]))
                route = route.model_copy(update={"doc_ids": wanted})
            result = search(os_client, q.query, embedder=ol, doc_ids=wanted, top_k=20)
            if result.degraded:
                raise RuntimeError(f"검색 장애: {q.id}: {result.degraded}")
            records.append({
                "question": q.model_dump(), "route": route.model_dump(),
                "result": result.model_dump(), "retrieval_seconds": time.monotonic() - started,
            })
            print(f"prepare {i}/{len(gold.questions)} {q.id}", flush=True)
    write_json(snapshot, records)
    references = {}
    pdfs = {slugify(p): p for p in Path("data/inbox").glob("*.pdf")}
    for doc_id, pdf_path in pdfs.items():
        needed = sorted({
            p + delta for q in gold.questions if q.doc_id == doc_id
            for p in q.pages for delta in [-1, 0, 1]
        })
        if doc_id.startswith("hxr-"):
            needed = sorted(set(needed) | {12})
        if doc_id.startswith("agentic-"):
            needed = sorted(set(needed) | {1})
        if doc_id.startswith("2024-"):
            needed = list(range(1, 16))
        with pdfplumber.open(pdf_path) as pdf:
            references[doc_id] = {
                str(p): pdf.pages[p - 1].extract_text() or ""
                for p in needed if 1 <= p <= len(pdf.pages)
            }
    write_json(root / "references.json", references)
    sources = [GOLD, *sorted(Path("haeindex").glob("*.py")), Path(__file__)]
    write_json(root / "manifest.json", {
        "created": datetime.now(UTC).isoformat(), "models": MODELS,
        "embedding": DEFAULT_EMBED_MODEL, "documents": counts,
        "num_ctx": 8192, "temperature": 0, "seed": 0, "think": False,
        "depth": 20, "top_k": 5, "min_cos": 0.78,
        "snapshot_sha256": digest(snapshot),
        "files": {str(p): digest(p) for p in sources},
    })


class MeasuredBedrock(Bedrock):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
        self.calls = []

    def chat(self, messages, **kwargs):
        started = time.monotonic()
        call = {"num_predict": kwargs.get("num_predict", 512)}
        try:
            reply = super().chat(messages, **kwargs)
            call["text"] = reply.content
            return reply
        except Exception as exc:
            call["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            call["seconds"] = time.monotonic() - started
            self.calls.append(call)


def run(root: Path, model: str, limit: int | None = None) -> None:
    records = json.loads((root / "candidates.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if digest(root / "candidates.json") != manifest["snapshot_sha256"]:
        raise ValueError("후보 스냅샷이 변경됐습니다")
    output = root / f"{model.replace(':', '_')}.jsonl"
    completed = {}
    if output.exists():
        completed = {r["id"]: r for r in map(json.loads, output.read_text().splitlines())}
    with MeasuredBedrock(model) as ol:
        write_json(root / f"{model.replace(':', '_')}.model.json", {
            "provider": "bedrock", "model_id": model, "region": ol.region,
            "embedding": ol.embed_model, "benchmark_sha256": digest(Path(__file__)),
            "execution_files": {str(p): digest(p) for p in sorted(Path("haeindex").glob("*.py"))},
        })
        executed = 0
        for i, record in enumerate(records, 1):
            q = Question.model_validate(record["question"])
            if q.id in completed:
                continue
            route = Route.model_validate(record["route"])
            candidate = Result.model_validate(record["result"])
            ol.calls.clear()
            started = time.monotonic()
            row = {"id": q.id, "model": model, "question": q.query, "bucket": q.bucket}
            try:
                ranked = rerank(ol, q.query, candidate.hits, depth=20)
                result = select_evidence(
                    candidate.model_copy(update={"hits": ranked.hits}),
                    top_k=5, allow_multiple_docs=route.allow_multiple_docs,
                )
                # 거부된 경우도 검색 근거를 남겨 모델/검색 실패를 구분한다.
                row["evidence"] = assemble(result).model_dump()
                row["hit_ids"] = [h.chunk_id for h in result.hits]
                row["rerank_parse_fails"] = ranked.parse_fails
                row["retrieval_score"] = score_query(
                    q, [h.source for h in result.hits], 5
                ).model_dump()
                row["answer"] = answer(ol, result, q.query).model_dump()
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["model_seconds"] = time.monotonic() - started
            row["retrieval_seconds"] = record["retrieval_seconds"]
            row["calls"] = list(ol.calls)
            with output.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = row.get("error") or row.get("answer", {}).get("refusal") or "answer"
            print(f"{model} {i}/{len(records)} {q.id} "
                  f"{row['model_seconds']:.2f}s {status}", flush=True)
            executed += 1
            if limit is not None and executed >= limit:
                return


def review(root: Path) -> None:
    records = json.loads((root / "candidates.json").read_text())
    refs = json.loads((root / "references.json").read_text())
    all_rows = {}
    for model in MODELS:
        path = root / f"{model.replace(':', '_')}.jsonl"
        all_rows[model] = {
            r["id"]: r for r in map(json.loads, path.read_text().splitlines())
        } if path.exists() else {}
    key = {}
    for i, record in enumerate(records):
        q = record["question"]
        models = list(MODELS)
        random.Random(90210 + i).shuffle(models)
        candidates = {}
        for n, model in enumerate(models):
            label = chr(65 + n)
            key[f"{q['id']}:{label}"] = model
            if q["id"] not in all_rows[model]:
                continue
            row = all_rows[model][q["id"]]
            candidates[label] = {
                "answer": row.get("answer"), "error": row.get("error"),
                "evidence": row.get("evidence"),
            }
        write_json(root / "blind" / f"{q['id']}.json", {
            "question": q,
            "reference_pages": {
                str(p): refs[q["doc_id"]][str(p)]
                for p in sorted({p + d for p in q["pages"] for d in [-1, 0, 1]})
                if str(p) in refs[q["doc_id"]]
            },
            "candidates": candidates,
        })
    write_json(root / "blind_key.json", key)


def wait_for_model(root: Path, model: str) -> None:
    """다른 모델의 마지막 답변이 저장된 뒤 시작해 GPU 측정이 겹치지 않게 한다."""
    expected = {r["question"]["id"] for r in json.loads((root / "candidates.json").read_text())}
    path = root / f"{model.replace(':', '_')}.jsonl"
    print(f"Waiting for {model} to finish before starting the next model", flush=True)
    while True:
        if path.exists():
            try:
                rows = [json.loads(s) for s in path.read_text().splitlines()]
            except json.JSONDecodeError:
                rows = []  # 다른 프로세스가 마지막 행을 쓰는 중이면 다시 읽는다.
            if len(rows) == len(expected) and {r["id"] for r in rows} == expected:
                return
        time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "run-all", "review"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model", choices=MODELS or None)
    parser.add_argument("--after", choices=MODELS or None, help="이 모델의 실행 완료 후 run 시작")
    parser.add_argument("--limit", type=int, help="이번 run에서 추가 실행할 최대 문항 수")
    args = parser.parse_args()
    if args.after and (args.command != "run" or args.after == args.model):
        parser.error("--after는 run에서 다른 모델을 기다릴 때만 사용할 수 있습니다")
    if args.limit is not None and (args.command != "run" or args.limit < 1):
        parser.error("--limit는 run에서 양수로 지정해야 합니다")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "run":
        if not args.model:
            parser.error("run에는 --model이 필요합니다")
        if args.after:
            wait_for_model(args.root, args.after)
        run(args.root, args.model, args.limit)
    elif args.command == "run-all":
        if not MODELS:
            parser.error("HAEINDEX_BEDROCK_BENCHMARK_MODELS를 설정하세요")
        total = len(json.loads((args.root / "candidates.json").read_text()))
        pending = []
        for model in MODELS:
            path = args.root / f"{model.replace(':', '_')}.jsonl"
            if not path.exists() or len(path.read_text().splitlines()) < total:
                pending.append(model)
        while pending:
            ready = pending.pop(0)
            run(args.root, ready)
    else:
        review(args.root)


if __name__ == "__main__":
    main()

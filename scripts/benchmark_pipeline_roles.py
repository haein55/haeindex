"""실제 강화 파이프라인에서 답변/분석 모델 조합을 비교한다.

대표 표본 파일럿:
  .venv/bin/python -m scripts.benchmark_pipeline_roles run --config current
전체 파일럿과 요약:
  .venv/bin/python -m scripts.benchmark_pipeline_roles run-all
  .venv/bin/python -m scripts.benchmark_pipeline_roles summarize

각 조합은 캐시를 사용하지 않는다. 검색, 프롬프트, 호출 예산, 비전 모델은 같게 두고
답변 모델과 분석 모델만 바꾼다. JSONL은 문항마다 append하므로 중단 후 재개할 수 있다.
"""

import argparse
import json
import statistics
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from haeindex.bedrock import (
    DEFAULT_HEAVY_MODEL,
    DEFAULT_LIGHT_MODEL,
    DEFAULT_MEDIUM_MODEL,
    Bedrock,
)
from haeindex.evaluate import Goldset, Question, score_query
from haeindex.index import client, doc_counts
from haeindex.pipeline import PipelineTrace, run_pipeline

ROOT = Path("work/pipeline-role-benchmark-20260921")
GOLD = Path("goldset/all.yaml")
PILOT_IDS = (
    "bid-01",
    "mohw-01",
    "cam-05",
    "rule-12",
    "cam-01",
    "bid-03",
    "cert-01",
    "paper-03",
    "absent-01",
    "absent-09",
)
CONFIGS = {
    "current": {
        "answer": DEFAULT_HEAVY_MODEL,
        "analysis": DEFAULT_MEDIUM_MODEL,
    },
    "sonnet": {
        "answer": DEFAULT_MEDIUM_MODEL,
        "analysis": DEFAULT_MEDIUM_MODEL,
    },
    "opus-haiku": {
        "answer": DEFAULT_HEAVY_MODEL,
        "analysis": DEFAULT_LIGHT_MODEL,
    },
    "sonnet-haiku": {
        "answer": DEFAULT_MEDIUM_MODEL,
        "analysis": DEFAULT_LIGHT_MODEL,
    },
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def elapsed(trace: PipelineTrace) -> float:
    return round(sum(stage.seconds for stage in trace.stages), 3)


def page_hit(question: Question, trace: PipelineTrace) -> bool | None:
    if question.expect_refusal:
        return None
    for hit in trace.result.hits:
        start = int(hit.source.get("page", 0))
        end = int(hit.source.get("end_page", start))
        if any(start <= page <= end for page in question.pages):
            return True
    return False


def cited_page_hit(question: Question, trace: PipelineTrace) -> bool | None:
    if question.expect_refusal:
        return None
    cited = set(trace.answer.cited)
    return any(
        block.n in cited and block.page in question.pages
        for block in trace.answer.context.blocks
    )


def row_for(question: Question, config: str, trace: PipelineTrace) -> dict:
    events = trace.events
    errors = [event["error"] for event in events if event.get("error")]
    score = None
    if not question.expect_refusal:
        score = score_query(question, [hit.source for hit in trace.result.hits], 10).model_dump()
    answered = bool(trace.answer.text)
    refused = trace.answer.refusal is not None
    pipeline_pass = refused if question.expect_refusal else answered
    return {
        "id": question.id,
        "bucket": question.bucket,
        "doc_id": question.doc_id,
        "question": question.query,
        "expected_pages": question.pages,
        "expected_refusal": question.expect_refusal,
        "config": config,
        "models": {**CONFIGS[config], "vision": DEFAULT_MEDIUM_MODEL},
        "seconds": elapsed(trace),
        "answered": answered,
        "refused": refused,
        "refusal": trace.answer.refusal,
        "pipeline_pass": pipeline_pass,
        "page_hit": page_hit(question, trace),
        "cited_page_hit": cited_page_hit(question, trace),
        "retrieval_score": score,
        "calls": trace.calls,
        "input_tokens": sum(int(event.get("input_tokens", 0)) for event in events),
        "output_tokens": sum(int(event.get("output_tokens", 0)) for event in events),
        "repair_count": trace.answer.citation_retries,
        "model_errors": errors,
        "truncated": any("Truncated" in error for error in errors),
        "error": trace.error,
        "answer": trace.answer.text,
        "citations": [
            block.model_dump()
            for block in trace.answer.context.blocks
            if block.n in trace.answer.cited
        ],
        "trace": trace.model_dump(mode="json"),
    }


def run(root: Path, config: str, *, all_questions: bool = False, limit: int | None = None) -> None:
    output = root / f"{config}.jsonl"
    completed = set()
    if output.exists():
        completed = {json.loads(line)["id"] for line in output.read_text().splitlines() if line}
    gold = Goldset.load(GOLD)
    questions = (
        gold.questions
        if all_questions
        else [q for q in gold.questions if q.id in PILOT_IDS]
    )
    if not all_questions:
        by_id = {q.id: q for q in questions}
        questions = [by_id[qid] for qid in PILOT_IDS]
    os_client = client()
    known = sorted(doc_counts(os_client))
    gold.verify(known)
    settings = CONFIGS[config]
    executed = 0
    try:
        with ExitStack() as stack:
            answer = stack.enter_context(Bedrock(settings["answer"], num_ctx=16384))
            analysis = stack.enter_context(Bedrock(settings["analysis"], num_ctx=16384))
            vision = stack.enter_context(Bedrock(DEFAULT_MEDIUM_MODEL, num_ctx=16384))
            for index, question in enumerate(questions, 1):
                if question.id in completed:
                    continue
                started = time.monotonic()
                try:
                    trace = run_pipeline(
                        os_client,
                        answer,
                        analysis,
                        question.query,
                        known_docs=known,
                        explicit_docs=[question.doc_id],
                        vision=vision,
                        cache=None,
                    )
                    row = row_for(question, config, trace)
                except Exception as exc:
                    row = {
                        "id": question.id,
                        "bucket": question.bucket,
                        "doc_id": question.doc_id,
                        "question": question.query,
                        "config": config,
                        "models": {**settings, "vision": DEFAULT_MEDIUM_MODEL},
                        "seconds": round(time.monotonic() - started, 3),
                        "pipeline_pass": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                status = "pass" if row.get("pipeline_pass") else row.get("error") or "fail"
                print(
                    f"{config} {index}/{len(questions)} {question.id} "
                    f"{row['seconds']:.2f}s {status}",
                    flush=True,
                )
                executed += 1
                if limit is not None and executed >= limit:
                    return
    finally:
        os_client.close()
    write_json(
        root / "manifest.json",
        {
            "created": datetime.now(UTC).isoformat(),
            "goldset": str(GOLD),
            "pilot_ids": list(PILOT_IDS),
            "configs": CONFIGS,
            "vision": DEFAULT_MEDIUM_MODEL,
            "cache": False,
            "explicit_document": True,
        },
    )


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, int(len(ordered) * 0.95 + 0.999999) - 1)]


def summarize(root: Path) -> None:
    summary = {}
    for config in CONFIGS:
        path = root / f"{config}.jsonl"
        if not path.exists():
            continue
        rows = [
            row
            for line in path.read_text().splitlines()
            if line
            for row in [json.loads(line)]
            if row["id"] in PILOT_IDS
        ]
        timings = [float(row["seconds"]) for row in rows]
        ranked = [row for row in rows if not row.get("expected_refusal", False)]
        summary[config] = {
            "n": len(rows),
            "pipeline_pass_rate": statistics.fmean(bool(r.get("pipeline_pass")) for r in rows),
            "answer_rate": statistics.fmean(bool(r.get("answered")) for r in rows),
            "expected_refusal_pass_rate": statistics.fmean(
                bool(r.get("refused")) for r in rows if r.get("expected_refusal", False)
            ),
            "page_hit_rate": statistics.fmean(bool(r.get("page_hit")) for r in ranked),
            "cited_page_hit_rate": statistics.fmean(bool(r.get("cited_page_hit")) for r in ranked),
            "mean_seconds": statistics.fmean(timings),
            "p95_seconds": percentile95(timings),
            "mean_calls": statistics.fmean(float(r.get("calls", 0)) for r in rows),
            "input_tokens": sum(int(r.get("input_tokens", 0)) for r in rows),
            "output_tokens": sum(int(r.get("output_tokens", 0)) for r in rows),
            "repairs": sum(int(r.get("repair_count", 0)) for r in rows),
            "truncated": sum(bool(r.get("truncated")) for r in rows),
            "errors": [r["id"] for r in rows if r.get("error")],
            "failures": [r["id"] for r in rows if not r.get("pipeline_pass")],
        }
    write_json(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "run-all", "summarize"))
    parser.add_argument("--config", choices=CONFIGS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.command == "run":
        if not args.config:
            parser.error("run에는 --config가 필요합니다")
        run(args.root, args.config, all_questions=args.all_questions, limit=args.limit)
    elif args.command == "run-all":
        for config in CONFIGS:
            run(args.root, config, all_questions=args.all_questions, limit=args.limit)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()

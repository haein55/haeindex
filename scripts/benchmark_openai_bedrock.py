"""Compare OpenAI gpt-oss with Claude through the same Bedrock RAG pipeline.

All runs use the same index, explicit documents, prompts, token caps, and Sonnet
vision. No LLM cache or index writes. Semantic grades are separate from runtime
success; a model error never counts as a correct absence response.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import yaml

from haeindex.answer import Refusal
from haeindex.bedrock import DEFAULT_MEDIUM_MODEL, Bedrock
from haeindex.evaluate import Goldset
from haeindex.index import INDEX, client, doc_counts
from haeindex.pipeline import run_pipeline

ROOT = Path("work/provider-comparison-20260921")
PILOT_IDS = [
    "bid-01",
    "cam-05",
    "cam-01",
    "rule-12",
    "absent-01",
    "mohw-01",
    "bid-03",
    "cert-01",
    "paper-03",
    "absent-09",
]
OSS120 = "openai.gpt-oss-120b-1:0"
OSS20 = "openai.gpt-oss-20b-1:0"
CONFIGS = {
    "sonnet": (DEFAULT_MEDIUM_MODEL, DEFAULT_MEDIUM_MODEL),
    "oss120": (OSS120, OSS120),
    "oss20": (OSS20, OSS20),
    "oss120-answer": (OSS120, DEFAULT_MEDIUM_MODEL),
    "sonnet-oss120": (DEFAULT_MEDIUM_MODEL, OSS120),
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_for(root, config):
    path = root / f"{config}.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line]
        if path.exists()
        else []
    )


def measurements(q, trace, *, expected_absent):
    answered = bool(trace.answer.text)
    proper_refusal = trace.answer.refusal in {Refusal.INSUFFICIENT_EVIDENCE, Refusal.NO_HITS}
    source_hits = {h.chunk_id: h.source for h in trace.result.hits}

    def page_matches(source):
        start = int(source.get("page", 0))
        end = int(source.get("end_page", start))
        return source.get("doc_id") == q.doc_id and any(start <= p <= end for p in q.pages)

    cited = [b for b in trace.answer.context.blocks if b.n in trace.answer.cited]
    # Spans include measured token usage even when a truncated output raises before
    # TaskRunner can copy the usage into its event. These are generation tokens only.
    spans = [s for s in trace.spans if s["category"] == "bedrock.chat"]
    return {
        "expected_absent": expected_absent,
        "answered": answered,
        "refusal": trace.answer.refusal,
        "runtime_pass": (proper_refusal and not trace.error) if expected_absent else answered,
        "page_hit": None if expected_absent else any(page_matches(s) for s in source_hits.values()),
        "cited_page_hit": None
        if expected_absent
        else any(page_matches(source_hits.get(b.chunk_id, b.model_dump())) for b in cited),
        "calls": trace.calls,
        "input_tokens": sum(s.get("input_tokens", 0) for s in spans),
        "output_tokens": sum(s.get("output_tokens", 0) for s in spans),
        "model_errors": [e["error"] for e in trace.events if e.get("error")],
        "error": trace.error,
        "answer": trace.answer.text,
        "citations": [b.model_dump() for b in cited],
        "trace": trace.model_dump(mode="json"),
    }


def run(root, configs, limit):
    root.mkdir(parents=True, exist_ok=True)
    sources = [
        *sorted(Path("haeindex").glob("*.py")),
        Path(__file__),
        Path("goldset/all.yaml"),
        Path("goldset/answer-rubric.yaml"),
    ]
    hashes = {str(p): digest(p) for p in sources}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["files"] != hashes:
            raise ValueError("실행 코드/평가 기준이 변경됐습니다. 새 --root를 사용하세요.")
    else:
        manifest = {
            "created": datetime.now(UTC).isoformat(),
            "files": hashes,
            "configs": CONFIGS,
            "pilot_ids": PILOT_IDS,
            "vision": DEFAULT_MEDIUM_MODEL,
            "num_ctx": 16384,
            "cache": False,
            "explicit_document": True,
            "max_calls": 18,
            "index": INDEX,
            "reasoning": "provider default",
        }
    gold = Goldset.load(Path("goldset/all.yaml"))
    rubric = yaml.safe_load(Path("goldset/answer-rubric.yaml").read_text())
    by_id = {q.id: q for q in gold.questions}
    done = {config: {r["id"] for r in rows_for(root, config)} for config in configs}
    with ExitStack() as stack:
        os_client = client()
        stack.callback(os_client.close)
        counts = doc_counts(os_client)
        gold.verify(sorted(counts))
        stats = os_client.indices.stats(index=INDEX, metric="docs", level="shards")
        state = stats["indices"][INDEX]
        index_state = {
            "uuid": state["uuid"],
            "docs": state["primaries"]["docs"],
            "seq_no": {
                shard: copy.get("seq_no")
                for shard, copies in state["shards"].items()
                for copy in copies
                if copy["routing"]["primary"]
            },
        }
        if "index_state" in manifest and manifest["index_state"] != index_state:
            raise ValueError("색인 상태가 달라졌습니다. 새 --root를 사용하세요.")
        manifest["index_state"] = index_state
        manifest["documents"] = counts
        dump(manifest_path, manifest)
        clients = {
            model: stack.enter_context(Bedrock(model, num_ctx=16384))
            for model in {DEFAULT_MEDIUM_MODEL, *(m for c in configs for m in CONFIGS[c])}
        }
        # Rotate order by question so a provider isn't always tested first.
        for number, qid in enumerate(PILOT_IDS[:limit]):
            question = by_id[qid]
            order = configs[number % len(configs) :] + configs[: number % len(configs)]
            for config in order:
                if qid in done[config]:
                    continue
                answer_model, analysis_model = CONFIGS[config]
                started = time.monotonic()
                print(f"START {config} {qid}", flush=True)
                trace = run_pipeline(
                    os_client,
                    clients[answer_model],
                    clients[analysis_model],
                    question.query,
                    known_docs=sorted(counts),
                    explicit_docs=[question.doc_id],
                    vision=clients[DEFAULT_MEDIUM_MODEL],
                    cache=None,
                    progress=lambda name, detail: print(f"  {name}", flush=True),
                )
                row = {
                    "id": qid,
                    "config": config,
                    "bucket": question.bucket,
                    "question": question.query,
                    "doc_id": question.doc_id,
                    "expected_pages": question.pages,
                    "criterion": rubric["criteria"].get(qid, "문서에 근거가 없어야 거부가 정답"),
                    "models": {
                        "answer": answer_model,
                        "analysis": analysis_model,
                        "vision": DEFAULT_MEDIUM_MODEL,
                    },
                    "seconds": round(time.monotonic() - started, 3),
                    **measurements(
                        question, trace, expected_absent=qid in rubric["expected_absent"]
                    ),
                }
                with (root / f"{config}.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"DONE {config} {qid}: {row['seconds']}s "
                    f"{'answer' if row['answered'] else row['refusal']} calls={row['calls']}",
                    flush=True,
                )


def summarize(root):
    summary = {}
    for config in CONFIGS:
        rows = rows_for(root, config)
        if not rows:
            continue
        times = sorted(r["seconds"] for r in rows)
        summary[config] = {
            "n": len(rows),
            "runtime_pass": sum(r["runtime_pass"] for r in rows),
            "answers": sum(r["answered"] for r in rows),
            "mean_seconds": round(statistics.mean(times), 2),
            "p95_seconds": times[-1],
            "calls": sum(r["calls"] for r in rows),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "failures": {
                r["id"]: r["error"] or r["refusal"] for r in rows if not r["runtime_pass"]
            },
            "semantic_accuracy": "requires blind grading",
        }
    dump(root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def blind(root):
    groups = {}
    for config in CONFIGS:
        for row in rows_for(root, config):
            groups.setdefault(row["id"], []).append((config, row))
    key, records = {}, []
    rng = random.Random(20260921)
    for qid, group in groups.items():
        rng.shuffle(group)
        for i, (config, row) in enumerate(group):
            label = f"{qid}:{chr(65 + i)}"
            key[label] = config
            records.append(
                {
                    "label": label,
                    **{
                        k: row[k]
                        for k in (
                            "question",
                            "doc_id",
                            "criterion",
                            "expected_absent",
                            "answer",
                            "refusal",
                            "error",
                            "citations",
                        )
                    },
                }
            )
    dump(root / "blind-key.json", key)
    dump(root / "blind-answers.json", records)
    print(f"Exported {len(records)} blind answers")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "summarize", "blind"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--configs", nargs="+", choices=CONFIGS, default=["sonnet", "oss120", "oss20"]
    )
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=10)
    args = parser.parse_args()
    if args.command == "run":
        run(args.root, args.configs, args.limit)
    elif args.command == "blind":
        blind(args.root)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()

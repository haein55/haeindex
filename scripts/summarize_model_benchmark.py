"""모델명을 가린 판정을 실행 결과와 결합해 비교표와 답변 기록을 저장한다."""

import csv
import json
import math
import random
import statistics
from pathlib import Path

import yaml

from scripts.benchmark_models import MODELS, ROOT, write_json

DEST = Path("docs/benchmarks/2026-09-14")
GRADES = DEST / "blind-grades.yaml"
ALLOWED = {"correct", "partial", "incorrect", "error"}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def bootstrap_difference(left, right):
    rng = random.Random(42)
    diffs = [a - b for a, b in zip(left, right, strict=True)]
    means = [statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000)]
    return [percentile(means, 0.025), percentile(means, 0.975)]


def main():
    key = json.loads((ROOT / "blind_key.json").read_text())
    grades = yaml.safe_load(GRADES.read_text())
    questions = [r["question"] for r in json.loads((ROOT / "candidates.json").read_text())]
    rubric = yaml.safe_load(Path("goldset/answer-rubric.yaml").read_text())
    if set(grades) != {q["id"] for q in questions}:
        raise ValueError("판정 문항이 골든셋과 일치하지 않습니다")
    judged = {}
    for qid, answers in grades.items():
        if set(answers) != set("ABCDE"):
            raise ValueError(f"{qid}: 모델 다섯 개 판정이 필요합니다")
        for label, grade in answers.items():
            if grade["grade"] not in ALLOWED or not grade["reason"].strip():
                raise ValueError(f"{qid}:{label}: 유효한 판정과 이유가 필요합니다")
            judged[qid, key[f"{qid}:{label}"]] = grade
    summary, outputs, vectors, details = [], [], {}, {}
    for model in MODELS:
        name = model.replace(":", "_")
        rows = [json.loads(s) for s in (ROOT / f"{name}.jsonl").read_text().splitlines()]
        by_id = {r["id"]: r for r in rows}
        if len(rows) != len(questions) or set(by_id) != {q["id"] for q in questions}:
            raise ValueError(f"{model}: 실행 결과가 누락되거나 중복되었습니다")
        vectors[model] = [int(judged[q["id"], model]["grade"] == "correct") for q in questions]
        times = [r["model_seconds"] for r in rows]
        absent = [q for q in questions if q["id"] in rubric["expected_absent"]]
        answerable = [q for q in questions if q["id"] not in rubric["expected_absent"]]
        answered = [r for r in rows if r.get("answer", {}).get("text")]
        kinds = [judged[q["id"], model]["grade"] for q in questions]
        detail = json.loads((ROOT / f"{name}.model.json").read_text())
        details[model] = detail
        tag = detail.get("tag", {})
        model_detail = detail.get("details") or {}
        summary.append({
            "model": model, "size_GB": round(tag.get("size", 0) / 1e9, 2),
            "quantization": model_detail.get("quantization_level", "managed"),
            "correct": kinds.count("correct"), "total": len(questions),
            "accuracy_pct": round(100 * kinds.count("correct") / len(questions), 2),
            "partial": kinds.count("partial"), "incorrect": kinds.count("incorrect"),
            "citation_issue_questions": sum(
                judged[q["id"], model].get("citation_issue", False) for q in questions
            ),
            "errors": kinds.count("error"),
            "absent_correct": sum(judged[q["id"], model]["grade"] == "correct" for q in absent),
            "absent_total": len(absent), "answers_returned": len(answered),
            "answerable_correct": sum(
                judged[q["id"], model]["grade"] == "correct" for q in answerable
            ),
            "answerable_total": len(answerable),
            "false_refusals": sum(
                bool(by_id[q["id"]].get("answer", {}).get("refusal")) for q in answerable
            ),
            "low_confidence_questions": sum(
                r.get("answer", {}).get("refusal") == "low_confidence" for r in rows
            ),
            "mean_model_seconds": round(statistics.mean(times), 2),
            "p95_model_seconds": round(percentile(times, 0.95), 2),
            "mean_search_seconds": round(statistics.mean(r["retrieval_seconds"] for r in rows), 3),
            "mean_llm_calls": round(statistics.mean(len(r["calls"]) for r in rows), 2),
            "language_retry_questions": sum(
                r.get("answer", {}).get("language_retries", 0) > 0 for r in rows
            ),
            "language_failures": sum(
                r.get("answer", {}).get("refusal") == "language_mismatch" for r in rows
            ),
            "rerank_parse_fails": sum(r.get("rerank_parse_fails", 0) for r in rows),
            "ndcg_at_5": round(statistics.mean(
                r.get("retrieval_score", {}).get("ndcg", 0) for r in rows if r["bucket"] != "E"
            ), 4),
        })
        for q in questions:
            row = by_id[q["id"]]
            ans = row.get("answer", {})
            outputs.append({
                "id": q["id"], "question": q["query"], "model": model,
                **judged[q["id"], model], "text": ans.get("text", ""),
                "refusal": ans.get("refusal"), "error": row.get("error"),
                "sources": [{k: b[k] for k in ["n", "doc_id", "page", "label"]}
                            for b in ans.get("context", {}).get("blocks", [])
                            if b["n"] in ans.get("cited", [])],
            })
    summary.sort(key=lambda s: (-s["correct"], s["mean_model_seconds"]))
    winner = summary[0]["model"]
    write_json(DEST / "summary.json", {
        "summary": summary, "observed_winner": winner,
        "paired_bootstrap_accuracy_difference_95ci": {
            m: bootstrap_difference(vectors[winner], v) for m, v in vectors.items() if m != winner
        },
    })
    write_json(DEST / "answers.json", outputs)
    write_json(DEST / "models.json", details)
    manifest = json.loads((ROOT / "manifest.json").read_text())
    revision = ROOT / "comparison-revision.json"
    write_json(DEST / "manifest.json", {
        **manifest, "prepared_models": manifest["models"], "models": MODELS,
        "comparison_revision": json.loads(revision.read_text()) if revision.exists() else None,
    })
    with (DEST / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

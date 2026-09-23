"""Combine independently graded answers with measured latency and Bedrock cost."""

import argparse
import json
import statistics
from pathlib import Path

ROOTS = [
    Path("work/provider-comparison-20260921"),
    Path("work/provider-frontier-comparison-20260921"),
]
DEST = Path("docs/benchmarks/2026-09-21")
# USD per million tokens; US geographic / regional Standard tier.
PRICES = {
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0": (3.30, 16.50, 0.33, 4.125),
    "openai.gpt-oss-120b-1:0": (0.15, 0.60, 0.15, 0.15),
    "openai.gpt-oss-20b-1:0": (0.07, 0.30, 0.07, 0.07),
    "us.openai.gpt-5.6-terra": (2.20, 13.20, 0.22, 2.75),
    "us.openai.gpt-5.6-sol": (4.40, 22.00, 0.44, 5.50),
    "us.openai.gpt-6-astra": (11.00, 55.00, 1.10, 13.75),
}


def generation_cost(row):
    total = 0
    for span in row["trace"]["spans"]:
        if span["category"] != "bedrock.chat":
            continue
        input_rate, output_rate, read_rate, write_rate = PRICES[span["model"]]
        read, write = span.get("cached_input_tokens", 0), span.get("cache_write_tokens", 0)
        base = max(0, span.get("input_tokens", 0) - read - write)
        total += (
            base * input_rate
            + read * read_rate
            + write * write_rate
            + span.get("output_tokens", 0) * output_rate
        ) / 1e6
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    grade_path = DEST / "openai-grades.json"
    grades = json.loads(grade_path.read_text()) if grade_path.exists() else {}
    summaries, answers = [], []
    for root in ROOTS:
        for path in sorted(root.glob("*.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            if args.pilot:
                rows = [
                    r
                    for r in rows
                    if r["id"] in {"bid-01", "cam-05", "cam-01", "rule-12", "absent-01"}
                ]
            if not rows:
                continue
            config = path.stem
            judged = [grades.get(f"{config}:{r['id']}") for r in rows]
            complete = all(judged)
            times = sorted(r["seconds"] for r in rows)
            costs = [generation_cost(r) for r in rows]
            summaries.append(
                {
                    "config": config,
                    "n": len(rows),
                    "graded": sum(g is not None for g in judged),
                    "correct": sum(g["grade"] == "correct" for g in judged) if complete else None,
                    "runtime_pass": sum(r["runtime_pass"] for r in rows),
                    "mean_seconds": round(statistics.mean(times), 2),
                    "p95_seconds": times[-1],
                    "mean_generation_usd": round(statistics.mean(costs), 5),
                    "total_generation_usd": round(sum(costs), 5),
                    "input_tokens": sum(r["input_tokens"] for r in rows),
                    "output_tokens": sum(r["output_tokens"] for r in rows),
                }
            )
            for row, grade, cost in zip(rows, judged, costs, strict=True):
                answers.append(
                    {
                        k: row[k]
                        for k in [
                            "config",
                            "id",
                            "question",
                            "criterion",
                            "answer",
                            "refusal",
                            "error",
                            "seconds",
                        ]
                    }
                    | {
                        "grade": grade,
                        "generation_usd": round(cost, 6),
                        "citations": [
                            {k: c[k] for k in ["n", "doc_id", "page", "chunk_id"]}
                            for c in row["citations"]
                        ],
                    }
                )
    result = {
        "summary": summaries,
        "prices_per_million": PRICES,
        "cost_scope": "generation including Sonnet vision; excludes Titan, hosting, tax",
        "cost_sources": [
            "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json",
            "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrockFoundationModels/current/us-east-1/index.json",
            "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-terra.html",
            "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html",
            "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html",
        ],
    }
    prefix = "openai-pilot" if args.pilot else "openai"
    for filename, value in [
        (f"{prefix}-summary.json", result),
        (f"{prefix}-answers.json", answers),
    ]:
        (DEST / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Summarize actual measured usage, including tokens spent on failed attempts."""

import json
import statistics
from pathlib import Path

from scripts.benchmark_auxiliary_roles import ROOT, dump, read

# USD / 1M tokens, AWS us-east-1 regional standard pricing, 2026-09-21.
RATES = {
    "haiku": (1.1, 5.5, 0.11, 1.375),
    "sonnet": (3.3, 16.5, 0.33, 4.125),
    "terra": (2.2, 13.2, 0.22, 2.75),
    "sol": (4.4, 22, 0.44, 5.5),
}


def usage(rows, name):
    spans = [s for row in rows for s in row["spans"] if s["category"] == "bedrock.chat"]
    inp, out, cached, written = RATES[name]
    usd = (
        sum(
            max(
                0,
                s.get("input_tokens", 0)
                - s.get("cached_input_tokens", 0)
                - s.get("cache_write_tokens", 0),
            )
            * inp
            + s.get("output_tokens", 0) * out
            + s.get("cached_input_tokens", 0) * cached
            + s.get("cache_write_tokens", 0) * written
            for s in spans
        )
        / 1e6
    )
    return {
        "jobs": len(rows),
        "calls": sum(r["calls"] for r in rows),
        "failed_jobs": sum(bool(r.get("error")) for r in rows),
        "truncated_attempts": sum("Truncated" in e["error"] for r in rows for e in r["events"]),
        "mean_seconds": statistics.fmean(r["seconds"] for r in rows),
        "input_tokens": sum(s.get("input_tokens", 0) for s in spans),
        "output_tokens": sum(s.get("output_tokens", 0) for s in spans),
        "estimated_usd": usd,
    }


def summarize(root):
    result = {}
    manifest = read(root / "manifest.json")
    documents = manifest["enrich_docs"]
    chunk_count = sum(manifest["counts"][d] for d in documents)
    batch_count = sum((manifest["counts"][d] + 2) // 3 for d in documents)
    for name in ["haiku", "terra", "sol"]:
        rows = [read(p) for p in (root / "enrich" / name).glob("*.json")]
        cards = [read(p) for p in (root / "cards" / name).glob("*.json")]
        if len(rows) != batch_count or len(cards) != len(documents):
            raise ValueError(f"Incomplete expected chunk batches and card documents: {name}")
        accepted = [x for row in rows for x in row.get("accepted", [])]
        result[name] = {
            "chunks": chunk_count,
            "accepted": len({x["chunk_id"] for x in accepted}),
            "rules": sum(len(x["rules"]) for x in accepted),
            "chunks_usage": usage(rows, name),
            "cards_usage": usage(cards, name),
            "section_cards": sum(len(c.get("sections", [])) for c in cards),
            "document_cards": sum("document" in c for c in cards),
        }
        result[name]["total_usd"] = (
            result[name]["chunks_usage"]["estimated_usd"]
            + result[name]["cards_usage"]["estimated_usd"]
        )
    dump(root / "enrichment-generation-summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Refresh the vision cost estimate with any automatic cache usage accounted for.
    vision = read(root / "vision-summary.json")
    for name in RATES:
        rows = [read(p) for p in (root / "vision" / name).glob("*.json")]
        measured = usage(rows, name)
        vision[name].update(
            estimated_usd=measured["estimated_usd"], mean_usd=measured["estimated_usd"] / len(rows)
        )
    dump(root / "vision-summary.json", vision)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    summarize(parser.parse_args().root)

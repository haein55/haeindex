"""Use the shared benchmark with Bedrock-hosted OpenAI frontier models."""

import argparse
import json
from pathlib import Path

from haeindex.bedrock import DEFAULT_MEDIUM_MODEL, Bedrock
from scripts import benchmark_openai_bedrock as benchmark
from scripts.bedrock_responses import BedrockResponses

ROOT = Path("work/provider-frontier-comparison-20260921")
TERRA = "us.openai.gpt-5.6-terra"
SOL = "us.openai.gpt-5.6-sol"
ASTRA = "us.openai.gpt-6-astra"
CONFIGS = {
    "sonnet": (DEFAULT_MEDIUM_MODEL, DEFAULT_MEDIUM_MODEL),
    "terra": (TERRA, TERRA),
    "sol": (SOL, SOL),
    "astra": (ASTRA, ASTRA),
    "astra-sonnet": (ASTRA, DEFAULT_MEDIUM_MODEL),
    "astra-terra": (ASTRA, TERRA),
    "sol-terra": (SOL, TERRA),
    "sonnet-terra": (DEFAULT_MEDIUM_MODEL, TERRA),
}


def model_client(model, **kwargs):
    cls = BedrockResponses if model.startswith("us.openai.") else Bedrock
    return cls(model, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "summarize", "blind"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=["terra", "sol", "astra"])
    parser.add_argument("--limit", type=int, choices=range(1, 11), default=10)
    args = parser.parse_args()
    benchmark.CONFIGS = CONFIGS
    benchmark.Bedrock = model_client
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "adapter-manifest.json"
    adapter_manifest = {
        "files": {
            str(p): benchmark.digest(p)
            for p in [Path(__file__), Path("scripts/bedrock_responses.py")]
        },
        "provider": "AWS Bedrock",
        "api": "Responses",
        "configs": CONFIGS,
        "reasoning": {"terra": "none", "sol": "none", "astra": "low"},
        "schema": "native strict JSON schema; same instructions and Pydantic validation",
        "vision": DEFAULT_MEDIUM_MODEL,
    }
    if args.command == "run":
        if (
            manifest_path.exists()
            and json.loads(manifest_path.read_text())["files"] != adapter_manifest["files"]
        ):
            raise ValueError("Adapter changed; use a new --root")
        benchmark.dump(manifest_path, adapter_manifest)
        benchmark.run(args.root, args.configs, args.limit)
    elif args.command == "blind":
        benchmark.blind(args.root)
    else:
        benchmark.summarize(args.root)


if __name__ == "__main__":
    main()

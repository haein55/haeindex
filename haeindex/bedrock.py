"""AWS Bedrock Converse and Titan embedding client."""

import base64
import json
import math
import os
import re
import unicodedata
from collections.abc import Sequence
from typing import Any

import boto3
from pydantic import BaseModel

DEFAULT_REGION = "us-east-1"
DEFAULT_HEAVY_MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"
DEFAULT_MEDIUM_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_LIGHT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
EMBED_DIM = 1024
_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[­​-‏﻿]")


def normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _ZERO_WIDTH.sub("", normalized)
    return _WS.sub(" ", normalized).strip()


class ChatResult(BaseModel):
    content: str
    done_reason: str | None = None


class Truncated(RuntimeError):
    pass


class BedrockError(RuntimeError):
    pass


def bedrock_model(role: str, override: str | None = None) -> str:
    if override:
        return override
    defaults = {
        "answer": ("LLM_HEAVY_MODEL_ID", DEFAULT_HEAVY_MODEL),
        "analysis": ("LLM_MEDIUM_MODEL_ID", DEFAULT_MEDIUM_MODEL),
        "vision": ("LLM_MEDIUM_MODEL_ID", DEFAULT_MEDIUM_MODEL),
        "enrich": ("LLM_LIGHT_MODEL_ID", DEFAULT_LIGHT_MODEL),
    }
    rocket_env, default = defaults.get(role, defaults["analysis"])
    return os.environ.get(f"HAEINDEX_BEDROCK_{role.upper()}_MODEL") or os.environ.get(
        rocket_env, default
    )


def bedrock_embedding_model() -> str:
    return (
        os.environ.get("HAEINDEX_BEDROCK_EMBED_MODEL")
        or os.environ.get("EMBEDDING_MODEL_ID")
        or DEFAULT_EMBED_MODEL
    )


class Bedrock:
    """Bedrock Converse + Titan embedding adapter.

    Model IDs stay explicit because availability differs by AWS account and region.
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        region: str | None = None,
        embed_model: str | None = None,
        embed_dim: int = EMBED_DIM,
        num_ctx: int = 16384,
    ) -> None:
        model_id = model_id or bedrock_model("answer")
        self.chat_model = model_id
        self.region = (
            region
            or os.environ.get("BEDROCK_REGION")
            or os.environ.get("AWS_BEDROCK_REGION")
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or DEFAULT_REGION
        )
        self.embed_model = embed_model or bedrock_embedding_model()
        self.embed_dim = embed_dim
        self.num_ctx = num_ctx
        # Rocket exposes the Bedrock bearer token under this shorter name.
        # Botocore's official provider reads AWS_BEARER_TOKEN_BEDROCK.
        if api_key := os.environ.get("BEDROCK_API_KEY"):
            os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", api_key)
        self._runtime = boto3.client("bedrock-runtime", region_name=self.region)

    def close(self) -> None:
        close = getattr(self._runtime, "close", None)
        if close:
            close()

    def __enter__(self) -> "Bedrock":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def model_digest(self) -> str:
        # Bedrock model/inference-profile IDs are versioned and stable cache identities.
        return f"bedrock:{self.region}:{self.chat_model}"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for raw in texts:
            text = normalize(raw)
            if not text:
                vectors.append([0.0] * self.embed_dim)
                continue
            try:
                response = self._runtime.invoke_model(
                    modelId=self.embed_model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(
                        {"inputText": text, "dimensions": self.embed_dim, "normalize": True}
                    ),
                )
                data = json.loads(response["body"].read())
                vector = data["embedding"]
            except Exception as exc:  # boto3 exposes service-specific generated exceptions.
                raise BedrockError(f"Bedrock 임베딩 실패: {exc}") from exc
            if len(vector) != self.embed_dim:
                raise BedrockError(
                    f"Bedrock 임베딩 차원이 {len(vector)}이고 {self.embed_dim}를 기대했습니다"
                )
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors

    def embed_batched(self, texts: Sequence[str], batch: int = 16) -> list[list[float]]:
        # Titan text embeddings accept one input per request.
        return self.embed(texts)

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float = 0.0,
        seed: int = 0,
        num_predict: int = 512,
        response_schema: dict[str, Any] | None = None,
    ) -> ChatResult:
        del seed  # Converse has no portable seed option.
        system, converted = [], []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "system":
                system.append({"text": content})
                continue
            blocks: list[dict[str, Any]] = [{"text": content}]
            for encoded in message.get("images", []):
                try:
                    blocks.append(
                        {"image": {"format": "png", "source": {"bytes": base64.b64decode(encoded)}}}
                    )
                except (ValueError, TypeError) as exc:
                    raise BedrockError("Bedrock 이미지 입력이 올바른 base64가 아닙니다") from exc
            converted.append(
                {"role": "assistant" if role == "assistant" else "user", "content": blocks}
            )
        if response_schema:
            system.append(
                {"text": "반환할 JSON Schema:\n" + json.dumps(response_schema, ensure_ascii=False)}
            )
        try:
            response = self._runtime.converse(
                modelId=self.chat_model,
                system=system,
                messages=converted,
                inferenceConfig={"temperature": temperature, "maxTokens": num_predict},
            )
        except Exception as exc:
            raise BedrockError(f"Bedrock Converse 실패: {exc}") from exc
        stop = response.get("stopReason")
        if stop == "max_tokens":
            raise Truncated(f"Bedrock 응답이 maxTokens={num_predict}에서 잘렸습니다")
        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block.get("text", "") for block in content)
        return ChatResult(content=text, done_reason=stop)

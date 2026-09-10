import math
import re
import unicodedata
from collections.abc import Sequence

import httpx
from pydantic import BaseModel

BASE_URL = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024
READ_TIMEOUT = 300.0
_WS = re.compile(r"\s+")
_ZERO_WIDTH = re.compile(r"[­​-‏﻿]")


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH.sub("", t)
    return _WS.sub(" ", t).strip()


class ChatResult(BaseModel):
    content: str
    done_reason: str | None = None


class Truncated(RuntimeError):
    pass


class Ollama:
    def __init__(
        self,
        base_url: str = BASE_URL,
        embed_model: str = EMBED_MODEL,
        embed_dim: int = EMBED_DIM,
        chat_model: str = CHAT_MODEL,
        num_ctx: int = 8192,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chat_model = chat_model
        self.num_ctx = num_ctx
        self._client = httpx.Client(timeout=httpx.Timeout(READ_TIMEOUT, connect=5.0))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Ollama":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        cleaned = [normalize(t) for t in texts]
        r = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embed_model, "input": cleaned, "keep_alive": "30m"},
        )
        r.raise_for_status()
        vectors = r.json()["embeddings"]

        out: list[list[float]] = []
        for text, vec in zip(cleaned, vectors, strict=True):
            if not text:
                out.append([0.0] * self.embed_dim)
                continue
            if len(vec) != self.embed_dim:
                raise ValueError(
                    f"임베딩 차원이 {len(vec)} 인데 {self.embed_dim} 를 기대했다. "
                    f"--embed-model {self.embed_model!r} 을 확인하라"
                )
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_batched(self, texts: Sequence[str], batch: int = 16) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            out.extend(self.embed(texts[i : i + batch]))
        return out

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int = 0,
        num_predict: int = 512,
    ) -> ChatResult:
        r = self._client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.chat_model,
                "messages": list(messages),
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                    "num_predict": num_predict,
                    "seed": seed,
                },
            },
        )
        r.raise_for_status()
        data = r.json()
        done = data.get("done_reason")
        if done == "length":
            raise Truncated(
                f"응답이 num_predict={num_predict} 에서 잘렸다. "
                "반쯤 나온 답을 성공으로 기록하면 손실이 안 보인다"
            )
        return ChatResult(content=data.get("message", {}).get("content", ""), done_reason=done)

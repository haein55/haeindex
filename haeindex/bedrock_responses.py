"""Bedrock-hosted OpenAI Responses client; embeddings continue to use Titan."""

import copy
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from haeindex.bedrock import Bedrock, BedrockError, ChatResult, Truncated
from haeindex.profiling import span


def strict_schema(schema):
    result = copy.deepcopy(schema)

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object":
                value["additionalProperties"] = False
                value["required"] = list(value.get("properties", {}))
            for child in value.values():
                visit(child)

    visit(result)
    return result


class BedrockResponses(Bedrock):
    def __init__(self, model, *, reasoning=None, **kwargs):
        super().__init__(model, **kwargs)
        self.reasoning = reasoning or ("low" if "gpt-6-astra" in model else "none")
        self._key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or os.environ.get("BEDROCK_API_KEY")
        if not self._key:
            raise ValueError("Bedrock Responses requires the existing Bedrock API key")

    def model_digest(self):
        return super().model_digest() + f":responses:{self.reasoning}:strict-v1"

    def _post(self, payload):
        request = Request(
            f"https://bedrock-runtime.{self.region}.amazonaws.com/openai/v1/responses",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=180) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                error = json.loads(exc.read()).get("error", {})
                code = error.get("code") or error.get("type") or "http_error"
                param = error.get("param") or ""
            except (ValueError, AttributeError):
                code, param = "http_error", ""
            raise BedrockError(
                f"Bedrock Responses HTTP {exc.code}: {code} {param}".strip()
            ) from None
        except (URLError, TimeoutError) as exc:
            raise BedrockError(f"Bedrock Responses connection: {type(exc).__name__}") from None

    def chat(self, messages, *, temperature=0, seed=0, num_predict=512, response_schema=None):
        del seed
        system, converted = [], []
        for message in messages:
            content = str(message.get("content", ""))
            if message.get("role") == "system":
                system.append(content)
                continue
            role = "assistant" if message.get("role") == "assistant" else "user"
            blocks = [
                {"type": "output_text" if role == "assistant" else "input_text", "text": content}
            ]
            blocks.extend(
                {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}
                for encoded in message.get("images", [])
            )
            converted.append({"role": role, "content": blocks})
        payload = {
            "model": self.chat_model,
            "input": converted,
            "max_output_tokens": num_predict,
            "reasoning": {"effort": self.reasoning},
            "store": False,
        }
        if self.reasoning == "none":
            payload["temperature"] = temperature
        if response_schema:
            system.append("반환할 JSON Schema:\n" + json.dumps(response_schema, ensure_ascii=False))
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "pipeline_result",
                    "strict": True,
                    "schema": strict_schema(response_schema),
                }
            }
        payload["instructions"] = "\n".join(system)
        with span(
            "bedrock.chat",
            "Bedrock Responses",
            model=self.chat_model,
            reasoning=self.reasoning,
            max_tokens=num_predict,
        ) as metrics:
            response = self._post(payload)
            usage = response.get("usage") or {}
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            metrics.update(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cached_input_tokens": int(
                        (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
                    ),
                    "cache_write_tokens": int(
                        (usage.get("input_tokens_details") or {}).get("cache_write_tokens", 0)
                    ),
                    "reasoning_tokens": int(
                        (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                    ),
                    "resolved_model": response.get("model", self.chat_model),
                }
            )
            if response.get("status") == "incomplete":
                reason = (response.get("incomplete_details") or {}).get("reason", "unknown")
                if reason == "max_output_tokens":
                    raise Truncated(f"Bedrock Responses max_output_tokens={num_predict}")
                raise BedrockError(f"Bedrock Responses incomplete: {reason}")
            if response.get("status") != "completed":
                raise BedrockError(f"Bedrock Responses status: {response.get('status', 'missing')}")
            blocks = [
                block
                for item in response.get("output", [])
                if item.get("type") == "message"
                for block in item.get("content", [])
            ]
            if any(block.get("type") == "refusal" for block in blocks):
                raise BedrockError("Bedrock Responses model refused request")
            text = "".join(
                block.get("text", "") for block in blocks if block.get("type") == "output_text"
            )
            if not text.strip():
                raise BedrockError("Bedrock Responses returned no answer text")
        return ChatResult(
            content=text,
            done_reason="completed",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            server_latency_ms=0,
        )

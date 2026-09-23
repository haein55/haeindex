import base64
import io
import json

import pytest

from haeindex import bedrock, models
from haeindex.bedrock import Truncated
from haeindex.profiling import Profiler, reset_task, set_task


class Runtime:
    def __init__(self):
        self.requests = []
        self.stop = "end_turn"

    def converse(self, **request):
        self.requests.append(request)
        return {
            "stopReason": self.stop,
            "output": {"message": {"content": [{"text": '{"text":"ok"}'}]}},
            "usage": {"inputTokens": 12, "outputTokens": 4},
            "metrics": {"latencyMs": 345},
        }

    def invoke_model(self, **request):
        self.requests.append(request)
        return {
            "body": io.BytesIO(
                json.dumps({"embedding": [3.0, 4.0, 0.0], "inputTextTokenCount": 2}).encode()
            )
        }

    def close(self):
        pass


def test_bedrock_converse_preserves_system_schema_and_image(monkeypatch):
    runtime = Runtime()
    monkeypatch.setattr(bedrock.boto3, "client", lambda *a, **k: runtime)
    llm = bedrock.Bedrock("model-v1", embed_dim=3)
    encoded = base64.b64encode(b"png").decode()
    result = llm.chat(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "read", "images": [encoded]},
        ],
        response_schema={"type": "object"},
    )
    request = runtime.requests[0]
    assert result.content == '{"text":"ok"}'
    assert (result.input_tokens, result.output_tokens, result.server_latency_ms) == (12, 4, 345)
    assert request["modelId"] == "model-v1"
    assert request["system"][0]["text"] == "rules"
    assert "JSON Schema" in request["system"][1]["text"]
    assert request["messages"][0]["content"][1]["image"]["source"]["bytes"] == b"png"
    runtime.stop = "max_tokens"
    with pytest.raises(Truncated):
        llm.chat([{"role": "user", "content": "again"}], num_predict=10)


def test_bedrock_titan_embedding_is_normalized(monkeypatch):
    runtime = Runtime()
    monkeypatch.setattr(bedrock.boto3, "client", lambda *a, **k: runtime)
    llm = bedrock.Bedrock("model-v1", embed_dim=3)
    assert llm.embed(["hello", ""])[0] == pytest.approx([0.6, 0.8, 0.0])
    assert llm.embed([""])[0] == [0.0, 0.0, 0.0]


def test_bedrock_profiler_records_generation_embedding_tokens_and_task(monkeypatch):
    runtime = Runtime()
    monkeypatch.setattr(bedrock.boto3, "client", lambda *a, **k: runtime)
    llm = bedrock.Bedrock("model-v1", embed_dim=3)
    profiler = Profiler(stage="testing")
    profiler_token = profiler.activate()
    task_token = set_task("answer-draft")
    try:
        llm.chat([{"role": "user", "content": "hello"}])
        reset_task(task_token)
        task_token = None
        llm.embed(["hello"])
    finally:
        if task_token is not None:
            reset_task(task_token)
        profiler.deactivate(profiler_token)
    assert [event["category"] for event in profiler.spans] == [
        "bedrock.chat",
        "bedrock.embed",
    ]
    assert profiler.spans[0]["task"] == "answer-draft"
    assert profiler.spans[0]["input_tokens"] == 12
    assert profiler.spans[1]["input_tokens"] == 2


def test_bedrock_maps_rocket_api_key_to_aws_bearer_token(monkeypatch):
    runtime = Runtime()
    monkeypatch.setenv("BEDROCK_API_KEY", "rocket-token")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setattr(bedrock.boto3, "client", lambda *a, **k: runtime)

    bedrock.Bedrock("model-v1")

    assert bedrock.os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "rocket-token"


def test_model_client_uses_benchmarked_text_model_by_default(monkeypatch):
    monkeypatch.delenv("HAEINDEX_BEDROCK_ANSWER_MODEL", raising=False)
    monkeypatch.delenv("LLM_MEDIUM_MODEL_ID", raising=False)
    sentinel = object()
    captured = []
    monkeypatch.setattr(
        models, "BedrockResponses", lambda model, **kwargs: captured.append(model) or sentinel
    )
    assert models.model_client("answer") is sentinel
    assert captured == [bedrock.DEFAULT_TEXT_MODEL]


def test_model_client_preserves_vision_and_enrichment_models(monkeypatch):
    for name in (
        "LLM_MEDIUM_MODEL_ID",
        "LLM_LIGHT_MODEL_ID",
        "HAEINDEX_BEDROCK_VISION_MODEL",
        "HAEINDEX_BEDROCK_ENRICH_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    captured = []
    monkeypatch.setattr(models, "Bedrock", lambda model, **kwargs: captured.append(model))
    models.model_client("vision")
    models.model_client("enrich")
    assert captured == [bedrock.DEFAULT_MEDIUM_MODEL, bedrock.DEFAULT_LIGHT_MODEL]


def test_gpt_oss_still_uses_converse(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(models, "Bedrock", lambda *a, **k: sentinel)
    assert models.model_client("answer", model="openai.gpt-oss-120b-1:0") is sentinel


def test_model_client_always_uses_bedrock(monkeypatch):
    monkeypatch.setenv("HAEINDEX_BEDROCK_ANSWER_MODEL", "model-v1")
    sentinel = object()
    monkeypatch.setattr(models, "Bedrock", lambda *a, **k: sentinel)
    assert models.model_client("answer") is sentinel


def test_role_models_fall_back_to_required_answer_model(monkeypatch):
    monkeypatch.setenv("LLM_MEDIUM_MODEL_ID", "rocket-medium-v1")
    monkeypatch.delenv("HAEINDEX_BEDROCK_ANALYSIS_MODEL", raising=False)
    assert bedrock.bedrock_model("analysis") == "rocket-medium-v1"
    assert bedrock.bedrock_model("answer", "override-v2") == "override-v2"


def test_bedrock_indexes_are_isolated_from_previous_embedding_space():
    from haeindex.augmentation import SECTION_INDEX
    from haeindex.document_cards import DOC_INDEX
    from haeindex.index import INDEX

    assert INDEX == "haeindex_bedrock_v1"
    assert SECTION_INDEX == "haeindex_bedrock_sections_v1"
    assert DOC_INDEX == "haeindex_bedrock_docs_v1"

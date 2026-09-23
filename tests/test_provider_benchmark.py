from haeindex.answer import Answer, Refusal
from haeindex.evaluate import Question
from haeindex.pipeline import PipelineTrace
from haeindex.search import Hit, Result
from scripts.benchmark_openai_bedrock import measurements


def test_responses_preserves_schema_and_image_without_mutating_schema():
    from scripts.bedrock_responses import BedrockResponses

    model = object.__new__(BedrockResponses)
    model.chat_model = "us.openai.gpt-5.6-terra"
    model.reasoning = "none"
    requests = []

    def post(payload):
        requests.append(payload)
        return {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}
            ],
            "usage": {"input_tokens": 12, "output_tokens": 4},
        }

    model._post = post
    schema = {"type": "object", "properties": {"ok": {"type": "boolean", "default": False}}}
    reply = model.chat(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "read", "images": ["cG5n"]},
        ],
        response_schema=schema,
    )
    request = requests[0]
    assert request["input"][0]["content"][1]["image_url"] == "data:image/png;base64,cG5n"
    assert request["text"]["format"]["schema"]["required"] == ["ok"]
    assert "required" not in schema and "default" in schema["properties"]["ok"]
    assert request["store"] is False
    assert (reply.input_tokens, reply.output_tokens) == (12, 4)


def test_responses_truncation_is_not_accepted_as_a_completed_answer():
    import pytest

    from haeindex.bedrock import Truncated
    from scripts.bedrock_responses import BedrockResponses

    model = object.__new__(BedrockResponses)
    model.chat_model = "us.openai.gpt-6-astra"
    model.reasoning = "low"

    def post(payload):
        assert "temperature" not in payload
        return {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}

    model._post = post
    with pytest.raises(Truncated):
        model.chat([{"role": "user", "content": "question"}])


def test_model_error_is_not_a_correct_absence_refusal():
    q = Question(id="absent", doc_id="doc", bucket="E", query="missing")
    trace = PipelineTrace(
        question=q.query,
        result=Result(query=q.query),
        answer=Answer(refusal=Refusal.MODEL_FAILURE),
        error="invalid JSON",
    )
    assert not measurements(q, trace, expected_absent=True)["runtime_pass"]
    trace.answer = Answer(refusal=Refusal.INSUFFICIENT_EVIDENCE)
    trace.error = ""
    assert measurements(q, trace, expected_absent=True)["runtime_pass"]


def test_page_number_in_another_document_is_not_a_retrieval_hit():
    q = Question(id="fact", doc_id="wanted", bucket="A", query="fact", pages=[3])
    trace = PipelineTrace(
        question=q.query,
        result=Result(
            query=q.query,
            hits=[Hit(chunk_id="wrong#0", fused=1, source={"doc_id": "wrong", "page": 3})],
        ),
    )
    assert not measurements(q, trace, expected_absent=False)["page_hit"]


def test_failed_generation_tokens_are_counted_from_spans():
    q = Question(id="fact", doc_id="wanted", bucket="A", query="fact", pages=[3])
    trace = PipelineTrace(
        question=q.query,
        result=Result(query=q.query),
        spans=[
            {
                "category": "bedrock.chat",
                "input_tokens": 123,
                "output_tokens": 45,
                "error": "Truncated",
            },
            {"category": "bedrock.embed", "input_tokens": 99},
        ],
    )
    result = measurements(q, trace, expected_absent=False)
    assert (result["input_tokens"], result["output_tokens"]) == (123, 45)

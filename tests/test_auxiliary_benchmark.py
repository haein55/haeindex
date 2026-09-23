"""Guard measurement errors that can reverse the model selection."""

import io
import json
from types import SimpleNamespace

import pytest

from haeindex.evaluate import Question
from scripts.benchmark_auxiliary_roles import embedding_request
from scripts.evaluate_auxiliary_roles import aggregate, grade


@pytest.mark.parametrize("kind", ["search_query", "search_document"])
def test_cohere_preserves_query_document_mode(kind):
    requests = []

    def invoke_model(**kwargs):
        requests.append(kwargs)
        return {"body": io.BytesIO(json.dumps({"embeddings": {"float": [[2.0] * 1024]}}).encode())}

    llm = SimpleNamespace(_runtime=SimpleNamespace(invoke_model=invoke_model))
    vectors, _ = embedding_request(llm, "cohere-v4", ["가\n 나"], kind)
    body = json.loads(requests[0]["body"])
    assert body["input_type"] == kind
    assert body["texts"] == ["가 나"]
    assert body["output_dimension"] == 1024
    assert body["truncate"] == "NONE"
    assert sum(x * x for x in vectors[0]) == pytest.approx(1)


def test_global_benchmark_does_not_count_a_page_from_another_document():
    q = Question(id="q", doc_id="wanted", bucket="A", query="질문", pages=[3])
    rows = [{"_score": 1, "_source": {"chunk_id": "x", "doc_id": "other", "page": 3}}]
    result = grade(q, {"vector": rows})
    assert not result["hit@1"]
    assert not result["hit@5"]
    assert result["recall@5"] == 0


def test_card_summary_does_not_require_chunk_hit_fields():
    result = aggregate([{"id": "a", "doc@1": True}, {"id": "b", "doc@1": False}])
    assert result["doc@1"] == 0.5

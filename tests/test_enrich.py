from haeindex.enrich import enrich, parse_enrichment


def test_메타데이터_JSON을_읽는다() -> None:
    got = parse_enrichment(
        '설명\n{"summary":"휴가 규정", "keywords":["연차"], '
        '"entities":["제10조"], "content_type":"requirement", "language":"ko"}'
    )
    assert got is not None
    assert got.summary == "휴가 규정"
    assert got.entities == ["제10조"]


def test_허용하지_않은_유형은_실패다() -> None:
    raw = '{"summary":"x","content_type":"추측","language":"ko"}'
    assert parse_enrichment(raw) is None


def test_JSON이_아니면_실패다() -> None:
    assert parse_enrichment("요약입니다") is None


def test_프롬프트의_JSON_예시와_본문을_함께_보낸다() -> None:
    class FakeLLM:
        def chat(self, messages, **kwargs):
            prompt = messages[0]["content"]
            assert '"summary"' in prompt
            assert "연차는 15일" in prompt
            raw = (
                '{"summary":"연차", "keywords":[], "entities":[], '
                '"content_type":"fact", "language":"ko"}'
            )
            return type("Reply", (), {"content": raw})()

    assert enrich(FakeLLM(), "연차는 15일") is not None

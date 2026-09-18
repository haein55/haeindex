from haeindex.index import mappings


def test_LLM_메타데이터는_원문과_별도_필드다() -> None:
    props = mappings(dim=4)["properties"]
    assert props["body"]["index"] is False
    assert props["summary"]["type"] == "text"
    assert props["keywords"]["type"] == "text"
    assert props["entities"]["type"] == "text"
    assert props["content_type"] == {"type": "keyword"}
    assert props["language"] == {"type": "keyword"}

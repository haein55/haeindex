from types import SimpleNamespace

from haeindex.index import doc_stats, mappings


def test_LLM_메타데이터는_원문과_별도_필드다() -> None:
    props = mappings(dim=4)["properties"]
    assert props["body"]["index"] is False
    assert props["summary"]["type"] == "text"
    assert props["keywords"]["type"] == "text"
    assert props["entities"]["type"] == "text"
    assert props["content_type"] == {"type": "keyword"}
    assert props["language"] == {"type": "keyword"}


def test_document_stats_report_last_indexed_page() -> None:
    client = SimpleNamespace(
        indices=SimpleNamespace(exists=lambda **kwargs: True),
        search=lambda **kwargs: {
            "aggregations": {
                "d": {
                    "buckets": [
                        {"key": "long", "doc_count": 120, "last_page": {"value": 120.0}}
                    ]
                }
            }
        },
    )
    assert doc_stats(client) == {"long": {"chunks": 120, "last_page": 120}}

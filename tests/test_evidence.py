from haeindex.evidence import select_evidence
from haeindex.search import Hit, Result


def hit(chunk_id: str, doc_id: str) -> Hit:
    return Hit(chunk_id=chunk_id, fused=1.0, source={"chunk_id": chunk_id, "doc_id": doc_id})


def test_일반_질문은_첫_근거와_같은_문서만_남긴다() -> None:
    result = Result(query="q", hits=[hit("a", "doc-a"), hit("b", "doc-b"), hit("c", "doc-a")])
    selected = select_evidence(result, top_k=5, allow_multiple_docs=False)
    assert [h.chunk_id for h in selected.hits] == ["a", "c"]


def test_비교_질문은_여러_문서를_남긴다() -> None:
    result = Result(query="q", hits=[hit("a", "doc-a"), hit("b", "doc-b")])
    selected = select_evidence(result, top_k=5, allow_multiple_docs=True)
    assert [h.chunk_id for h in selected.hits] == ["a", "b"]

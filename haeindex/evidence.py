from haeindex.search import Result


def select_evidence(result: Result, *, top_k: int, allow_multiple_docs: bool) -> Result:
    """일반 질문은 가장 강한 한 문서만, 명시적 비교 질문은 여러 문서를 허용한다."""
    if allow_multiple_docs or not result.hits:
        return result.model_copy(update={"hits": result.hits[:top_k]})
    selected = str(result.hits[0].source.get("doc_id", ""))
    hits = [h for h in result.hits if str(h.source.get("doc_id", "")) == selected]
    return result.model_copy(update={"hits": hits[:top_k]})

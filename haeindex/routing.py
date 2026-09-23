import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_DATE_SUFFIX = re.compile(r"(?:^|[-_])(\d{6,8})(?:$|[-_])")
_PART = re.compile(r"[가-힣]{2,}|[a-zA-Z]+|\d+")
_COMPARE = re.compile(
    r"비교|차이|각각|공통|서로|두\s*(?:문서|파일)|여러\s*(?:문서|파일)|"
    r"compare|difference|both|each\s+(?:document|file)",
    re.IGNORECASE,
)


class Route(BaseModel):
    model_config = ConfigDict(frozen=True)

    language: str
    doc_ids: list[str] = Field(default_factory=list)
    allow_multiple_docs: bool = False
    reason: str = "전체 검색"
    clarification: str = ""


def detect_language(text: str) -> str:
    """질문의 주 언어만 가른다. 파일명·모델명의 영문은 한국어 판정을 뒤집지 않는다."""
    ko = len(_HANGUL.findall(text))
    en = len(_LATIN.findall(text))
    if ko:
        return "ko"
    return "en" if en else "ko"


def is_multi_document_question(question: str) -> bool:
    return bool(_COMPARE.search(question))


def _aliases(doc_id: str) -> set[str]:
    low = doc_id.lower()
    aliases = {low, low.replace("-", " ").replace("_", " ")}
    aliases.add(_DATE_SUFFIX.sub("-", low).strip("-_ "))
    parts = _PART.findall(low)
    aliases.update(p for p in parts if len(p) >= 3 and not p.isdigit())

    # 영문 제품명은 연속된 두 토큰으로 언급되는 경우가 많다: hxr-mc88.
    for a, b in zip(parts, parts[1:], strict=False):
        if not (a.isdigit() and b.isdigit()):
            aliases.add(f"{a} {b}")
            aliases.add(f"{a}-{b}")
    return {a.strip() for a in aliases if len(a.strip()) >= 3}


def mentioned_docs(question: str, doc_ids: Sequence[str]) -> list[str]:
    normalized = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", question.lower()).strip()
    compact = normalized.replace(" ", "")
    found: list[tuple[int, str]] = []
    for doc_id in doc_ids:
        matched = [
            alias
            for alias in _aliases(doc_id)
            if alias in normalized or alias.replace(" ", "").replace("-", "") in compact
        ]
        if matched:
            found.append((max(map(len, matched)), doc_id))
    return [doc_id for _, doc_id in sorted(found, key=lambda x: (-x[0], x[1]))]


def route_question(
    question: str,
    doc_ids: Sequence[str],
    *,
    explicit_doc_ids: Sequence[str] = (),
) -> Route:
    multi = is_multi_document_question(question)
    mentioned = mentioned_docs(question, doc_ids)
    if explicit_doc_ids:
        selected = list(dict.fromkeys(explicit_doc_ids))
        missing = [doc for doc in mentioned if doc not in selected]
        clarification = ""
        if len(mentioned) > 1 and missing:
            names = ", ".join(missing)
            clarification = (
                f"질문에 함께 언급된 {names} 문서가 현재 검색 범위에 없습니다. "
                "해당 문서를 함께 선택하거나 전체 문서에서 다시 질문해 주세요."
                if detect_language(question) == "ko"
                else f"The question also names {names}, outside the selected documents. "
                "Select those documents too, or search all documents."
            )
        return Route(
            language=detect_language(question),
            doc_ids=selected,
            allow_multiple_docs=multi or len(selected) > 1,
            reason="--doc 명시",
            clarification=clarification,
        )
    return Route(
        language=detect_language(question),
        doc_ids=mentioned,
        # 적용·해당 여부처럼 '비교'라는 말 없이 여러 문서를 연결하는 질문도 보존한다.
        allow_multiple_docs=multi or len(mentioned) > 1,
        reason="질문의 파일명" if mentioned else "전체 검색",
    )

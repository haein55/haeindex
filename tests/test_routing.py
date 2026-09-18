from haeindex.routing import (
    detect_language,
    is_multi_document_question,
    mentioned_docs,
    route_question,
)

DOCS = [
    "취업규칙-260417",
    "3-홍천양수발전소-1-2호기-토건공사-입찰안내서",
    "hxr-mc88-manual-kr1",
]


def test_파일명이_질문에_있으면_문서를_고른다() -> None:
    assert mentioned_docs("입찰안내서 제17조를 알려줘", DOCS) == [DOCS[1]]
    assert mentioned_docs("취업규칙에서 휴가는?", DOCS) == [DOCS[0]]


def test_영문_제품명도_문서를_고른다() -> None:
    assert mentioned_docs("HXR-MC88 포맷 방법", DOCS) == [DOCS[2]]


def test_관련_없는_질문은_문서를_억지로_고르지_않는다() -> None:
    assert mentioned_docs("휴가는 며칠인가요", DOCS) == []


def test_한국어_질문에_영문_파일명이_있어도_한국어다() -> None:
    assert detect_language("HXR-MC88 파일에서 포맷 방법은?") == "ko"
    assert detect_language("How do I format the card?") == "en"


def test_비교_질문만_여러_문서를_허용한다() -> None:
    assert is_multi_document_question("두 파일의 차이를 비교해줘")
    assert not is_multi_document_question("입찰안내서 내용을 알려줘")
    route = route_question("취업규칙과 입찰안내서를 비교해줘", DOCS)
    assert route.allow_multiple_docs
    assert route.doc_ids == [DOCS[1], DOCS[0]]


def test_문서가_많아도_명시된_파일만_고른다() -> None:
    many = [f"서로다른보고서-{i:04d}" for i in range(500)] + DOCS
    route = route_question("취업규칙-260417 파일의 휴가 조항", many)
    assert route.doc_ids == [DOCS[0]]

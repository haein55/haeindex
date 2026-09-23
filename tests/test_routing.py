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


CERTIFICATE = "서울시-평생학습포털-수료증-개인정보보호교육"
CROSS_DOCUMENT_QUESTION = "취업규칙 교육훈련에 서울시 평생학습포털 수료증의 내용이 해당해"


def test_두_문서의_해당_여부_질문은_비교라는_말_없이도_함께_검색한다():
    route = route_question(CROSS_DOCUMENT_QUESTION, [*DOCS, CERTIFICATE])
    assert set(route.doc_ids) == {DOCS[0], CERTIFICATE}
    assert route.allow_multiple_docs
    assert not route.clarification


def test_선택_범위_밖의_문서가_함께_필요하면_범위를_확인한다():
    route = route_question(
        CROSS_DOCUMENT_QUESTION, [*DOCS, CERTIFICATE], explicit_doc_ids=[DOCS[0]]
    )
    assert route.doc_ids == [DOCS[0]]
    assert CERTIFICATE in route.clarification
    assert "함께 선택" in route.clarification


def test_두_문서를_모두_선택했으면_확인_질문_없이_진행한다():
    selected = [DOCS[0], CERTIFICATE]
    route = route_question(
        CROSS_DOCUMENT_QUESTION, [*DOCS, CERTIFICATE], explicit_doc_ids=selected
    )
    assert route.doc_ids == selected
    assert route.allow_multiple_docs
    assert not route.clarification


def test_단일_문서_질문은_문서_혼합을_허용하지_않는다():
    route = route_question("취업규칙의 교육훈련 항목을 알려줘", [*DOCS, CERTIFICATE])
    assert route.doc_ids == [DOCS[0]]
    assert not route.allow_multiple_docs

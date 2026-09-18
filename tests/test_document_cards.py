from haeindex.chunks import Chunk
from haeindex.document_cards import (
    card_bm25_body,
    card_mappings,
    fuse_card_hits,
    generate_card,
    infer_source_language,
    normalize_document_type,
    parse_card,
    sampled_content,
    search_lexical_docs,
)


def chunk(seq: int, title: str) -> Chunk:
    return Chunk(
        doc_id="d",
        chunk_id=f"d#c{seq:04d}",
        seq=seq,
        title=title,
        path=title,
        depth=0,
        page=seq + 1,
        end_page=seq + 1,
        body=f"{title} 본문 " * 100,
        text=title,
        char_len=100,
        token_len=20,
    )


def row(doc_id: str) -> dict:
    return {"_source": {"doc_id": doc_id, "summary": doc_id}}


def test_문서_카드_JSON을_검증하고_파일명_별칭을_보존한다() -> None:
    raw = (
        '{"summary":"취업 규정", "topics":["휴가"], "entities":["제63조"], '
        '"aliases":["사규"], "document_type":"policy", "language":"ko"}'
    )
    card = parse_card(raw, doc_id="취업규칙-260417", source="취업규칙_260417.pdf")
    assert card is not None
    assert "취업규칙_260417.pdf" in card.aliases
    assert "사규" in card.aliases


def test_문서_전체에서_고르게_표본을_뽑는다() -> None:
    content = sampled_content([chunk(i, f"절{i}") for i in range(100)])
    assert "절0" in content
    assert "절99" in content
    assert len(content) <= 7000


def test_문서_검색은_파일명과_별칭을_강하게_본다() -> None:
    fields = card_bm25_body("질문", 20)["query"]["multi_match"]["fields"]
    assert "source^5" in fields
    assert "aliases^5" in fields


def test_문서_카드도_BM25와_kNN을_RRF로_합친다() -> None:
    hits = fuse_card_hits(
        {"bm25": [row("a"), row("both")], "knn": [row("b"), row("both")]}
    )
    assert hits[0].doc_id == "both"


def test_문서_인덱스에는_별도_임베딩이_있다() -> None:
    props = card_mappings(dim=4)["properties"]
    assert props["embedding"]["dimension"] == 4
    assert props["summary"]["type"] == "text"


def test_파일명에서_명확한_문서_유형은_LLM_오분류를_보정한다() -> None:
    assert normalize_document_type("manual", "홍천 공사 입찰안내서.pdf") == "guide"
    assert normalize_document_type("survey", "AGENTIC RAG SURVEY.pdf") == "paper"


def test_카드_설명어가_아니라_원문_표본으로_문서_언어를_정한다() -> None:
    assert infer_source_language("This document explains agentic retrieval." * 10) == "en"
    assert infer_source_language("이 문서는 취업규칙을 설명합니다. HXR MC88" * 10) == "ko"


def test_카드_JSON_실패시_다른_seed로_한번_더_시도한다() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.seeds = []

        def chat(self, messages, **kwargs):
            self.seeds.append(kwargs["seed"])
            if len(self.seeds) == 1:
                return type("Reply", (), {"content": "JSON 아님"})()
            raw = (
                '{"summary":"논문", "topics":["RAG"], "entities":[], '
                '"aliases":[], "document_type":"paper", "language":"en"}'
            )
            return type("Reply", (), {"content": raw})()

    llm = FakeLLM()
    got = generate_card(llm, "paper", "paper.pdf", [chunk(0, "RAG")])
    assert got is not None
    assert llm.seeds == [0, 1]


def test_희귀_원문_용어는_전역_BM25에서_문서를_보존한다() -> None:
    class Indices:
        def exists(self, index):
            return True

    class FakeOS:
        indices = Indices()

        def search(self, index, body):
            assert body["collapse"] == {"field": "doc_id"}
            assert body["query"]["bool"]["should"][0]["multi_match"]["query"] == "XLR"
            return {
                "hits": {
                    "hits": [
                        {"_source": {"doc_id": "hxr"}},
                        {"_source": {"doc_id": "hxr"}},
                        {"_source": {"doc_id": "other"}},
                    ]
                }
            }

    assert search_lexical_docs(FakeOS(), "XLR") == ["hxr", "other"]

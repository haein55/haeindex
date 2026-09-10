from haeindex.blocks import Block
from haeindex.chunks import (
    build_sections,
    chunk_fixed,
    chunk_sections,
    report,
    token_len_ko,
)
from haeindex.headings import Heading


def blk(text: str, page: int, order: int, size: float = 10.0) -> Block:
    return Block(
        page_index=page,
        order=order,
        column=0,
        text=text,
        size=size,
        fontname="HCRBatang",
        x0=20.0,
        x1=200.0,
        top=100.0,
        bottom=100.0 + size,
        n_chars=len(text),
    )


def head(text: str, page: int, order: int, level: int, size: float = 12.0) -> Heading:
    return Heading(
        page_index=page,
        order=order,
        text=text,
        size=size,
        fontname="HCRBatang-Bold",
        signal="size",
        level=level,
    )


def test_한글은_음절로_센다() -> None:
    spaced = "메모리 카드를"
    assert len(spaced.split(" ")) == 2
    assert token_len_ko(spaced) == 6


def test_스택으로_계층을_만든다() -> None:
    blocks = [
        blk("제1장 총칙", 1, 0),
        blk("제1조 목적", 1, 1),
        blk("본문이다", 1, 2),
        blk("제2장 채용", 1, 3),
    ]
    hs = [head("제1장 총칙", 1, 0, 1), head("제1조 목적", 1, 1, 2), head("제2장 채용", 1, 3, 1)]
    secs = build_sections("d", blocks, hs)
    assert [s.path for s in secs] == ["제1장 총칙", "제1장 총칙 > 제1조 목적", "제2장 채용"]
    assert [s.depth for s in secs] == [0, 1, 0]
    assert secs[1].parent_id == secs[0].section_id


def test_제목은_본문에_안_들어간다() -> None:
    blocks = [blk("제1조 목적", 1, 0), blk("본문이다", 1, 1)]
    (sec,) = build_sections("d", blocks, [head("제1조 목적", 1, 0, 1)])
    assert sec.body == "본문이다"
    assert sec.title == "제1조 목적"


def test_접두어는_text_에만_붙는다() -> None:
    blocks = [blk("제1조 목적", 1, 0), blk("본문이다" * 40, 1, 1)]
    secs = build_sections("d", blocks, [head("제1조 목적", 1, 0, 1)])
    (c,) = chunk_sections(secs)
    assert c.text.startswith("제1조 목적")
    assert not c.body.startswith("제1조 목적")


def test_짧은_절을_합치고_절_id_를_다_들고_간다() -> None:
    blocks = []
    hs = []
    for i in range(4):
        blocks.append(blk(f"제{i + 1}조", 1, i * 2))
        blocks.append(blk("짧다", 1, i * 2 + 1))
        hs.append(head(f"제{i + 1}조", 1, i * 2, 1))
    secs = build_sections("d", blocks, hs)
    chunks = chunk_sections(secs, min_chars=100)
    assert len(chunks) == 1
    assert len(chunks[0].section_ids) == 4


def test_긴_절은_쪼개고_모두_상한_아래다() -> None:
    body = "\n".join("가" * 90 for _ in range(30))
    blocks = [blk("제1조", 1, 0), blk(body, 1, 1)]
    secs = build_sections("d", blocks, [head("제1조", 1, 0, 1)])
    chunks = chunk_sections(secs, max_chars=500)
    assert len(chunks) > 1
    assert all(c.char_len <= 500 for c in chunks)


def test_제목이_없으면_고정_길이로_폴백한다() -> None:
    blocks = [blk("문장 하나" * 20, 1, i) for i in range(10)]
    chunks = chunk_fixed("d", blocks, max_chars=400)
    assert len(chunks) > 1
    assert all(c.section_ids == [] for c in chunks)
    assert all(c.text == c.body for c in chunks)


def test_이중_렌더를_되돌려_본문에_넣는다() -> None:
    blocks = [blk("제제1조조", 1, 0), blk("본본문문이이다다", 1, 1)]
    hs = [head("제제1조조", 1, 0, 1)]
    (sec,) = build_sections("d", blocks, hs, fix_doubled=True)
    assert sec.title == "제1조"
    assert sec.body == "본문이다"


def test_보고서가_안_덮인_절을_짚는다() -> None:
    blocks = [blk("제1조", 1, 0), blk("본문", 1, 1)]
    secs = build_sections("d", blocks, [head("제1조", 1, 0, 1)])
    rep = report("d", "numbering", secs, chunk_sections(secs), 1200)
    assert rep.uncovered_sections == []
    assert report("d", "numbering", secs, [], 1200).uncovered_sections == [secs[0].section_id]

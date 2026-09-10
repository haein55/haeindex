from haeindex.blocks import Block
from haeindex.headings import (
    assign_levels,
    body_ceiling,
    detect_by_font,
    detect_by_numbering,
    font_weight,
    heading_sequences,
    longest_increasing_run,
    modal_size,
    repeated_texts,
    undouble,
    weight_coverage,
)


def blk(
    text: str, page: int = 1, order: int = 0, size: float = 10.0, font: str = "HCRBatang"
) -> Block:
    return Block(
        page_index=page,
        order=order,
        column=0,
        text=text,
        size=size,
        fontname=font,
        x0=20.0,
        x1=200.0,
        top=100.0,
        bottom=100.0 + size,
        n_chars=len(text),
    )


def test_굵기_표기가_달라도_읽는다() -> None:
    assert font_weight("SDGothic-cBd") == 5
    assert font_weight("XYZ+HCRBatang-Bold") == 5
    assert font_weight("NimbusRomNo9L-Medi") == 3
    assert font_weight("SDMyeongjo-aLt") == 1
    assert font_weight("NimbusRomNo9L-Regu") == 2


def test_굵기_정보가_없으면_모름이다() -> None:
    for name in ("HCRBatang", "H2mjsM", "HCRDotum", "F9", "ArialMT"):
        assert font_weight(name) is None


def test_semibold_가_bold_보다_먼저_걸린다() -> None:
    assert font_weight("Foo-Semibold") == 4
    assert font_weight("Foo-Bold") == 5


def test_모름과_보통은_다르게_센다() -> None:
    known = [blk("가", font="HCRBatang-Bold"), blk("나", font="HCRBatang")]
    assert weight_coverage(known) == 0.5
    assert weight_coverage([blk("가", font="H2mjsM")]) == 0.0


def test_본문_크기는_문자수로_가중한다() -> None:
    body = [blk("본문이아주길다" * 5, size=10.0)]
    title = [blk("제목", size=20.0) for _ in range(3)]
    assert modal_size(body + title) == 10.0


def test_크기나_굵기로_제목을_찾는다() -> None:
    bs = [
        blk("본문 문장이다", size=10.0, font="HCRBatang"),
        blk("큰 제목", size=14.0, font="HCRBatang"),
        blk("굵은 제목", size=10.0, font="HCRBatang-Bold"),
    ]
    got = detect_by_font(bs, ceiling=10.0)
    assert [h.signal for h in got] == ["size", "bold"]


def test_굵기_모름은_제목으로_보지_않는다() -> None:
    bs = [blk("본문", size=10.0, font="H2mjsM"), blk("이것도 본문", size=10.0, font="F9")]
    assert detect_by_font(bs, ceiling=10.0) == []


def test_숫자만_있는_블록은_제목이_아니다() -> None:
    bs = [blk("12", size=20.0), blk("①", size=20.0), blk("iv", size=20.0)]
    assert detect_by_font(bs, ceiling=10.0) == []


def test_반복되는_머리말은_빼되_첫_출현은_남긴다() -> None:
    bs = [blk("레코딩", page=p, order=0) for p in range(1, 6)]
    drop = repeated_texts(bs)
    assert (1, 0) not in drop
    assert len(drop) == 4


def test_같은_페이지_반복은_머리말이_아니다() -> None:
    bs = [blk("표 항목", page=1, order=i) for i in range(5)]
    assert repeated_texts(bs) == set()


def test_연속_수열을_찾는다() -> None:
    bs = [blk(f"제{i}조(목적)", page=1 + i // 3, order=i) for i in range(1, 13)]
    (s,) = [x for x in heading_sequences(bs) if x.pattern == "제N조"]
    assert s.observed == 12
    assert s.missing == []


def test_줄_중간의_참조는_수열이_아니다() -> None:
    bs = [
        blk("「선박안전법」 제27조제1항제2호에 따라", page=1),
        blk("「전파법 시행령」 제117조제2항에 따라", page=2),
        blk("「수산업법」 제40조에 따라 허가받은", page=3),
    ]
    assert [s for s in heading_sequences(bs) if s.pattern == "제N조"] == []


def test_연도는_수열이_아니다() -> None:
    bs = [blk("2025. Agentic RAG survey.", page=i) for i in range(1, 4)]
    assert heading_sequences(bs) == []


def test_한_페이지에_모인_번호는_수열이_아니다() -> None:
    bs = [blk(f"{i}. 표 항목", page=1, order=i) for i in range(1, 8)]
    assert heading_sequences(bs) == []


def test_이중_렌더를_되돌리면_수열이_보인다() -> None:
    bs = [blk(f"제제{i}조조((목목적적))", page=1 + i // 3, order=i) for i in range(1, 13)]
    assert heading_sequences(bs) == []
    assert [s.pattern for s in heading_sequences(bs, fix_doubled=True)] == ["제N조"]


def test_undouble_에서_숫자를_지킬_수도_있다() -> None:
    assert undouble("제제1조조", digits=False) == "제1조"
    assert undouble("1000톤", digits=False) == "1000톤"


def test_번호로_제목을_뽑는다() -> None:
    bs = [blk("제1조(목적)", page=1), blk("본문이다", page=1, order=1), blk("제2조(정의)", page=2)]
    got = detect_by_numbering(bs, "제N조")
    assert [h.text for h in got] == ["제1조(목적)", "제2조(정의)"]


def test_레벨은_큰_글꼴이_1이다() -> None:
    hs = detect_by_font(
        [blk("장", size=20.0), blk("절", size=14.0), blk("항", size=12.0)], ceiling=10.0
    )
    levels = {h.text: h.level for h in assign_levels(hs)}
    assert levels == {"장": 1, "절": 2, "항": 3}


def test_짧은_런이_반복되면_수열이_아니다() -> None:
    bs = [blk(f"{i}. 비고 항목", page=page, order=i) for page in range(1, 6) for i in range(1, 4)]
    assert heading_sequences(bs) == []


def test_긴_증가런이_있으면_수열이다() -> None:
    bs = [blk(f"제{i}조(목적)", page=1 + i // 3, order=i) for i in range(1, 15)]
    assert [s.pattern for s in heading_sequences(bs)] == ["제N조"]


def test_연속_중복은_런을_끊지_않는다() -> None:
    assert longest_increasing_run([1, 2, 2, 3, 3, 4]) == 4
    assert longest_increasing_run([1, 2, 3, 1, 2, 3]) == 3


def test_숫자도_되돌려야_조문이_보인다() -> None:
    assert undouble("제제11조조") == "제1조"
    assert undouble("제제1100조조") == "제10조"
    assert undouble("11000000톤") == "1000톤"
    assert undouble("제제11조조", digits=False) == "제11조"


def test_번호_제목에도_상한을_건다() -> None:
    bs = [blk("2025. 5", page=1), blk("1. 목적", page=1, order=1)]
    got = detect_by_numbering(bs, "N.")
    assert [h.text for h in got] == ["1. 목적"]


def test_본문_크기가_둘이면_상한을_쓴다() -> None:
    bs = [blk("본문가나다라마" * 4, size=8.0) for _ in range(6)]
    bs += [blk("본문가나다라마" * 4, size=8.5) for _ in range(5)]
    bs += [blk("제목", size=12.0)]
    assert modal_size(bs) == 8.0
    assert body_ceiling(bs) == 8.5
    got = detect_by_font(bs, ceiling=body_ceiling(bs))
    assert [h.text for h in got] == ["제목"]


def test_최장런이_긴_패턴을_먼저_고른다() -> None:
    bs = [blk(f"제{i}조(목적)", page=1 + i // 3, order=i) for i in range(1, 20)]
    bs += [blk(f"{i}. 목록", page=page, order=50 + i) for page in range(1, 9) for i in range(1, 10)]
    seqs = heading_sequences(bs)
    assert seqs[0].pattern == "제N조"


def test_조사가_붙으면_상호_참조다() -> None:
    bs = [blk(f"제{i}조(목적)", page=1 + i // 3, order=i) for i in range(1, 15)]
    bs.append(blk("제3조의 규정에 따른 계약에 관한 업무", page=1, order=90))
    bs.append(blk("세칙 제97조 또는 제97조의2에 해당하는", page=2, order=91))
    got = detect_by_numbering(bs, "제N조")
    assert all("규정에 따른" not in h.text for h in got)
    assert len(got) == 14

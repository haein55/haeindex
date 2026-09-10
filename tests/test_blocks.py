from haeindex.blocks import build_blocks, crosses_gutter, size_tiers
from haeindex.load_pdf import PDFChar, PDFPage


def ch(
    text: str, x0: float, baseline: float, size: float = 10.0, font: str = "AAA+Body"
) -> PDFChar:
    return PDFChar(
        text=text,
        x0=x0,
        x1=x0 + size * 0.5,
        top=baseline - size,
        bottom=baseline,
        size=size,
        fontname=font,
    )


def row(s: str, y: float, x: float, size: float = 10.0, font: str = "AAA+Body") -> list[PDFChar]:
    return [ch(c, x + i * size * 0.5, y, size, font) for i, c in enumerate(s)]


def page(chars: list[PDFChar], width: float = 362.0, height: float = 516.0) -> PDFPage:
    return PDFPage(page_index=1, width=width, height=height, chars=chars)


LEFT_X = 20.0
RIGHT_X = 190.0
GUTTER = [(177.0, 184.0)]


def test_단을_나누면_좌우가_섞이지_않는다() -> None:
    chars = row("왼쪽줄하나", 100, LEFT_X) + row("오른쪽줄하나", 100, RIGHT_X)
    mixed = build_blocks(page(chars), [])
    split = build_blocks(page(chars), GUTTER)
    assert len(mixed) == 1
    assert mixed[0].text == "왼쪽줄하나 오른쪽줄하나"
    assert [b.text for b in split] == ["왼쪽줄하나", "오른쪽줄하나"]
    assert [b.column for b in split] == [0, 1]


def test_같은_단의_다른_줄은_따로_묶인다() -> None:
    chars = row("첫째줄", 100, LEFT_X) + row("둘째줄", 120, LEFT_X)
    blocks = build_blocks(page(chars), GUTTER)
    assert [b.text for b in blocks] == ["첫째줄", "둘째줄"]


def test_간격이_크면_공백을_넣는다() -> None:
    chars = row("앞", 100, LEFT_X) + row("뒤", 100, LEFT_X + 60)
    (b,) = build_blocks(page(chars), [])
    assert b.text == "앞 뒤"


def test_이미_있는_공백을_겹치지_않는다() -> None:
    chars = row("가 나", 100, LEFT_X)
    (b,) = build_blocks(page(chars), [])
    assert b.text == "가 나"


def test_크기는_중앙값이라_한_글자에_흔들리지_않는다() -> None:
    chars = row("본문본문본문", 110, LEFT_X, size=10.0)
    chars.append(ch("®", LEFT_X + 90, 110, size=30.0))
    (b,) = build_blocks(page(chars), [])
    assert b.size == 10.0


def test_밑선이_같으면_크기가_달라도_한_줄이다() -> None:
    chars = [ch("1", LEFT_X, 110, size=14.0)]
    chars += row("MENU 버튼을 누릅니다", 110, LEFT_X + 12, size=10.0)
    (b,) = build_blocks(page(chars), [])
    assert b.text.startswith("1")
    assert "누릅니다" in b.text


def test_지배_글꼴을_고르고_서브셋_접두어를_뗀다() -> None:
    chars = row("본문본문본문", 100, LEFT_X, font="XYZ+HCRBatang")
    chars += row("굵", 100, LEFT_X + 90, font="XYZ+HCRBatang-Bold")
    (b,) = build_blocks(page(chars), [])
    assert b.fontname == "HCRBatang"


def test_공백만_있는_줄은_블록이_아니다() -> None:
    chars = row("   ", 100, LEFT_X)
    assert build_blocks(page(chars), []) == []


def test_gutter_를_가로지르는_블록을_찾아낸다() -> None:
    chars = row("왼쪽줄하나", 100, LEFT_X) + row("오른쪽줄하나", 100, RIGHT_X)
    (mixed,) = build_blocks(page(chars), [])
    (left, right) = build_blocks(page(chars), GUTTER)
    assert crosses_gutter(mixed, GUTTER)
    assert not crosses_gutter(left, GUTTER)
    assert not crosses_gutter(right, GUTTER)


def test_크기_티어는_문자수로_가중한다() -> None:
    chars = row("본문이길다본문이길다", 100, LEFT_X, size=10.0)
    chars += row("제목", 60, LEFT_X, size=20.0)
    tiers = size_tiers(build_blocks(page(chars), []))
    assert tiers[10.0] > tiers[20.0]

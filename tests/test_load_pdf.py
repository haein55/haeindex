from haeindex.load_pdf import PDFChar, doubled_ratio, line_bin


def ch(text: str, x0: float, top: float, size: float = 12.0) -> PDFChar:
    return PDFChar(
        text=text,
        x0=x0,
        x1=x0 + size * 0.5,
        top=top,
        bottom=top + size,
        size=size,
        fontname="H2mjsM",
    )


def line(s: str, y: float, x: float = 10.0, step: float = 6.0) -> list[PDFChar]:
    return [ch(c, x + i * step, y) for i, c in enumerate(s)]


def test_이중_렌더를_잡는다() -> None:
    a = line("제1조목적", 100)
    b = line("제1조목적", 100.2, x=10.3)
    assert doubled_ratio(a + b * 4) > 0.3


def test_스트림_순서로_재면_놓친다() -> None:
    text = "제1조목적" * 2
    stream = sum(1 for x, y in zip(text, text[1:], strict=False) if x == y) / (len(text) - 1)
    assert stream == 0.0
    assert doubled_ratio((line("제1조목적", 100) + line("제1조목적", 100.1, x=10.2)) * 3) > 0.3


def test_정상_문서는_낮게_나온다() -> None:
    chars = [
        c
        for i, s in enumerate(
            ["메모리 카드를 포맷하는 방법", "배터리 충전 시간 안내", "화이트 밸런스 조정하기"]
        )
        for c in line(s, 100 + i * 14)
    ]
    assert doubled_ratio(chars) < 0.1


def test_숫자는_세지_않는다() -> None:
    chars = [
        c
        for i, s in enumerate(["1000 2025 3333", "4444 5555 6666", "7777 8888 9999"])
        for c in line(s, 100 + i * 14)
    ]
    assert doubled_ratio(chars) < 0.1


def test_줄_버킷은_글자_크기에_비례한다() -> None:
    small = [ch("가", 10 + i * 6, 100, size=9) for i in range(10)]
    big = [ch("가", 10 + i * 24, 100, size=36) for i in range(10)]
    assert line_bin(big) > line_bin(small) * 3

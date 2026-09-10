from haeindex.headings import SeqStat
from haeindex.load_pdf import PDFChar, PDFPage
from haeindex.profile import decide, space_char_ratio


def seq(pattern: str = "제N조", run: int = 20) -> SeqStat:
    return SeqStat(
        pattern=pattern,
        observed=run,
        max_ordinal=run,
        pages=run,
        longest_run=run,
        missing=[],
    )


def test_쪽수가_적으면_구조를_안_믿는다() -> None:
    method, pattern, why = decide(n_pages=1, sequences=[seq()], n_font_headings=50)
    assert method == "none"
    assert pattern is None
    assert "1쪽" in why["heading_method"]


def test_수열이_있으면_수열을_쓴다() -> None:
    method, pattern, why = decide(n_pages=30, sequences=[seq()], n_font_headings=200)
    assert (method, pattern) == ("numbering", "제N조")
    assert "자기검증" in why["heading_method"]


def test_수열이_없으면_글꼴로_간다() -> None:
    method, pattern, _ = decide(n_pages=30, sequences=[], n_font_headings=50)
    assert (method, pattern) == ("font", None)


def test_둘_다_안_되면_구조가_없는_문서다() -> None:
    method, _, why = decide(n_pages=30, sequences=[], n_font_headings=0)
    assert method == "none"
    assert "구조가 없는" in why["heading_method"]


def test_공백_문자_비율을_잰다() -> None:
    def ch(text: str) -> PDFChar:
        return PDFChar(text=text, x0=0.0, x1=5.0, top=0.0, bottom=10.0, size=10.0, fontname="A")

    page = PDFPage(
        page_index=1,
        width=100.0,
        height=100.0,
        chars=[ch("가"), ch(" "), ch("나"), ch(" ")],
    )
    assert space_char_ratio([page]) == 0.5
    assert space_char_ratio([]) == 0.0

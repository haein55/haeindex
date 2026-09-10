import statistics
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from haeindex.geometry import Span, column_of, page_gutters
from haeindex.load_pdf import PDFChar, PDFPage, line_bin

DEFAULT_SPACE_RATIO = 0.7


class Block(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_index: int
    order: int
    column: int
    text: str
    size: float
    fontname: str
    x0: float
    x1: float
    top: float
    bottom: float
    n_chars: int


def _join(chars: Sequence[PDFChar], space_ratio: float) -> str:
    ordered = sorted(chars, key=lambda c: c.x0)
    widths = [c.x1 - c.x0 for c in ordered if c.x1 > c.x0]
    threshold = space_ratio * (statistics.median(widths) if widths else 1.0)

    parts = [ordered[0].text]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        gap = cur.x0 - prev.x1
        if gap > threshold and not prev.text.isspace() and not cur.text.isspace():
            parts.append(" ")
        parts.append(cur.text)
    return " ".join("".join(parts).split())


def _dominant_font(chars: Sequence[PDFChar]) -> str:
    named = Counter(c.fontname.split("+")[-1] for c in chars if not c.text.isspace())
    return named.most_common(1)[0][0] if named else ""


def build_blocks(
    page: PDFPage,
    gutters: Sequence[Span],
    space_ratio: float = DEFAULT_SPACE_RATIO,
) -> list[Block]:
    if not page.chars:
        return []
    bin_h = line_bin(page.chars)
    rows: dict[tuple[int, int], list[PDFChar]] = {}
    for c in page.chars:
        col = column_of((c.x0 + c.x1) / 2, gutters)
        row = round(c.bottom / bin_h)
        rows.setdefault((col, row), []).append(c)

    out: list[Block] = []
    for (col, _), chars in sorted(rows.items()):
        content = [c for c in chars if not c.text.isspace()]
        if not content:
            continue
        text = _join(chars, space_ratio)
        if not text:
            continue
        out.append(
            Block(
                page_index=page.page_index,
                order=len(out),
                column=col,
                text=text,
                size=round(statistics.median(c.size for c in content), 2),
                fontname=_dominant_font(content),
                x0=round(min(c.x0 for c in content), 2),
                x1=round(max(c.x1 for c in content), 2),
                top=round(min(c.top for c in content), 2),
                bottom=round(max(c.bottom for c in content), 2),
                n_chars=len(content),
            )
        )
    return out


def page_blocks(
    page: PDFPage,
    *,
    header: float = 0.0,
    footer: float = 1.0,
    bins: int = 160,
    min_gap: float = 3.0,
    space_ratio: float = DEFAULT_SPACE_RATIO,
) -> list[Block]:
    gutters = page_gutters(
        page.chars,
        page.width,
        page.height,
        header=header,
        footer=footer,
        bins=bins,
        min_gap=min_gap,
    )
    return build_blocks(page, gutters, space_ratio)


def crosses_gutter(block: Block, gutters: Sequence[Span]) -> bool:
    return any(block.x0 < g0 and block.x1 > g1 for g0, g1 in gutters)


def size_tiers(blocks: Sequence[Block], size_bin: float = 0.5) -> Counter[float]:
    weighted: Counter[float] = Counter()
    for b in blocks:
        weighted[round(b.size / size_bin) * size_bin] += b.n_chars
    return weighted

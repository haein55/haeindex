import math
from collections import Counter
from collections.abc import Sequence

from haeindex.load_pdf import PDFChar

Span = tuple[float, float]


def find_gutters(
    spans: Sequence[Span],
    extent: float,
    bins: int = 160,
    min_gap: float = 3.0,
    occupancy: float = 0.0,
) -> list[Span]:
    if not spans or extent <= 0:
        return []
    bin_w = extent / bins
    counts = [0] * bins
    for x0, x1 in spans:
        lo = max(0, int(x0 / bin_w))
        hi = min(bins - 1, int(x1 / bin_w))
        for i in range(lo, hi + 1):
            counts[i] += 1
    min_count = max(1, math.ceil(occupancy * len(spans)))
    covered = [c >= min_count for c in counts]

    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for i, filled in enumerate(covered):
        if not filled and start is None:
            start = i
        elif filled and start is not None:
            gaps.append((start, i - 1))
            start = None
    if start is not None:
        gaps.append((start, bins - 1))

    return [
        (a * bin_w, (b + 1) * bin_w)
        for a, b in gaps
        if a > 0 and b < bins - 1 and (b - a + 1) * bin_w >= min_gap
    ]


def column_of(x_center: float, gutters: Sequence[Span]) -> int:
    return sum(1 for _, g1 in gutters if x_center >= g1)


def in_band(char: PDFChar, page_height: float, header: float, footer: float) -> bool:
    return header * page_height <= char.top <= footer * page_height


def page_gutters(
    chars: Sequence[PDFChar],
    width: float,
    height: float,
    *,
    header: float = 0.0,
    footer: float = 1.0,
    bins: int = 160,
    min_gap: float = 3.0,
) -> list[Span]:
    body = [c for c in chars if in_band(c, height, header, footer)]
    if not body:
        return []
    return find_gutters([(c.x0, c.x1) for c in body], width, bins=bins, min_gap=min_gap)


def column_counts(
    pages: Sequence,
    *,
    header: float = 0.0,
    footer: float = 1.0,
    bins: int = 160,
    min_gap: float = 3.0,
) -> Counter[int]:
    out: Counter[int] = Counter()
    for p in pages:
        g = page_gutters(
            p.chars, p.width, p.height, header=header, footer=footer, bins=bins, min_gap=min_gap
        )
        out[len(g) + 1] += 1
    return out


def stable_columns(counts: Counter[int], min_share: float = 0.5) -> int:
    total = sum(counts.values())
    if not total:
        return 1
    n, hits = counts.most_common(1)[0]
    return n if hits / total >= min_share else 1

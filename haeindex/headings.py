import re
from collections import Counter
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from haeindex.blocks import Block

WEIGHT_RANKS: tuple[tuple[str, int], ...] = (
    ("ultralight", 0),
    ("extralight", 0),
    ("thin", 0),
    ("light", 1),
    ("regular", 2),
    ("roman", 2),
    ("book", 2),
    ("normal", 2),
    ("regu", 2),
    ("semibold", 4),
    ("demibold", 4),
    ("demi", 4),
    ("medium", 3),
    ("medi", 3),
    ("extrabold", 6),
    ("ultrabold", 6),
    ("black", 6),
    ("heavy", 6),
    ("bold", 5),
    ("lt", 1),
    ("rg", 2),
    ("md", 3),
    ("bd", 5),
)
BOLD_RANK = 5
MAX_ORDINAL = 200
MIN_OBSERVED = 3
MIN_CONTIGUITY = 0.5
MIN_SEQUENCE_PAGES = 3
MIN_RUN = 8
MAX_HEADING_CHARS = 80

NUMBER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^제\s*(\d+)\s*조(?![의에를은는이가와과])", "제N조"),
    (r"^제\s*(\d+)\s*장(?![의에를은는이가와과])", "제N장"),
    (r"^제\s*(\d+)\s*절(?![의에를은는이가와과])", "제N절"),
    (r"^(\d+)\.(\d+)", "N.N"),
    (r"^(\d+)\.\s+\S", "N."),
    (r"^(?:CHAPTER|Chapter)\s+(\d+)", "Chapter N"),
    (r"^(?:SECTION|Section)\s+(\d+)", "Section N"),
    (r"^(\d+)\s*단계", "N단계"),
)

ONLY_SYMBOL = re.compile(r"^\s*(?:\d+|[ivxlcIVXLC]+|[①-⑳]|[·•▪▶–—-])\s*$")
_DOUBLED = re.compile(r"([^\W\d_])\1")
_DOUBLED_ALL = re.compile(r"([^\W_])\1")


def undouble(text: str, digits: bool = True) -> str:
    return (_DOUBLED_ALL if digits else _DOUBLED).sub(r"\1", text)


def font_weight(fontname: str) -> int | None:
    name = fontname.split("+")[-1].lower()
    for word, rank in WEIGHT_RANKS:
        if word in name:
            return rank
    return None


def weight_coverage(blocks: Sequence[Block]) -> float:
    if not blocks:
        return 0.0
    known = sum(1 for b in blocks if font_weight(b.fontname) is not None)
    return known / len(blocks)


def size_weights(blocks: Sequence[Block], size_bin: float = 0.5) -> Counter[float]:
    weighted: Counter[float] = Counter()
    for b in blocks:
        weighted[round(b.size / size_bin) * size_bin] += b.n_chars
    return weighted


def modal_size(blocks: Sequence[Block], size_bin: float = 0.5) -> float:
    weighted = size_weights(blocks, size_bin)
    return weighted.most_common(1)[0][0] if weighted else 0.0


def body_ceiling(blocks: Sequence[Block], cover: float = 0.8, size_bin: float = 0.5) -> float:
    weighted = size_weights(blocks, size_bin)
    total = sum(weighted.values())
    if not total:
        return 0.0
    taken = 0
    ceiling = 0.0
    for size, chars in weighted.most_common():
        taken += chars
        ceiling = max(ceiling, size)
        if taken / total >= cover:
            break
    return ceiling


def repeated_texts(blocks: Sequence[Block], min_pages: int = 3) -> set[tuple[int, int]]:
    seen: dict[str, list[tuple[int, int]]] = {}
    for b in sorted(blocks, key=lambda b: (b.page_index, b.order)):
        key = re.sub(r"\s+", "", b.text).lower()
        if key:
            seen.setdefault(key, []).append((b.page_index, b.order))
    return {
        loc for locs in seen.values() if len({p for p, _ in locs}) >= min_pages for loc in locs[1:]
    }


class Heading(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_index: int
    order: int
    text: str
    size: float
    fontname: str
    signal: str
    level: int = 0


class SeqStat(BaseModel):
    model_config = ConfigDict(frozen=True)

    pattern: str
    observed: int
    max_ordinal: int
    pages: int
    longest_run: int
    missing: list[int]

    @property
    def contiguity(self) -> float:
        return self.observed / self.max_ordinal if self.max_ordinal else 0.0

    @property
    def usable(self) -> bool:
        return (
            self.observed >= MIN_OBSERVED
            and self.max_ordinal <= MAX_ORDINAL
            and self.contiguity >= MIN_CONTIGUITY
            and self.pages >= MIN_SEQUENCE_PAGES
            and self.longest_run >= MIN_RUN
        )


def longest_increasing_run(seq: Sequence[int]) -> int:
    dedup = [n for i, n in enumerate(seq) if i == 0 or n != seq[i - 1]]
    if not dedup:
        return 0
    best = cur = 1
    for a, b in zip(dedup, dedup[1:], strict=False):
        cur = cur + 1 if b > a else 1
        best = max(best, cur)
    return best


def heading_sequences(blocks: Sequence[Block], fix_doubled: bool = False) -> list[SeqStat]:
    ordered = sorted(blocks, key=lambda b: (b.page_index, b.order))
    out: list[SeqStat] = []
    for pattern, name in NUMBER_PATTERNS:
        rx = re.compile(pattern)
        hits: dict[int, set[int]] = {}
        run: list[int] = []
        for b in ordered:
            text = undouble(b.text) if fix_doubled else b.text
            m = rx.match(text.strip())
            if m and m.group(1).isdigit():
                n = int(m.group(1))
                if 1 <= n <= MAX_ORDINAL:
                    hits.setdefault(n, set()).add(b.page_index)
                    run.append(n)
        if not hits:
            continue
        mx = max(hits)
        stat = SeqStat(
            pattern=name,
            observed=len(hits),
            max_ordinal=mx,
            pages=len({p for ps in hits.values() for p in ps}),
            longest_run=longest_increasing_run(run),
            missing=[i for i in range(1, mx + 1) if i not in hits],
        )
        if stat.usable:
            out.append(stat)
    return sorted(out, key=lambda s: (-s.longest_run, -s.observed))


def _candidate(block: Block, drop: set[tuple[int, int]]) -> bool:
    text = block.text.strip()
    if not text or ONLY_SYMBOL.match(text) or len(text) > MAX_HEADING_CHARS:
        return False
    return (block.page_index, block.order) not in drop


def detect_by_font(
    blocks: Sequence[Block],
    *,
    ceiling: float,
    size_ratio: float = 1.05,
    drop: set[tuple[int, int]] | None = None,
) -> list[Heading]:
    drop = drop or set()
    out: list[Heading] = []
    for b in blocks:
        if not _candidate(b, drop):
            continue
        weight = font_weight(b.fontname)
        if b.size >= ceiling * size_ratio:
            signal = "size"
        elif weight is not None and weight >= BOLD_RANK:
            signal = "bold"
        else:
            continue
        out.append(
            Heading(
                page_index=b.page_index,
                order=b.order,
                text=b.text,
                size=b.size,
                fontname=b.fontname,
                signal=signal,
            )
        )
    return out


def detect_by_numbering(
    blocks: Sequence[Block],
    pattern: str,
    *,
    fix_doubled: bool = False,
    drop: set[tuple[int, int]] | None = None,
) -> list[Heading]:
    drop = drop or set()
    rx = re.compile(next(p for p, name in NUMBER_PATTERNS if name == pattern))
    out: list[Heading] = []
    for b in blocks:
        if not _candidate(b, drop):
            continue
        text = undouble(b.text) if fix_doubled else b.text
        m = rx.match(text.strip())
        if m and m.group(1).isdigit() and 1 <= int(m.group(1)) <= MAX_ORDINAL:
            out.append(
                Heading(
                    page_index=b.page_index,
                    order=b.order,
                    text=text,
                    size=b.size,
                    fontname=b.fontname,
                    signal=f"num:{pattern}",
                )
            )
    return out


def assign_levels(headings: Sequence[Heading], size_bin: float = 0.5) -> list[Heading]:
    tiers = sorted({round(h.size / size_bin) * size_bin for h in headings}, reverse=True)
    rank = {t: i + 1 for i, t in enumerate(tiers)}
    return [
        h.model_copy(update={"level": rank[round(h.size / size_bin) * size_bin]}) for h in headings
    ]
